"""Paged KV cache -- PagedAttention-style block memory management.

Instead of one contiguous, growing KV tensor per request, the cache is broken
into fixed-size BLOCKS drawn from a shared pool. A request holds a list of block
ids (its "block table"); blocks need not be contiguous in memory. This removes
padding waste and per-step reallocation, and lets freed blocks be reused.

This module is PLAIN PYTHON systems logic (allocator + block tables).
The tensor gather for attention lives in model.py.
"""

from vkllm.logger import get_logger

log = get_logger(__name__)


class BlockPool:
    """A shared pool of fixed-size KV blocks, managed with a free-list.

    allocate() hands out a free block id; free() returns one to the pool.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # TODO(you): a free-list of available block ids.
        #   Start with ALL block ids free: 0, 1, ..., num_blocks - 1.
        #   A simple list works well as a stack (append/pop).
        # all blocks start free: 0 .. num_blocks-1
        self.free_blocks: list[int] = list(range(num_blocks))

    def allocate(self) -> int:
        """Hand out one free block id. Raise if the pool is exhausted."""
        # exhausted = no free blocks left. The scheduler will later treat this
        # as backpressure (don't admit more work) rather than a crash.
        if not self.free_blocks:
            raise RuntimeError("KV cache full")
        return self.free_blocks.pop()   # pop() from the end = O(1)

    def free(self, block_id: int) -> None:
        """Return one block id to the pool for reuse."""
        # TODO(you): put block_id back into the free-list.
        self.free_blocks.append(block_id)

    def free_many(self, block_ids: list[int]) -> None:
        """Return several blocks (e.g. all of a finished request's blocks)."""
        for b in block_ids:
            self.free(b)

    def num_free(self) -> int:
        # TODO(you): how many blocks are currently free?
        return len(self.free_blocks)


class BlockTable:
    """Per-request map from logical token position -> physical block slots.

    A request thinks it has a contiguous sequence of `length` tokens, but the
    K/V actually live in scattered pool blocks listed in `block_ids`.
    """

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_size = pool.block_size
        self.block_ids: list[int] = []   # physical blocks this request owns, in order
        self.length = 0                  # number of tokens actually stored

    def _has_room(self) -> bool:
        """Is there a free slot in the current last block?"""
        # TODO(you): there is room iff length is NOT a multiple of block_size
        #   (and we have at least one block). Equivalently: capacity > length.
        #   capacity = len(block_ids) * block_size
        capacity = len(self.block_ids) * self.block_size
        return capacity > self.length

    def append_token(self) -> tuple[int, int]:
        """Reserve space for ONE more token. Returns (block_id, slot) to write into.

        Grabs a new block from the pool if the current blocks are full.
        """
        # TODO(you):
        #   1. if there's no room (length == capacity), allocate a new block from
        #      self.pool and append its id to self.block_ids.
        #   2. compute where this token goes:
        #        block_index_in_table = self.length // self.block_size
        #        slot                 = self.length % self.block_size
        #        block_id             = self.block_ids[block_index_in_table]
        #   3. increment self.length
        #   4. return (block_id, slot)
        if not self._has_room():
            self.block_ids.append(self.pool.allocate())

        block_index_in_table = self.length // self.block_size
        slot                 = self.length % self.block_size
        block_id             = self.block_ids[block_index_in_table]

        self.length += 1

        return (block_id, slot)


    def free(self) -> None:
        """Return all of this request's blocks to the pool (on retirement)."""
        # TODO(you): give every block back to the pool, then clear our list.
        self.pool.free_many(self.block_ids)
        self.block_ids = []
        self.length = 0



import torch


class PagedKVCache:
    """Physical block storage for ALL layers + write/gather ops.

    The BlockPool/BlockTable track WHICH blocks (integer ids). This class holds
    the actual tensors and reads/writes K/V at (block_id, slot). One shared,
    pre-allocated slab per layer -- allocated ONCE, never grows.

    Storage shape per layer: (num_blocks, block_size, kv_heads, head_dim)
    """

    def __init__(self, num_layers, num_blocks, block_size, kv_heads, head_dim, device="cpu"):
        self.block_size = block_size
        self.pool = BlockPool(num_blocks, block_size)
        # one big K slab and V slab per layer -- allocated once here.
        self.k_store = [
            torch.zeros(num_blocks, block_size, kv_heads, head_dim, device=device)
            for _ in range(num_layers)
        ]
        self.v_store = [
            torch.zeros(num_blocks, block_size, kv_heads, head_dim, device=device)
            for _ in range(num_layers)
        ]

    def new_block_table(self) -> BlockTable:
        """A fresh block table for one request (shared across its layers)."""
        return BlockTable(self.pool)

    def append(self, layer: int, block_table: BlockTable, positions, new_k, new_v) -> None:
        """Write new tokens' K/V into the request's blocks for one layer.

        new_k, new_v: (kv_heads, n_new, head_dim)   -- the new tokens for THIS layer
        positions:    the (block_id, slot) list for these tokens. For layer 0 we
                      grow the block table; later layers REUSE the same placements
                      (all layers of a request share one block table / length).
        """
        for i, (block_id, slot) in enumerate(positions):
            # new_k[:, i, :] is (kv_heads, head_dim) -> write into (block, slot)
            self.k_store[layer][block_id, slot] = new_k[:, i, :]
            self.v_store[layer][block_id, slot] = new_v[:, i, :]

    def gather(self, layer: int, block_table: BlockTable):
        """Assemble this request's scattered blocks into contiguous K/V.

        Returns (kv_heads, length, head_dim) for K and V -- exactly the shape the
        attention math expects (same as the old torch.cat cache produced).
        """
        length = block_table.length
        ids = block_table.block_ids
        # gather the owned blocks: (n_blocks, block_size, kv_heads, head_dim)
        k_blocks = self.k_store[layer][ids]     # fancy-index by block ids
        v_blocks = self.v_store[layer][ids]
        # flatten block+slot dims -> (n_blocks*block_size, kv_heads, head_dim)
        nb = len(ids)
        k_flat = k_blocks.reshape(nb * self.block_size, k_blocks.shape[2], k_blocks.shape[3])
        v_flat = v_blocks.reshape(nb * self.block_size, v_blocks.shape[2], v_blocks.shape[3])
        # keep only the real `length` tokens (drop the unused tail of last block)
        k = k_flat[:length].transpose(0, 1)     # (kv_heads, length, head_dim)
        v = v_flat[:length].transpose(0, 1)
        return k, v
