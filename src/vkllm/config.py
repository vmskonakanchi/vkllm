from dataclasses import dataclass
import transformers

@dataclass
class ModelConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    num_layers: int = 30
    num_q_heads: int = 9
    num_kv_heads: int = 3
    head_dim: int = 64
    intermediate_size: int = 1536
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position: int = 2048


    @classmethod
    def from_pretrained(cls, model_id_or_path: str) -> "ModelConfig":
        hf = transformers.AutoConfig.from_pretrained(model_id_or_path)
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_layers=hf.num_hidden_layers,        # note: HF calls it num_hidden_layers
            num_q_heads=hf.num_attention_heads,     # HF: num_attention_heads
            num_kv_heads=hf.num_key_value_heads,
            head_dim=hf.hidden_size // hf.num_attention_heads,  # computed (see below)
            intermediate_size=hf.intermediate_size,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta = getattr(hf, "rope_theta", None) or hf.rope_parameters["rope_theta"],
            max_position=hf.max_position_embeddings, # HF: max_position_embeddings
        )
