"""Centralized logging for VKLLM.

Thin wrapper over Python's stdlib `logging` so every module shares one
consistent format and level. This is the first taste of observability
(Phase 10): being able to SEE requests flow through the engine.

Usage:
    from vkllm.logger import get_logger
    log = get_logger(__name__)
    log.info("request %s admitted", req_id)

Control verbosity with the VKLLM_LOG_LEVEL env var (DEBUG/INFO/WARNING/...).
Default is INFO.
"""

import logging
import os

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("VKLLM_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("vkllm")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False   # don't double-log through the root logger
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the shared 'vkllm' logger."""
    _configure_once()
    # normalize "vkllm.scheduler" style names; if given __name__ that already
    # starts with vkllm, use as-is, else nest it under vkllm.
    if not name.startswith("vkllm"):
        name = f"vkllm.{name}"
    return logging.getLogger(name)
