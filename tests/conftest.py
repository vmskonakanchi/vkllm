"""Shared pytest fixtures.

The model + tokenizer load once per test session (they're expensive), and every
test reuses them via these fixtures.
"""

import pytest
import torch
from transformers import AutoTokenizer

from vkllm.model import Model

MODEL_ID = "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="session")
def model():
    torch.manual_seed(0)
    return Model(MODEL_ID)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)
