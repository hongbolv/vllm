# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for Qwen3-30B-A3B on 4 GPUs.

Covers:
  1. Basic import verification
  2. Health check (online serving via OpenAI-compatible API)
  3. Offline inference verification

Run:
    pytest tests/smoke_test_qwen3_30b_a3b.py -v -s

Requirements:
  - 4 GPUs available
  - Qwen3-30B-A3B model weights accessible (local path or HuggingFace Hub)
"""

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
NUM_GPUS = 4

# ---------------------------------------------------------------------------
# Skip marker: all tests in this file require 4 GPUs
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    current_platform.device_count() < NUM_GPUS,
    reason=f"Need at least {NUM_GPUS} GPUs to run Qwen3-30B-A3B smoke tests.",
)

# ---------------------------------------------------------------------------
# 1. Basic import verification
# ---------------------------------------------------------------------------


def test_basic_imports():
    """Verify that core vLLM classes can be imported without error."""
    from vllm import LLM, SamplingParams  # noqa: F401
    from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: F401

    assert LLM is not None
    assert SamplingParams is not None


# ---------------------------------------------------------------------------
# 2. Health check (online serving)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Start a vllm serve process for Qwen3-30B-A3B with TP=4."""
    args = [
        "--tensor-parallel-size",
        str(NUM_GPUS),
        "--max-model-len",
        "4096",
        "--trust-remote-code",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as s:
        yield s


def test_health_check(server):
    """GET /health should return 200 OK."""
    resp = requests.get(server.url_for("health"), timeout=30)
    assert resp.status_code == 200, (
        f"Health check failed: status={resp.status_code}, body={resp.text}"
    )


def test_models_endpoint(server):
    """GET /v1/models should list the loaded model."""
    client = server.get_client()
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    assert any(MODEL_NAME in mid or mid in MODEL_NAME for mid in model_ids), (
        f"Model {MODEL_NAME!r} not found in /v1/models response: {model_ids}"
    )


def test_chat_completion(server):
    """POST /v1/chat/completions should return a non-empty response."""
    client = server.get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is 1+1? Answer briefly."}],
        max_tokens=16,
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    assert content and len(content.strip()) > 0, (
        f"Empty chat completion response: {resp}"
    )


# ---------------------------------------------------------------------------
# 3. Offline inference verification
# ---------------------------------------------------------------------------


def test_offline_inference():
    """Run offline greedy inference with LLM on 4 GPUs."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=NUM_GPUS,
        max_model_len=4096,
        trust_remote_code=True,
    )

    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts), (
        f"Expected {len(prompts)} outputs, got {len(outputs)}"
    )
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        assert generated and len(generated.strip()) > 0, (
            f"Empty output for prompt {i!r}: {output}"
        )
        # Sanity: output must not be all NaN-related tokens
        assert "nan" not in generated.lower() or len(generated.strip()) > 5, (
            f"Suspicious NaN output for prompt {i!r}: {generated!r}"
        )
