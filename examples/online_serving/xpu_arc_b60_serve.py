# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Online serving with vLLM on Intel ARC B60 GPUs.

Launches a vLLM OpenAI-compatible server with:
  - Tensor Parallelism (TP=2): model split across 2 GPUs per replica
  - Expert Parallelism (EP=true): MoE experts distributed across TP ranks

Hardware: 4x Intel ARC B60 GPUs (TP=2 uses 2 GPUs)
Default Model: /models/Qwen3.5-35B-A3B

NOTE: DP>1 with `vllm serve` on XPU is not yet supported due to XCCL
cross-ZE_AFFINITY_MASK IPC failures (see PR #15). For DP>1 offline
inference, use torchrun with the offline script instead.

Usage:
    # Start the server (TP=2, DP=1):
    bash examples/online_serving/xpu_arc_b60_serve.sh

    # Or with custom options:
    bash examples/online_serving/xpu_arc_b60_serve.sh --max-model-len 512

    # Query the server (in another terminal):
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "/models/Qwen3.5-35B-A3B",
            "prompt": "The future of AI is",
            "max_tokens": 64
        }'
"""
# This file serves as documentation. See the companion shell script
# xpu_arc_b60_serve.sh for the actual launch command.
