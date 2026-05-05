# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Online serving with vLLM on 4x Intel ARC B60 GPUs.

Launches a vLLM OpenAI-compatible server with:
  - Tensor Parallelism (TP=2): model split across 2 GPUs per replica
  - Data Parallelism (DP=2): 2 replicas for higher throughput
  - Expert Parallelism (EP=true): MoE experts distributed across TP ranks

Hardware: 4x Intel ARC B60 GPUs

Usage:
    # Start the server:
    bash examples/online_serving/xpu_arc_b60_serve.sh

    # Or with custom options:
    bash examples/online_serving/xpu_arc_b60_serve.sh --max-model-len 2048

    # Query the server (in another terminal):
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ibm-research/PowerMoE-3b",
            "prompt": "The future of AI is",
            "max_tokens": 64
        }'
"""
# This file serves as documentation. See the companion shell script
# xpu_arc_b60_serve.sh for the actual launch command.
