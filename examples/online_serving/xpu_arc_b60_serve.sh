#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch vLLM OpenAI-compatible server on 4x Intel ARC B60 GPUs
# Configuration: TP=2, DP=2, EP=true
#
# Prerequisites:
#   - 4x Intel ARC B60 GPUs with drivers installed
#   - vLLM installed with XPU support:
#       VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
#   - triton-xpu installed:
#       pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu
#
# Usage:
#   bash examples/online_serving/xpu_arc_b60_serve.sh
#   bash examples/online_serving/xpu_arc_b60_serve.sh --max-model-len 2048
#   bash examples/online_serving/xpu_arc_b60_serve.sh --model "your-moe-model"

set -euo pipefail

MODEL="${MODEL:-ibm-research/PowerMoE-3b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
PORT="${PORT:-8000}"

echo "============================================================"
echo "  vLLM Server on 4x Intel ARC B60 GPUs"
echo "  Model: ${MODEL}"
echo "  Config: TP=2, DP=2, EP=true"
echo "  Port: ${PORT}"
echo "============================================================"

vllm serve "${MODEL}" \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --distributed-executor-backend mp \
    --enforce-eager \
    --port "${PORT}" \
    "$@"
