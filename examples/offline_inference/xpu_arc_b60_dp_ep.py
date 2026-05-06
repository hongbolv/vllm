# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Data Parallel + Expert Parallel inference on 4x Intel ARC B60 GPUs.

This script demonstrates running a Mixture-of-Experts (MoE) model using:
  - Tensor Parallelism (TP=2): each model replica is split across 2 GPUs
  - Data Parallelism (DP=2): 2 independent replicas process different data
  - Expert Parallelism (EP=true): MoE experts distributed across all 4 GPUs
    (EP size = DP × TP = 4; each GPU holds N/4 experts)

Total GPUs required: TP × DP = 2 × 2 = 4 (one per ARC B60 card)

Hardware Requirements:
  - 4x Intel ARC B60 GPUs
  - Intel GPU driver installed
  - PyTorch with XPU support (torch >= 2.8 recommended for xccl backend)

Launch with torchrun (single-node, 4 XPUs: TP=2, DP=2, EP=True):

  torchrun --nproc-per-node=4 \
      examples/offline_inference/xpu_arc_b60_dp_ep.py

torchrun spawns 4 processes (WORLD_SIZE=4 = TP x DP = 2 x 2):

  RANK 0: dp_rank=0, tp_rank=0, ep_rank=0  (DP group 0, holds experts 0..N/4-1)
  RANK 1: dp_rank=0, tp_rank=1, ep_rank=1  (DP group 0, holds experts N/4..N/2-1)
  RANK 2: dp_rank=1, tp_rank=0, ep_rank=2  (DP group 1, holds experts N/2..3N/4-1)
  RANK 3: dp_rank=1, tp_rank=1, ep_rank=3  (DP group 1, holds experts 3N/4..N-1)

EP status: With --enable-expert-parallel, EP size = DP × TP = 4. All 4 GPUs
participate in expert parallelism — each holds a unique N/4 subset of experts.
Tokens are routed across all ranks via all-to-all communication.

Key rule: TP partners (same dp_rank) MUST process the SAME prompt subset.
vllm dp_rank = RANK // tensor_parallel_size  (i.e. RANK // 2)

torchrun does NOT set ZE_AFFINITY_MASK, so all processes share a unified
Level Zero device namespace and XCCL IPC works correctly. This avoids the
cross-affinity IPC failure that occurs with the multiprocessing launch path
(see PR #15 for details on the limitation).
"""

import argparse

# ----- configuration --------------------------------------------------------
MODEL_PATH = "/models/Qwen3.5-35B-A3B"
TP_SIZE = 2
DP_SIZE = 2
MAX_MODEL_LEN = 256
GPU_MEMORY_UTILIZATION = 0.95
# Qwen3.5-35B-A3B is a ConditionalGeneration (multimodal) model.
# For text-only inference, language_model_only=True is required.
LANGUAGE_MODEL_ONLY = True
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-35B-A3B on 4x Intel ARC B60: TP=2, DP=2, EP=True"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Model name or path (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=TP_SIZE,
        help=f"Tensor parallel size (default: {TP_SIZE})",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=DP_SIZE,
        help=f"Data parallel size (default: {DP_SIZE})",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MAX_MODEL_LEN,
        help=f"Maximum model length (default: {MAX_MODEL_LEN})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEMORY_UTILIZATION,
        help=f"GPU memory utilization (default: {GPU_MEMORY_UTILIZATION})",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Enforce eager mode (default: True)",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        default=LANGUAGE_MODEL_ONLY,
        help="Use language model only, skip multimodal components "
        "(default: True for Qwen3.5-35B-A3B)",
    )
    return parser.parse_args()


# ----- sample prompts --------------------------------------------------------
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Explain quantum computing in simple terms:",
    "What is the difference between machine learning and deep learning?",
    "Write a short poem about the ocean:",
    "Describe the process of photosynthesis:",
]


if __name__ == "__main__":
    args = parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        data_parallel_size=args.dp_size,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        language_model_only=args.language_model_only,
        dtype="float16",
        num_gpu_blocks_override=100,
        disable_log_stats=True,
    )

    dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
    dp_size = llm.llm_engine.vllm_config.parallel_config.data_parallel_size

    # Split prompts across DP ranks
    my_prompts = [p for i, p in enumerate(PROMPTS) if i % dp_size == dp_rank]
    if not my_prompts:
        my_prompts = ["Placeholder"]

    print(f"[ARC B60] DP rank {dp_rank} processing {len(my_prompts)} prompts")

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=64,
    )

    outputs = llm.generate(my_prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"[ARC B60] DP rank {dp_rank}, "
            f"Prompt: {prompt!r}, "
            f"Generated: {generated_text!r}"
        )
