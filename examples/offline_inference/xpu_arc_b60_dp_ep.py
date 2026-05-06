# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Data Parallel + Expert Parallel inference on 4x Intel ARC B60 GPUs.

This script demonstrates running a Mixture-of-Experts (MoE) model using:
  - Tensor Parallelism (TP=2): each model replica is split across 2 GPUs
  - Data Parallelism (DP=2): 2 independent replicas process different data
  - Expert Parallelism (EP=true): MoE experts are distributed across TP ranks

Total GPUs required: TP × DP = 2 × 2 = 4 (one per ARC B60 card)

Hardware Requirements:
  - 4x Intel ARC B60 GPUs
  - Intel GPU driver installed
  - PyTorch with XPU support (torch >= 2.8 recommended for xccl backend)

============================================================================
Option A - multiprocessing (single-node, 4 XPUs: TP=2, DP=2, EP=True)
============================================================================

Just run:
  python examples/offline_inference/xpu_arc_b60_dp_ep.py

The script spawns 2 child processes (dp_rank 0 and 1). Each child process
runs vllm with tensor_parallel_size=2, which internally spawns 2 TP workers.
Total GPU usage: 4 XPUs (2 DP groups x 2 TP workers each).

NOTE: This mode requires the XPU DP fix that skips ZE_AFFINITY_MASK and uses
DP-adjusted local_rank offsets. Without the fix, XCCL hangs due to cross-
affinity IPC failures. See PR #15 for details.

============================================================================
Option B - torchrun (single-node, 4 XPUs: TP=2, DP=2, EP=True)
============================================================================

torchrun spawns 4 processes (WORLD_SIZE=4 = TP x DP = 2 x 2):

  torchrun --nproc-per-node=4 \
      examples/offline_inference/xpu_arc_b60_dp_ep.py --torchrun

Process layout:
  RANK 0: vllm dp_rank=0, tp_rank=0   (DP group 0, TP leader)
  RANK 1: vllm dp_rank=0, tp_rank=1   (DP group 0, TP follower)
  RANK 2: vllm dp_rank=1, tp_rank=0   (DP group 1, TP leader)
  RANK 3: vllm dp_rank=1, tp_rank=1   (DP group 1, TP follower)

Key rule: TP partners (same dp_rank) MUST process the SAME prompt subset.
vllm dp_rank = RANK // tensor_parallel_size  (i.e. RANK // 2)

NOTE: torchrun mode does NOT set ZE_AFFINITY_MASK, so all processes share
a unified Level Zero device namespace and XCCL IPC works correctly.
However, torchrun is NOT compatible with `vllm serve` (OpenAI API server).
"""

import argparse
import os
import sys
from multiprocessing import Process
from time import sleep

# ----- configuration --------------------------------------------------------
MODEL_PATH = "/home/media/Hongbo/models/Qwen3.5-35B-A3B"
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
        "--torchrun",
        action="store_true",
        help="Use torchrun launch mode (Option B)",
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
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Number of seconds before unresponsive process is killed.",
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


# ----- Option A: multiprocessing worker -------------------------------------
def run_dp_worker(dp_rank, dp_size, args):
    """Worker function for Option A (multiprocessing launch)."""
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"

    from vllm import LLM, SamplingParams

    # Split prompts across DP ranks
    my_prompts = [p for i, p in enumerate(PROMPTS) if i % dp_size == dp_rank]
    if not my_prompts:
        my_prompts = ["Placeholder"]

    print(
        f"[ARC B60] DP rank {dp_rank} processing "
        f"{len(my_prompts)} prompts with TP={args.tp_size}, EP=true"
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=64,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        enable_expert_parallel=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        language_model_only=args.language_model_only,
        dtype="float16",
        num_gpu_blocks_override=100,
        disable_log_stats=True,
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

    print(f"[ARC B60] DP rank {dp_rank} completed {len(outputs)} generations.")

    sleep(1)


# ----- Option B: torchrun worker --------------------------------------------
def run_torchrun(args):
    """Worker function for Option B (torchrun launch)."""
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
        trust_remote_code=True,
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


# ----- main ------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    if args.torchrun:
        # Option B: torchrun launch
        print("[Option B] torchrun mode")
        run_torchrun(args)
    else:
        # Option A: multiprocessing launch
        print(
            f"[Option A] multiprocessing mode: DP={args.dp_size}, "
            f"TP={args.tp_size}, EP=True"
        )
        print(f"  Model: {args.model}")

        # Set shared master port for all DP workers
        from vllm.utils.network_utils import get_open_port

        master_port = str(get_open_port())
        os.environ["VLLM_DP_MASTER_PORT"] = master_port

        procs = []
        for dp_rank in range(args.dp_size):
            proc = Process(
                target=run_dp_worker,
                args=(dp_rank, args.dp_size, args),
            )
            proc.start()
            procs.append(proc)

        exit_code = 0
        for proc in procs:
            proc.join(timeout=args.timeout)
            if proc.exitcode is None:
                print(f"Killing process {proc.pid} (timed out after {args.timeout}s)")
                proc.kill()
                exit_code = 1
            elif proc.exitcode:
                exit_code = proc.exitcode

        if exit_code == 0:
            print("\n" + "=" * 60)
            print("  All DP ranks completed successfully!")
            print("=" * 60)

        sys.exit(exit_code)
