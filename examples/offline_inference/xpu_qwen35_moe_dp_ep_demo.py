# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demo: Qwen3.5-35B-A3B on Intel XPU with TP=2, DP=2, EP=True

With DP=2 and EP=True, each XPU device acts as a separate data-parallel
rank while experts are distributed across ranks via alltoall.

============================================================================
Option A - multiprocessing (single-node, 4 XPUs: TP=2, DP=2, EP=True)
============================================================================

Just run:
  python examples/offline_inference/xpu_qwen35_moe_dp_ep_demo.py

The script spawns 2 child processes (dp_rank 0 and 1). Each child process
runs vllm with tensor_parallel_size=2, which internally spawns 2 TP workers.
Total GPU usage: 4 XPUs (2 DP groups x 2 TP workers each).

============================================================================
Option B - torchrun (single-node, 4 XPUs: TP=2, DP=2, EP=True)
============================================================================

torchrun spawns 4 processes (WORLD_SIZE=4 = TP x DP = 2 x 2):

  torchrun --nproc-per-node=4 \\
      examples/offline_inference/xpu_qwen35_moe_dp_ep_demo.py --torchrun

Process layout:
  RANK 0: vllm dp_rank=0, tp_rank=0   (DP group 0, TP leader)
  RANK 1: vllm dp_rank=0, tp_rank=1   (DP group 0, TP follower)
  RANK 2: vllm dp_rank=1, tp_rank=0   (DP group 1, TP leader)
  RANK 3: vllm dp_rank=1, tp_rank=1   (DP group 1, TP follower)

Key rule: TP partners (same dp_rank) MUST process the SAME prompt subset.
vllm dp_rank = RANK // tensor_parallel_size  (i.e. RANK // 2)
"""

import argparse
import os
import sys
from multiprocessing import Process
from time import sleep

# ----- configuration --------------------------------------------------------
MODEL_PATH = "/models/Qwen3.5-35B-A3B"  # local model path, adjust as needed
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
        description="Qwen3.5-35B-A3B XPU DP+EP demo"
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
    from vllm.utils.network_utils import get_open_port

    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    if "VLLM_DP_MASTER_PORT" not in os.environ:
        os.environ["VLLM_DP_MASTER_PORT"] = str(get_open_port())

    from vllm import LLM, SamplingParams

    # Split prompts across DP ranks
    my_prompts = [p for i, p in enumerate(PROMPTS) if i % dp_size == dp_rank]
    if not my_prompts:
        my_prompts = ["Placeholder"]

    print(f"[DP rank {dp_rank}] Processing {len(my_prompts)} prompts")

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
            f"[DP rank {dp_rank}] Prompt: {prompt!r}\n"
            f"  Generated: {generated_text!r}\n"
        )

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
    my_prompts = [
        f"{i}.{p}" for i, p in enumerate(PROMPTS) if i % dp_size == dp_rank
    ]
    if not my_prompts:
        my_prompts = ["Placeholder"]

    print(f"[DP rank {dp_rank}] Processing {len(my_prompts)} prompts")

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
            f"[DP rank {dp_rank}] Prompt: {prompt!r}\n"
            f"  Generated: {generated_text!r}\n"
        )


# ----- main ------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    if args.torchrun:
        # Option B: torchrun launch
        # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE before this point
        print("[Option B] torchrun mode")
        run_torchrun(args)
    else:
        # Option A: multiprocessing launch
        print(f"[Option A] multiprocessing mode: DP={args.dp_size}, "
              f"TP={args.tp_size}, EP=True")

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
            proc.join(timeout=600)
            if proc.exitcode is None:
                print(f"Killing process {proc.pid} (timed out after 600s)")
                proc.kill()
                exit_code = 1
            elif proc.exitcode:
                exit_code = proc.exitcode

        sys.exit(exit_code)
