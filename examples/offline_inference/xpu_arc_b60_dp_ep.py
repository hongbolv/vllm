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

Usage:
    python examples/offline_inference/xpu_arc_b60_dp_ep.py

    # With custom model (must be a MoE model for EP):
    python examples/offline_inference/xpu_arc_b60_dp_ep.py \
        --model="ibm-research/PowerMoE-3b"

    # Adjust max model length for memory constraints:
    python examples/offline_inference/xpu_arc_b60_dp_ep.py \
        --model="ibm-research/PowerMoE-3b" \
        --max-model-len=1024

Environment:
    The script automatically sets ZE_AFFINITY_MASK for each DP rank to control
    which Intel XPU devices are visible to each process. For 4 ARC B60 GPUs:
      - DP rank 0 sees GPUs 0,1 (for TP=2)
      - DP rank 1 sees GPUs 2,3 (for TP=2)
"""

import os
from time import sleep

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port


def create_parser():
    parser = FlexibleArgumentParser(
        description="Data Parallel + Expert Parallel on 4x Intel ARC B60"
    )

    # Add all engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="ibm-research/PowerMoE-3b",
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        # ARC B60 has limited memory; set a conservative default
        max_model_len=1024,
        # Use enforce_eager by default since XPU graph support is limited
        enforce_eager=True,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Number of seconds before unresponsive process is killed.",
    )

    return parser


def main(
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    engine_args,
):
    """Main function for each DP rank process."""
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Sample prompts for inference
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Intel ARC GPUs are designed for",
        "Machine learning inference requires",
        "The advantage of expert parallelism is",
        "Data parallel processing allows",
    ] * 10

    # Distribute prompts across DP ranks
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(prompts) == 0:
        prompts = ["Placeholder"]

    print(
        f"[ARC B60] DP rank {global_dp_rank} processing "
        f"{len(prompts)} prompts with TP=2, EP=true"
    )

    # Different sampling params per rank for demonstration
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=[16, 32][global_dp_rank % 2],
    )

    # Create LLM engine - each DP rank uses 2 GPUs (TP=2)
    llm = LLM(**engine_args)
    outputs = llm.generate(prompts, sampling_params)

    # Print sample outputs
    for i, output in enumerate(outputs):
        if i >= 3:
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"[ARC B60] DP rank {global_dp_rank}, "
            f"Prompt: {prompt!r}, "
            f"Generated: {generated_text!r}"
        )

    print(f"[ARC B60] DP rank {global_dp_rank} completed {len(outputs)} generations.")

    # Give engines time to pause their processing loops before exiting
    sleep(1)


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())

    # Extract DP-specific args
    dp_size = args.pop("data_parallel_size")
    timeout = args.pop("timeout")

    # Remaining args are engine args
    engine_args = args

    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    print("=" * 60)
    print("  vLLM on 4x Intel ARC B60 GPUs")
    print(
        f"  Configuration: TP={engine_args.get('tensor_parallel_size', 2)}, "
        f"DP={dp_size}, EP=true"
    )
    print(f"  Model: {engine_args.get('model', 'ibm-research/PowerMoE-3b')}")
    print(f"  Backend: {engine_args.get('distributed_executor_backend', 'mp')}")
    print("=" * 60)

    from multiprocessing import Process

    procs = []
    for local_dp_rank in range(dp_size):
        global_dp_rank = local_dp_rank
        proc = Process(
            target=main,
            args=(
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                engine_args,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop in time.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("  All DP ranks completed successfully!")
        print("=" * 60)

    exit(exit_code)
