# SPDX-License-Identifier: Apache-2.0
# Demo: Qwen3-30B-A3B on Intel Arc Pro B60 with TP=2, DP=2, EP=True
#
# With DP=2 and EP=True, each XPU device acts as a separate data-parallel
# rank while experts are distributed across ranks via alltoall (AgRs backend).
#
# ============================================================================
# Option A - multiprocessing (single-node, 4 XPUs: TP=2, DP=2, EP=True)
# ============================================================================
#
# Just run:
#   python examples/xpu_qwen_moe_dp_ep_demo.py
#
# The script spawns 2 child processes (dp_rank 0 and 1). Each child process
# runs vllm with tensor_parallel_size=2, which internally spawns 2 TP workers.
# Total GPU usage: 4 XPUs (2 DP groups × 2 TP workers each).
#
# ============================================================================
# Option B - torchrun (single-node, 4 XPUs: TP=2, DP=2, EP=True)
# ============================================================================
#
# torchrun spawns 4 processes (WORLD_SIZE=4 = TP×DP = 2×2):
#
#   torchrun --nproc-per-node=4 \
#       examples/xpu_qwen_moe_dp_ep_demo.py --torchrun
#
# Process layout:
#   RANK 0: vllm dp_rank=0, tp_rank=0   (DP group 0, TP leader)
#   RANK 1: vllm dp_rank=0, tp_rank=1   (DP group 0, TP follower)
#   RANK 2: vllm dp_rank=1, tp_rank=0   (DP group 1, TP leader)
#   RANK 3: vllm dp_rank=1, tp_rank=1   (DP group 1, TP follower)
#
# Key rule: TP partners (same dp_rank) MUST process the SAME prompt subset.
# vllm dp_rank = RANK // tensor_parallel_size  (i.e. RANK // 2)
#
# ============================================================================
# Option C - torchrun (multi-node, 2 nodes × 2 XPUs: TP=2, DP=2, EP=True)
# ============================================================================
#
# Suppose you have:
#   Node 0 (master): IP = 192.168.1.100, has 2 XPUs  (RANK 0, 1)
#   Node 1:          IP = 192.168.1.101, has 2 XPUs  (RANK 2, 3)
#
# Step 1 - On Node 0 (master), run:
#   torchrun \
#       --nnodes=2 \
#       --node-rank=0 \
#       --nproc-per-node=2 \
#       --master-addr=192.168.1.100 \
#       --master-port=29500 \
#       examples/xpu_qwen_moe_dp_ep_demo.py --torchrun
#
# Step 2 - On Node 1, run:
#   torchrun \
#       --nnodes=2 \
#       --node-rank=1 \
#       --nproc-per-node=2 \
#       --master-addr=192.168.1.100 \
#       --master-port=29500 \
#       examples/xpu_qwen_moe_dp_ep_demo.py --torchrun
#
# Both nodes must use the SAME --master-addr and --master-port.
# torchrun on each node waits until all nodes connect before starting.
#
# Process layout across nodes (same as Option B):
#   Node 0 RANK 0: vllm dp_rank=0, tp_rank=0
#   Node 0 RANK 1: vllm dp_rank=0, tp_rank=1
#   Node 1 RANK 2: vllm dp_rank=1, tp_rank=0
#   Node 1 RANK 3: vllm dp_rank=1, tp_rank=1
#
# Key torchrun flags:
#   --nnodes          Total number of nodes (machines)
#   --node-rank       This node's rank (0-indexed)
#   --nproc-per-node  Number of processes (XPUs) on THIS node
#   --master-addr     IP address of node-rank=0 (the master)
#   --master-port     Port for rank coordination (must be open/same on all nodes)
# ============================================================================

import os
import sys
from time import sleep

# Triton is not available on XPU; disable torch.compile to avoid errors
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# XPU model loading can be slow; increase the engine startup timeout
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1800")
# Workaround: fall back to UR Level Zero v1 adapter for stability (avoid v2 crash)
os.environ.setdefault("SYCL_UR_USE_LEVEL_ZERO_V2", "0")
# Workaround for Intel GPU driver (NEO Compute Runtime) buffer compression bug:
# dist.all_gather on xccl returns corrupted data for non-16-byte-aligned buffers
# when render compression is enabled. Disabling compression fixes this.
os.environ.setdefault("NEOReadDebugKeys", "1")
os.environ.setdefault("EnableImplicitScaling", "0")
os.environ.setdefault("RenderCompressedBuffersEnabled", "0")

MODEL_PATH = "/home/media/Hongbo/models/Qwen3-30B-A3B"

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a short poem about artificial intelligence:",
]


def run_dp_rank(
    dp_size: int,
    tp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    use_torchrun: bool = False,
    should_print_results: bool = True,
):
    """Worker function executed by each DP rank."""
    from vllm import LLM, SamplingParams

    if not use_torchrun:
        # Set DP coordination env vars (multiprocessing mode)
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    if should_print_results:
        print("=" * 60)
        print(
            f"[DP rank {global_dp_rank}] Qwen3-30B-A3B  "
            f"TP={tp_size}, DP={dp_size}, EP=True"
        )
        print(f"[DP rank {global_dp_rank}] Device: Intel Arc Pro B60 (XPU)")
        print("=" * 60)

    engine_kwargs = dict(
        model=MODEL_PATH,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=True,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=256,
        enforce_eager=True,
        swap_space=4,
        gpu_memory_utilization=0.95,
        num_gpu_blocks_override=100,
    )

    if use_torchrun:
        # torchrun mode: pass DP size explicitly + external_launcher backend
        engine_kwargs["data_parallel_size"] = dp_size
        engine_kwargs["distributed_executor_backend"] = "external_launcher"

    llm = LLM(**engine_kwargs)

    # Each DP rank processes a different subset of prompts.
    # In torchrun mode with TP>1, TP partners share the same dp_rank and
    # MUST process the same prompt subset to keep TP allreduce in sync.
    my_prompts = [
        p for i, p in enumerate(PROMPTS) if i % dp_size == global_dp_rank
    ]
    if not my_prompts:
        my_prompts = ["Placeholder"]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    if should_print_results:
        print(
            f"\n[DP rank {global_dp_rank}] Generating {len(my_prompts)} responses...\n"
        )
    outputs = llm.generate(my_prompts, sampling_params)

    if should_print_results:
        print("=" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"[DP rank {global_dp_rank}] Prompt:    {prompt!r}")
            print(f"[DP rank {global_dp_rank}] Response:  {generated_text!r}")
            print("-" * 60)

        print(f"\n[DP rank {global_dp_rank}] Done!")
    sleep(1)


def main():
    use_torchrun = "--torchrun" in sys.argv
    tp_size = 2   # tensor_parallel_size used throughout this demo
    dp_size = 2   # data_parallel_size

    if use_torchrun:
        # torchrun mode: WORLD_SIZE = TP × DP = 4
        # torchrun assigns RANK/LOCAL_RANK/WORLD_SIZE env vars.
        #
        # Process layout (vllm's internal assignment):
        #   vllm dp_rank = RANK // tp_size
        #   vllm tp_rank = RANK %  tp_size
        #
        #   RANK 0 → dp_rank=0, tp_rank=0  (TP leader  of DP group 0)
        #   RANK 1 → dp_rank=0, tp_rank=1  (TP follower of DP group 0)
        #   RANK 2 → dp_rank=1, tp_rank=0  (TP leader  of DP group 1)
        #   RANK 3 → dp_rank=1, tp_rank=1  (TP follower of DP group 1)
        #
        # TP partners share the same dp_rank and MUST receive the SAME
        # prompt subset so that TP allreduce tensors stay in sync.
        # Only the TP leader (tp_rank==0) prints results.
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", tp_size * dp_size))

        expected_world_size = tp_size * dp_size
        if world_size != expected_world_size:
            print(
                f"[ERROR] WORLD_SIZE={world_size} but TP={tp_size} × DP={dp_size}"
                f" requires WORLD_SIZE={expected_world_size}.\n"
                f"  Re-run with: torchrun --nproc-per-node={expected_world_size} ...",
                file=sys.stderr,
            )
            sys.exit(1)

        # vllm computes dp_rank = RANK // (WORLD_SIZE // dp_size) = RANK // tp_size
        vllm_dp_rank = rank // tp_size
        vllm_tp_rank = rank % tp_size

        # Only the TP leader (tp_rank==0) per DP group prints results.
        # TP followers participate in collective ops but produce duplicate output;
        # pass should_print_results=False so they silently run generate() without
        # printing. Do NOT redirect sys.stdout -- vllm calls sys.stdout.fileno()
        # internally (e.g. in suppress_stdout()) which raises UnsupportedOperation
        # on io.StringIO objects.
        run_dp_rank(
            dp_size=dp_size,
            tp_size=tp_size,
            local_dp_rank=local_rank,
            global_dp_rank=vllm_dp_rank,
            dp_master_ip="",
            dp_master_port=0,
            use_torchrun=True,
            should_print_results=(vllm_tp_rank == 0),
        )
    else:
        # Multiprocessing mode: spawn one process per DP rank.
        # vllm will spawn TP workers internally within each DP process.
        # Must use 'spawn' on XPU to avoid re-initialization errors.
        import multiprocessing
        from multiprocessing import Process

        multiprocessing.set_start_method("spawn", force=True)

        from vllm.utils.network_utils import get_open_port

        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()

        procs = []
        for rank in range(dp_size):
            proc = Process(
                target=run_dp_rank,
                args=(dp_size, tp_size, rank, rank,
                      dp_master_ip, dp_master_port, False),
            )
            proc.start()
            procs.append(proc)

        exit_code = 0
        for proc in procs:
            proc.join(timeout=600)
            if proc.exitcode is None:
                print(f"Killing process {proc.pid} (timed out)")
                proc.kill()
                exit_code = 1
            elif proc.exitcode:
                exit_code = proc.exitcode

        sys.exit(exit_code)


if __name__ == "__main__":
    main()
