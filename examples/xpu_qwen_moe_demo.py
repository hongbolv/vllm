# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2024 Intel Corporation
# Demo: Qwen3-30B-A3B on 4x Intel Arc Pro B60 with DP=4

import os
from multiprocessing import Process
from time import sleep

# Triton is not available on XPU; disable torch.compile to avoid errors
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# XPU model loading can be slow; increase the engine startup timeout
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1800")

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port

DP_SIZE = 4
MODEL_PATH = "/home/media/Hongbo/models/Qwen3-30B-A3B"
PROCESS_TIMEOUT_SECONDS = 300

ALL_PROMPTS = [
    "Introduce yourself and describe your capabilities as a large language model.",
    ("Explain the differences between supervised learning, unsupervised learning, "
     "and reinforcement learning, and give a real-world example for each."),
    ("You are a senior software engineer. Write a Python function that implements "
     "a binary search tree with insert, search, and in-order traversal methods. "
     "Include docstrings and type hints."),
    ("Summarize the key milestones in the history of artificial intelligence from "
     "the 1950s to today, and discuss what challenges remain before achieving "
     "artificial general intelligence."),
]


def run_dp_rank(dp_rank: int, dp_master_ip: str, dp_master_port: int) -> None:
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(DP_SIZE)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Distribute prompts evenly across DP ranks.
    floor = len(ALL_PROMPTS) // DP_SIZE
    remainder = len(ALL_PROMPTS) % DP_SIZE

    def start(rank: int) -> int:
        return rank * floor + min(rank, remainder)

    prompts = ALL_PROMPTS[start(dp_rank):start(dp_rank + 1)]
    if not prompts:
        prompts = ["Placeholder"]

    print(f"DP rank {dp_rank}: processing {len(prompts)} prompt(s)")

    # Initialize the model with DP=4, TP=1 per rank.
    # enforce_eager=True since Triton is not available on XPU.
    # Qwen3-30B-A3B default max_position_embeddings is 32768; use 4096 here
    # for a practical demo on 4x Intel Arc Pro B60 (~6 GiB VRAM/card).
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    outputs = llm.generate(prompts, sampling_params)

    print("=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[DP rank {dp_rank}] Prompt:   {prompt!r}")
        print(f"[DP rank {dp_rank}] Response: {generated_text!r}")
        print("-" * 60)

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-30B-A3B Demo with Data Parallelism = 4")
    print("Device: 4x Intel Arc Pro B60 (XPU), 1 device per DP rank")
    print("=" * 60)

    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    procs: list[Process] = []
    for dp_rank in range(DP_SIZE):
        proc = Process(
            target=run_dp_rank,
            args=(dp_rank, dp_master_ip, dp_master_port),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=PROCESS_TIMEOUT_SECONDS)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that did not finish in time.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    raise SystemExit(exit_code)
