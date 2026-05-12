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

  RANK 0: dp_rank=0, tp_rank=0, ep_rank=0  (DP group 0, experts 0..N/4-1)
  RANK 1: dp_rank=0, tp_rank=1, ep_rank=1  (DP group 0, experts N/4..N/2-1)
  RANK 2: dp_rank=1, tp_rank=0, ep_rank=2  (DP group 1, experts N/2..3N/4-1)
  RANK 3: dp_rank=1, tp_rank=1, ep_rank=3  (DP group 1, experts 3N/4..N-1)

EP status: With --enable-expert-parallel, EP size = DP × TP = 4. All 4 GPUs
participate in expert parallelism — each holds a unique N/4 subset of experts.
Tokens are routed across all ranks via all-to-all communication.

Key rule: TP partners (same dp_rank) MUST process the SAME prompt subset.
vllm dp_rank = RANK // tensor_parallel_size  (i.e. RANK // 2)

torchrun does NOT set ZE_AFFINITY_MASK, so all processes share a unified
Level Zero device list. vLLM's XPU worker sets ZE_AFFINITY_MASK internally
to isolate each worker to its assigned device.

Token distribution example (8 prompts with varying lengths):
  dp_rank=0 (RANK 0,1): 30 tokens  (prompts 0,2,4,6)
  dp_rank=1 (RANK 2,3): 26 tokens  (prompts 1,3,5,7)
  With DP padding: both padded to max(30,26) = 30 tokens
"""

from vllm import LLM, SamplingParams


def main():
    # --- Prompts ---
    # All ranks see the same full prompt list; vLLM dispatches by dp_rank.
    # TP partners (same dp_rank) MUST receive the same prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "The best programming language is",
        "Artificial intelligence will",
        "The meaning of life is",
        "In the year 2050,",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # --- Engine ---
    llm = LLM(
        model="Qwen/Qwen3.5-32B-A4B",
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        seed=1,
        dtype="bfloat16",
        enforce_eager=True,
    )

    dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
    dp_size = llm.llm_engine.vllm_config.parallel_config.data_parallel_size

    # Split prompts across DP ranks (round-robin)
    my_prompts = [
        f"{idx}.{prompt}"
        for idx, prompt in enumerate(prompts)
        if idx % dp_size == dp_rank
    ]

    outputs = llm.generate(my_prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"DP Rank: {dp_rank} "
            f"Prompt: {prompt!r}\n"
            f"Generated text: {generated_text!r}\n"
        )


if __name__ == "__main__":
    main()
