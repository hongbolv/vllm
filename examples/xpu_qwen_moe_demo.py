# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2024 Intel Corporation
# Demo: Qwen3-30B-A3B on 4x Intel Arc Pro B60 with TP=4

import os

# Triton is not available on XPU; disable torch.compile to avoid errors
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# XPU model loading can be slow; increase the engine startup timeout
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1800")

from vllm import LLM, SamplingParams


def main():
    model_path = "/home/media/Hongbo/models/Qwen3-30B-A3B"

    print("=" * 60)
    print("Qwen3-30B-A3B Demo with Tensor Parallelism = 4")
    print("Device: 4x Intel Arc Pro B60 (XPU)")
    print("=" * 60)

    # Initialize the model with TP=4
    # enforce_eager=True since we are not using Triton
    # Qwen3-30B-A3B default max_position_embeddings is 32768; use 4096 here
    # for a practical demo on 4x Intel Arc Pro B60.
    # num_gpu_blocks_override is set to cover 4096 tokens (block_size=16 ->
    # 256 blocks minimum).
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        num_gpu_blocks_override=256,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    prompts = [
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

    print("\nGenerating responses...\n")
    outputs = llm.generate(prompts, sampling_params)

    print("=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Response:  {generated_text!r}")
        print("-" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
