# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Heavy workload benchmark for vLLM inference.

Generates a large batch of diverse prompts with long output sequences
to stress-test the inference engine under sustained GPU load.
Measures throughput (tokens/sec) and latency statistics.

Reference: Qwen3-30B-A3B MoE demo for multi-GPU inference.

Usage examples:

    # Basic heavy workload with default settings (GPU, eager mode)
    python examples/offline_inference/heavy_workload_benchmark.py

    # Custom model with tensor parallelism
    python examples/offline_inference/heavy_workload_benchmark.py \
        --model Qwen/Qwen3-30B-A3B \
        --tensor-parallel-size 4 \
        --num-prompts 200 \
        --max-tokens 512

    # XPU with explicit dtype (auto-detects XPU and configures eagerly)
    python examples/offline_inference/heavy_workload_benchmark.py \
        --model Qwen/Qwen3-30B-A3B \
        --dtype float16
"""

from __future__ import annotations

import argparse
import os
import time

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Diverse heavy prompts — each triggers long-form generation
# ---------------------------------------------------------------------------

HEAVY_PROMPTS = [
    # Analytical / reasoning
    "Write a detailed technical comparison of TCP and UDP protocols, "
    "including use cases, performance characteristics, header formats, "
    "and real-world examples in modern distributed systems:",
    "Explain the theory of general relativity step by step, starting "
    "from the equivalence principle, deriving the Einstein field "
    "equations, and discussing experimental confirmations:",
    "Describe the complete lifecycle of a machine learning project "
    "from data collection through model deployment, including data "
    "preprocessing, feature engineering, model selection, training, "
    "evaluation, and monitoring in production:",
    "Provide an in-depth analysis of the global semiconductor supply "
    "chain, covering chip design, fabrication, packaging, and the "
    "geopolitical factors affecting the industry:",
    # Creative / narrative
    "Write a science fiction short story about a team of explorers "
    "who discover an ancient alien civilization on a distant planet. "
    "Include rich world-building, character development, and a "
    "twist ending:",
    "Compose a detailed historical narrative about the development "
    "of computing from Charles Babbage's Analytical Engine through "
    "modern quantum computers, highlighting key breakthroughs and "
    "the people behind them:",
    "Write a comprehensive travel guide for visiting Japan, covering "
    "Tokyo, Kyoto, Osaka, and Hokkaido. Include cultural tips, "
    "recommended restaurants, transportation options, and hidden "
    "gems for each region:",
    "Create a detailed business plan for a startup that uses AI to "
    "revolutionize healthcare diagnostics, including market analysis, "
    "competitive landscape, revenue model, and technical architecture:",
    # Technical / code-heavy
    "Write a complete Python implementation of a B-tree data structure "
    "with insert, delete, search, and range query operations. Include "
    "comprehensive docstrings, type hints, and unit tests:",
    "Explain the architecture of a modern LLM inference engine like "
    "vLLM, covering continuous batching, PagedAttention, KV cache "
    "management, tensor parallelism, and CUDA graph optimization:",
    "Design a distributed key-value store with strong consistency "
    "guarantees. Describe the Raft consensus protocol implementation, "
    "log replication, leader election, and snapshotting in detail:",
    "Write a comprehensive tutorial on GPU programming with CUDA, "
    "covering thread hierarchy, memory model, synchronization "
    "primitives, and optimization techniques for matrix multiplication:",
    # Math / science
    "Derive the Navier-Stokes equations from first principles, "
    "explain the physical meaning of each term, discuss boundary "
    "conditions, and describe numerical methods for solving them "
    "including finite element and spectral methods:",
    "Explain quantum computing from qubits to quantum error correction. "
    "Cover superposition, entanglement, quantum gates, the circuit "
    "model, Shor's algorithm, Grover's algorithm, and the current "
    "state of quantum hardware:",
    "Provide a detailed mathematical treatment of neural network "
    "backpropagation, including the chain rule derivation, gradient "
    "computation for various layer types, and numerical stability "
    "considerations:",
    "Describe the Standard Model of particle physics, including all "
    "fundamental particles, their interactions, the Higgs mechanism, "
    "and open problems like dark matter and the hierarchy problem:",
    # Long-form analysis
    "Write a detailed comparison of major programming languages "
    "(Python, Rust, Go, C++, Java, TypeScript) for building "
    "high-performance backend services, covering type systems, "
    "concurrency models, ecosystem maturity, and deployment options:",
    "Analyze the evolution of deep learning architectures from "
    "AlexNet through Vision Transformers and large language models, "
    "discussing the key innovations at each stage and their impact "
    "on the field:",
    "Explain modern cryptographic protocols used in internet security, "
    "including TLS 1.3, certificate authorities, key exchange "
    "mechanisms, symmetric and asymmetric encryption, and post-quantum "
    "cryptography:",
    "Write a comprehensive guide to database internals, covering "
    "B-tree and LSM-tree storage engines, query optimization, "
    "transaction isolation levels, MVCC, and distributed database "
    "architecture:",
]


def expand_prompts(base_prompts: list[str], target_count: int) -> list[str]:
    """Replicate and cycle prompts to reach the target count."""
    prompts = []
    idx = 0
    while len(prompts) < target_count:
        prompts.append(base_prompts[idx % len(base_prompts)])
        idx += 1
    return prompts


def run_benchmark(
    model: str,
    prompts: list[str],
    sampling_params: SamplingParams,
    *,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    max_model_len: int = 4096,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.90,
    num_gpu_blocks_override: int | None = None,
    trust_remote_code: bool = True,
) -> tuple[dict, list]:
    """Run inference and return timing / throughput statistics."""

    llm_kwargs: dict = dict(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = num_gpu_blocks_override

    print(f"  Initializing LLM (enforce_eager={enforce_eager}) ...")
    init_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    init_elapsed = time.perf_counter() - init_start
    print(f"  LLM initialized in {init_elapsed:.2f}s")

    # Warmup with a small subset
    warmup_prompts = prompts[:2]
    warmup_params = SamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=16,
    )
    print("  Warming up ...")
    llm.generate(warmup_prompts, warmup_params)

    # Main benchmark run
    print(
        f"  Generating {len(prompts)} responses "
        f"(max_tokens={sampling_params.max_tokens}) ..."
    )
    gen_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    gen_elapsed = time.perf_counter() - gen_start

    # Collect statistics
    total_prompt_tokens = 0
    total_output_tokens = 0

    for output in outputs:
        total_prompt_tokens += len(output.prompt_token_ids)
        total_output_tokens += len(output.outputs[0].token_ids)

    total_tokens = total_prompt_tokens + total_output_tokens
    throughput = total_output_tokens / gen_elapsed if gen_elapsed > 0 else 0

    stats = {
        "num_requests": len(prompts),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "generation_time_s": gen_elapsed,
        "init_time_s": init_elapsed,
        "output_throughput_tok_s": throughput,
        "total_throughput_tok_s": total_tokens / gen_elapsed if gen_elapsed > 0 else 0,
        "enforce_eager": enforce_eager,
    }

    # Clean up to free GPU memory before a potential second run
    del llm

    return stats, outputs


def print_stats(stats: dict, label: str = "") -> None:
    """Pretty-print benchmark statistics."""
    header = f"Results{f' ({label})' if label else ''}"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")
    print(f"  Requests            : {stats['num_requests']}")
    print(f"  Total prompt tokens : {stats['total_prompt_tokens']}")
    print(f"  Total output tokens : {stats['total_output_tokens']}")
    print(f"  Init time           : {stats['init_time_s']:.2f}s")
    print(f"  Generation time     : {stats['generation_time_s']:.2f}s")
    print(f"  Output throughput   : {stats['output_throughput_tok_s']:.2f} tok/s")
    print(f"  Total throughput    : {stats['total_throughput_tok_s']:.2f} tok/s")
    print(f"{'=' * 60}")


def print_sample_outputs(outputs, num_samples: int = 3) -> None:
    """Print a few sample outputs."""
    print(f"\n--- Sample outputs (first {num_samples}) ---")
    for i, output in enumerate(outputs[:num_samples]):
        prompt_preview = output.prompt[:80] + ("..." if len(output.prompt) > 80 else "")
        generated = output.outputs[0].text[:200] + (
            "..." if len(output.outputs[0].text) > 200 else ""
        )
        print(f"  [{i}] Prompt:   {prompt_preview!r}")
        print(f"      Response: {generated!r}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Heavy workload benchmark for vLLM inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-1.3b",
        help="Model name or path (default: facebook/opt-1.3b)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to generate (default: 100)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max output tokens per request (default: 256)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type (default: auto)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution (no torch.compile). Automatically enabled on XPU.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (default: 0.90)",
    )
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help="Override KV cache block count (for memory-limited GPUs)",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Number of sample outputs to print (default: 3; 0 to disable)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trusting remote code in model (default: trust enabled)",
    )
    args = parser.parse_args()

    # Detect XPU and auto-configure — XPU only supports eager mode
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
            os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1800")
            args.enforce_eager = True
            print(
                "[INFO] XPU detected — forcing eager mode, "
                "torch.compile disabled, timeout extended to 1800s"
            )
    except ImportError:
        pass

    prompts = expand_prompts(HEAVY_PROMPTS, args.num_prompts)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    common_kwargs = dict(
        model=args.model,
        prompts=prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        trust_remote_code=not args.no_trust_remote_code,
    )

    print("=" * 60)
    print("  Heavy Workload Benchmark")
    print(f"  Model           : {args.model}")
    print(f"  Tensor Parallel : {args.tensor_parallel_size}")
    print(f"  Num Prompts     : {args.num_prompts}")
    print(f"  Max Tokens      : {args.max_tokens}")
    print(f"  Max Model Len   : {args.max_model_len}")
    print(f"  dtype           : {args.dtype}")
    print("=" * 60)

    # Single mode run
    stats, outputs = run_benchmark(
        **common_kwargs,
        enforce_eager=args.enforce_eager,
    )
    mode_label = "Eager" if args.enforce_eager else "Compiled"
    print_stats(stats, label=mode_label)
    if args.show_samples > 0:
        print_sample_outputs(outputs, args.show_samples)

    print("\nDone!")


if __name__ == "__main__":
    main()
