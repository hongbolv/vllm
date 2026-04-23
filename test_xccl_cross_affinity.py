"""
test_xccl_cross_affinity.py — 验证 XCCL 跨 ZE_AFFINITY_MASK IPC 通信假设

用法：
  # 模式 A（基线）：不设置 ZE_AFFINITY_MASK，模拟 TP=4/DP=1（预期正常）
  python test_xccl_cross_affinity.py --mode baseline

  # 模式 B（跨 affinity）：设置 ZE_AFFINITY_MASK，模拟 DP>1（预期 hang 如果假设成立）
  python test_xccl_cross_affinity.py --mode cross_affinity

  # 模式 C（单组 affinity）：所有进程使用相同 ZE_AFFINITY_MASK（对照组）
  python test_xccl_cross_affinity.py --mode same_affinity

目的：
  如果模式 B hang 但模式 A 正常，确认 XCCL 跨 ZE_AFFINITY_MASK IPC 通信是根因。
  如果模式 B 也正常，需要继续调查其他原因。

前提：
  - 系统有 4 块 Intel XPU GPU
  - 已安装 PyTorch with XPU 支持和 oneCCL
"""

import argparse
import multiprocessing
import os
import sys
import time


def worker_fn(rank: int, world_size: int, master_port: int, mode: str):
    """每个 worker 进程的入口函数"""
    import torch
    import torch.distributed as dist

    # 设置 CCL debug 日志
    os.environ["CCL_LOG_LEVEL"] = "debug"
    os.environ["CCL_LOG_FLUSH"] = "1"

    # 根据模式设置 ZE_AFFINITY_MASK
    if mode == "baseline":
        # 不设置 ZE_AFFINITY_MASK — 所有进程看到全部 4 GPU
        if "ZE_AFFINITY_MASK" in os.environ:
            del os.environ["ZE_AFFINITY_MASK"]
        affinity_info = "None (all GPUs visible)"
    elif mode == "cross_affinity":
        # 模拟 vLLM DP>1：rank 0,1 用 GPU 0,1；rank 2,3 用 GPU 2,3
        if rank < world_size // 2:
            os.environ["ZE_AFFINITY_MASK"] = "0,1"
        else:
            os.environ["ZE_AFFINITY_MASK"] = "2,3"
        affinity_info = os.environ["ZE_AFFINITY_MASK"]
    elif mode == "same_affinity":
        # 对照组：所有进程使用相同的 ZE_AFFINITY_MASK
        os.environ["ZE_AFFINITY_MASK"] = "0,1,2,3"
        affinity_info = os.environ["ZE_AFFINITY_MASK"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[Rank {rank}][PID={os.getpid()}] === Worker START ===", flush=True)
    print(
        f"[Rank {rank}][PID={os.getpid()}] mode={mode}, "
        f"ZE_AFFINITY_MASK={affinity_info}",
        flush=True,
    )

    # 初始化 XPU
    device_count = torch.xpu.device_count()
    print(
        f"[Rank {rank}][PID={os.getpid()}] device_count={device_count}", flush=True
    )

    for i in range(device_count):
        props = torch.xpu.get_device_properties(i)
        print(
            f"[Rank {rank}][PID={os.getpid()}]   xpu:{i} = {props.name}, "
            f"total_memory={props.total_memory}",
            flush=True,
        )

    # 计算 local_rank（模拟 vLLM 的行为）
    if mode == "cross_affinity":
        # 每个 affinity 组内的 local_rank：0 或 1
        local_rank = rank % (world_size // 2)
    elif mode == "baseline":
        local_rank = rank
    else:
        local_rank = rank

    # 确保 local_rank 不超过 device_count
    if local_rank >= device_count:
        print(
            f"[Rank {rank}][PID={os.getpid()}] WARNING: local_rank={local_rank} "
            f">= device_count={device_count}, clamping to {device_count - 1}",
            flush=True,
        )
        local_rank = device_count - 1

    # 设备绑定
    torch.xpu.set_device(local_rank)
    current_device = torch.xpu.current_device()
    device_name = torch.xpu.get_device_properties(current_device).name
    print(
        f"[Rank {rank}][PID={os.getpid()}] device bound: local_rank={local_rank}, "
        f"current_device={current_device}, device_name={device_name}",
        flush=True,
    )

    # 初始化 process group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    print(
        f"[Rank {rank}][PID={os.getpid()}] calling init_process_group: "
        f"backend=xccl, world_size={world_size}, rank={rank}",
        flush=True,
    )

    dist.init_process_group(
        backend="xccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    print(
        f"[Rank {rank}][PID={os.getpid()}] init_process_group DONE", flush=True
    )

    # warmup all_reduce（模拟 vLLM 的 warmup 逻辑）
    tensor = torch.zeros(1, device=f"xpu:{local_rank}")
    print(
        f"[Rank {rank}][PID={os.getpid()}] calling all_reduce, "
        f"tensor.device={tensor.device}",
        flush=True,
    )

    dist.all_reduce(tensor)
    print(
        f"[Rank {rank}][PID={os.getpid()}] all_reduce submitted (async)",
        flush=True,
    )

    # 显式同步 — 如果跨 affinity mask 假设成立，这里会 hang
    print(
        f"[Rank {rank}][PID={os.getpid()}] calling torch.xpu.synchronize()...",
        flush=True,
    )
    torch.xpu.synchronize()
    print(
        f"[Rank {rank}][PID={os.getpid()}] synchronize DONE ✅", flush=True
    )

    # 清理
    dist.destroy_process_group()
    print(
        f"[Rank {rank}][PID={os.getpid()}] === Worker COMPLETE ✅ ===", flush=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="验证 XCCL 跨 ZE_AFFINITY_MASK IPC 通信假设"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "cross_affinity", "same_affinity"],
        default="baseline",
        help=(
            "baseline: 不设置 ZE_AFFINITY_MASK（模拟 TP=4/DP=1）; "
            "cross_affinity: 分组设置（模拟 DP>1）; "
            "same_affinity: 统一设置（对照组）"
        ),
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="总 rank 数（默认 4）",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="TCP rendezvous 端口（默认 29500）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="每个 worker 的超时秒数（默认 120）",
    )
    args = parser.parse_args()

    print(f"=" * 60, flush=True)
    print(f"test_xccl_cross_affinity.py", flush=True)
    print(f"  mode:       {args.mode}", flush=True)
    print(f"  world_size: {args.world_size}", flush=True)
    print(f"  port:       {args.master_port}", flush=True)
    print(f"  timeout:    {args.timeout}s", flush=True)
    print(f"=" * 60, flush=True)

    # 使用 spawn 启动所有 worker
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for rank in range(args.world_size):
        p = ctx.Process(
            target=worker_fn,
            args=(rank, args.world_size, args.master_port, args.mode),
            name=f"worker-{rank}",
        )
        p.start()
        processes.append(p)

    # 等待所有 worker 完成或超时
    start_time = time.time()
    all_done = False
    while time.time() - start_time < args.timeout:
        if all(not p.is_alive() for p in processes):
            all_done = True
            break
        time.sleep(1)

    if not all_done:
        print(f"\n{'=' * 60}", flush=True)
        print(
            f"⏰ TIMEOUT after {args.timeout}s — some workers still running!",
            flush=True,
        )
        for i, p in enumerate(processes):
            if p.is_alive():
                print(f"  🔴 Worker rank={i} (PID={p.pid}) still alive — HANG",
                      flush=True)
                p.terminate()
            else:
                print(f"  ✅ Worker rank={i} exited with code {p.exitcode}",
                      flush=True)
        print(f"{'=' * 60}", flush=True)
        sys.exit(1)
    else:
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}", flush=True)
        print(f"✅ All workers completed in {elapsed:.1f}s", flush=True)
        for i, p in enumerate(processes):
            print(f"  Worker rank={i}: exit code {p.exitcode}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # 检查是否有非零退出码
        if any(p.exitcode != 0 for p in processes):
            print("⚠️ Some workers exited with non-zero code!", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
