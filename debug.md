# vLLM XPU DP>1 Hang 问题调试总结

## 一、问题描述

在 vLLM 的 **XPU 平台**上，当 **DP（数据并行度）> 1** 时，系统在 worker 初始化阶段 **hang 住**，无法继续执行。

- **正常场景：** TP=4 / DP=1 可以正常工作
- **正常场景：** 独立 `torchrun --nproc_per_node=4` 测试可以正常工作
- **异常场景：** vLLM DP > 1 时 hang

---

## 二、代码执行流（已通过源码及实测确认）

关键文件：`vllm/v1/worker/xpu_worker.py` → `XPUWorker.init_device()` 方法

| 行号 | 操作 | 执行状态 |
|------|------|----------|
| 62-68 | 设备初始化（`set_device_index`, `empty_cache`, 获取设备属性） | ✅ 已完成 |
| 72-78 | 设置 CCL 环境变量（`CCL_ATL_TRANSPORT`, `LOCAL_WORLD_SIZE`, `LOCAL_RANK`） | ✅ 已完成 |
| **80-86** | **`init_worker_distributed_environment(...)`** — 内含 `init_process_group` | ✅ **已成功** |
| **89-90** | **warmup all_reduce**: `torch.distributed.all_reduce(torch.zeros(1).xpu())` | ✅ **已提交成功（异步）** |
| — | 🔴 **显式插入 `torch.xpu.synchronize()` → hang 在此处** | ❌ **HANG** |
| 92-93 | `set_random_seed(...)` | ❌ **未到达** |
| 96 | `gc.collect()` | ❌ **未到达** |
| 97 | `torch.accelerator.empty_cache()` | ❌ **未到达** |
| 100 | `MemorySnapshot(device=self.device)` | ❌ **未到达** |

```python
# 第 89-90 行（原代码）
if torch.distributed.is_xccl_available():
    torch.distributed.all_reduce(torch.zeros(1).xpu())

# ↓ 插入显式同步后 hang 在这里 ↓
torch.xpu.synchronize()  # 🔴 HANG HERE — 所有 rank 均 hang

# 以下代码均不会执行
set_random_seed(self.model_config.seed)  # 第 93 行
gc.collect()                              # 第 96 行
torch.accelerator.empty_cache()           # 第 97 行
```

ApiServer 在 600 秒后超时（`VLLM_ENGINE_READY_TIMEOUT_S`）。

---

## 三、已确认的事实

1. **`init_process_group` 成功** — TCP rendezvous 完成，所有 rank 加入 WORLD group
2. **`all_reduce` 异步提交成功** — 操作入队到 XPU command queue 后立即返回
3. **CCL 报告 allreduce LL256 kernel 入队 "done"** — 所有 4 个 rank 均报告完成
4. **`torch.xpu.synchronize()` 永久 hang** — 设备端集合通信无法完成
5. **`gc.collect()` 及后续代码均不可达**
6. **TP=4/DP=1 在同一硬件上正常工作** — 相同的 `has_all_vertices_connected: 0` 拓扑和 LL256 ring 算法
7. **独立 `test_xccl.py` 使用 `torchrun --nproc_per_node=4` 正常工作** — 4 个 rank 全部完成 `init_process_group` + `all_reduce` + `synchronize()`，确认 PyTorch/XCCL 本身没有问题
8. **诊断日志确认没有两阶段初始化** — 所有 worker 均显示 `is_initialized_before=False`，不存在 `destroy_process_group`
9. ~~诊断日志确认 LOCAL_RANK 冲突~~ — **已被 Trace 数据证伪**：Trace 显示 `ZE_AFFINITY_MASK` 已正确设置（`0,1` 和 `2,3`），每个 rank 映射到唯一物理 GPU
10. **Trace 确认 `ZE_AFFINITY_MASK` 已设置** — EngineCore 0 的 worker 使用 `ZE_AFFINITY_MASK=0,1`（看到 2 块 GPU），EngineCore 1 使用 `ZE_AFFINITY_MASK=2,3`（看到另外 2 块 GPU）
11. **Trace 确认设备绑定正确** — `device_count=2`，`current_device=1` 在不同 affinity 组中映射到不同物理 GPU
12. **✅ `test_xccl_cross_affinity.py` 确认根因** — 模式 A（无 affinity mask）✅ 通过，模式 B（跨 affinity mask）🔴 HANG，模式 C（同组 affinity mask）✅ 通过 — **直接证明 XCCL 跨 `ZE_AFFINITY_MASK` IPC 通信是根因**

---

## 四、根因分析过程

### 4.1 排除的假设

以下假设均已被实验和日志数据证伪：

| # | 假设 | 排除依据 |
|---|------|----------|
| 1 | `has_all_vertices_connected: 0` 硬件拓扑问题 | TP=4/DP=1 使用完全相同的拓扑和 LL256 ring 算法，正常工作 |
| 2 | SYCL kernel 路径 vs host-side 路径问题 | `CCL_SYCL_KERNELS=0` 仍然 hang |
| 3 | PyTorch/XCCL 跨进程通信问题 | 独立 `torchrun --nproc_per_node=4` 测试完全正常 |
| 4 | 两阶段进程组初始化（destroy+rebuild） | 诊断日志显示 `is_initialized_before=False`，不存在 destroy/rebuild |
| 5 | `init_process_group` 失败 | CCL 日志证实 `comm { size: 4, id: 1 }` 正确创建 |
| 6 | `all_reduce` 未提交 | 所有 4 个 rank 均报告 "done"（内核已入队） |
| 7 | 个别 rank 未参与 | 4 个 rank 全部进入 allreduce |
| 8 | **LOCAL_RANK 冲突导致多 rank 映射同一 GPU** | **Trace 数据显示 `ZE_AFFINITY_MASK` 已正确设置，每个 rank 映射到唯一物理 GPU** |

### 4.2 ✅ 已确认的根因：XCCL 跨 `ZE_AFFINITY_MASK` IPC 通信失败

**通过 `test_xccl_cross_affinity.py` 三种模式的实验直接确认。**

```
                           EngineCore 0                      EngineCore 1
                           ZE_AFFINITY_MASK=0,1              ZE_AFFINITY_MASK=2,3
                           ──────────────────                ──────────────────
                           Worker 0: xpu:0 → 物理 GPU 0     Worker 2: xpu:0 → 物理 GPU 2
                           Worker 1: xpu:1 → 物理 GPU 1     Worker 3: xpu:1 → 物理 GPU 3

                           ↓                                 ↓
                           XCCL init_process_group(world_size=4) 创建 4-rank 通信组
                           4 个 rank 在 4 块不同物理 GPU 上（✅ 设备映射正确）
                           但 rank 0,1 和 rank 2,3 使用不同的 ZE_AFFINITY_MASK
                           ↓
                           XCCL IPC 需要跨 affinity mask 边界传输数据
                           Level Zero IPC 句柄无法在不同 affinity 组的进程间正确工作
                           ↓
                           allreduce LL256 ring kernel 在设备端死锁
                           → torch.xpu.synchronize() 永久 hang
```

**对比 TP=4/DP=1（正常）：**
所有 4 个 worker 由同一个 EngineCore 生成，**不设置 `ZE_AFFINITY_MASK`**，所有进程看到全部 4 块 GPU，
`local_rank` 分别为 0, 1, 2, 3，XCCL IPC 在同一设备命名空间内工作，不存在跨 affinity 边界通信。

**底层原因：** Level Zero 的 `zeMemOpenIpcHandle()` 在跨 `ZE_AFFINITY_MASK` 边界时无法正确解析 IPC 句柄，导致 allreduce kernel 在 GPU 上访问无效远端内存映射而死锁。详见第九节 9.2。

---

## 五、独立 XCCL 测试结果

### 5.1 测试脚本 `test_xccl.py`

```python
import os, torch, torch.distributed as dist

os.environ["CCL_LOG_LEVEL"] = "debug"
os.environ["CCL_LOG_FLUSH"] = "1"

dist.init_process_group(backend="xccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

print(f"Rank {rank}/{world_size}: init_process_group done", flush=True)

tensor = torch.zeros(1).xpu(rank % torch.xpu.device_count())
dist.all_reduce(tensor)
print(f"Rank {rank}: all_reduce submitted", flush=True)

torch.xpu.synchronize()
print(f"Rank {rank}: synchronize done!", flush=True)

dist.destroy_process_group()
```

### 5.2 测试结果

使用 `torchrun --nproc_per_node=4` 运行，**全部 4 个 rank 成功完成**：
- `init_process_group` ✅
- `all_reduce` 提交 ✅
- `torch.xpu.synchronize()` ✅ — **不 hang**
- `destroy_process_group` ✅

**结论：PyTorch/XCCL 本身在跨进程场景下工作正常，问题是 vLLM 的 DP>1 worker 初始化逻辑导致了 LOCAL_RANK 冲突。**

### 5.3 为什么 `test_xccl.py` 没有出现 LOCAL_RANK 冲突？

`test_xccl.py` 使用 `torchrun --nproc_per_node=4` 启动，这与 vLLM DP>1 场景有本质区别：

| | `torchrun` (test_xccl.py) | vLLM DP>1 |
|--|---------------------------|-----------|
| **进程管理** | `torchrun` 统一管理 4 个进程 | 多个 EngineCore 各自独立 spawn worker |
| **LOCAL_RANK 分配** | `torchrun` 自动分配 0,1,2,3（全局唯一） | 每个 EngineCore 内部从 0 开始（会重复） |
| **设备绑定** | `torchrun` 设置 `LOCAL_RANK` 环境变量，`xpu(LOCAL_RANK)` 各不相同 | `local_rank` 冲突导致 `xpu(0)` 和 `xpu(1)` 被多个 rank 共享 |
| **环境变量** | `torchrun` 设置 `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | vLLM 不依赖这些环境变量，参数通过函数传递，`LOCAL_RANK` 环境变量为 None |
| **设备可见性** | 所有进程看到全部 4 块 GPU，但 `LOCAL_RANK` 唯一 → 各用各的 GPU | 所有进程看到全部 4 块 GPU，`local_rank` 重复 → 多个 rank 用同一块 GPU |

关键差异：`torchrun` 是**单一启动器管理所有进程**，能确保 `LOCAL_RANK` 全局唯一。vLLM DP>1 是**多个 EngineCore 各自独立 spawn worker**，每个 EngineCore 内部 `local_rank` 从 0 开始，没有跨 EngineCore 的 `local_rank` 协调机制。

### 5.4 ~~LOCAL_RANK 冲突确认~~ → 已被 Trace 数据证伪

> **注意**：本节的分析基于第六节的早期诊断日志（PID 4185-4190），当时未观测到 `ZE_AFFINITY_MASK`。
> 第十节的 Trace 验证（PID 6834-6835）显示 `ZE_AFFINITY_MASK` 已正确设置，
> LOCAL_RANK 冲突假设不成立。详见第十节分析。

~~之前基于三重证据链认为 LOCAL_RANK 冲突是根因，但 Trace 数据显示：~~
~~1. `ZE_AFFINITY_MASK` 已设置（`0,1` 和 `2,3`）~~
~~2. `device_count=2`（每个 EngineCore 只看到 2 块 GPU）~~
~~3. 每个 rank 映射到唯一物理 GPU~~

当前需要重新分析根因，最可能是 **XCCL 跨 `ZE_AFFINITY_MASK` IPC 通信问题**。

---

## 六、诊断日志详细分析

### 6.1 诊断打印的实现

在 `vllm/distributed/parallel_state.py` 的 `init_distributed_environment()` 中添加了诊断打印，覆盖以下关键节点：
- **ENTRY**：函数入口参数（`world_size`, `rank`, `local_rank`, `distributed_init_method`, `backend`）
- **环境变量**：`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `ZE_AFFINITY_MASK`, `ONEAPI_DEVICE_SELECTOR`
- **DP 调整前**：`data_parallel_size`, `data_parallel_rank`, `world_size_across_dp`, `tensor_parallel_size`, 原始 `rank` 和 `world_size`
- **DP 调整后**：调整后的 `rank`, `world_size`, `ip`, `port`, `distributed_init_method`
- **init_process_group 调用前**：`backend`, `init_method`, `world_size`, `rank`, `is_initialized_before`
- **init_process_group 调用后**：`is_initialized`, `world_size`, `rank`, `backend`
- **最终 local_rank**：`local_rank`, `envs.LOCAL_RANK`, `ZE_AFFINITY_MASK`, `xpu_device_count`

### 6.2 实测日志数据（TP=2, DP=2, 4 GPU）

```
# === EngineCore 0 的 worker ===
(Worker pid=4185) [DEBUG-DP] init_distributed_environment ENTRY:
  world_size=2, rank=0, local_rank=0,
  distributed_init_method=tcp://127.0.0.1:41965, backend=xccl
(Worker pid=4185) env: RANK=None, LOCAL_RANK=0, WORLD_SIZE=None,
  MASTER_ADDR=None, MASTER_PORT=None
(Worker pid=4185) DP adjustment BEFORE:
  data_parallel_size=2, data_parallel_rank=0,
  world_size_across_dp=4, tensor_parallel_size=2,
  original_rank=0, original_world_size=2
(Worker pid=4185) DP adjustment AFTER:
  adjusted_rank=0, adjusted_world_size=4,
  ip=127.0.0.1, port=46163, distributed_init_method=tcp://127.0.0.1:46163
(Worker pid=4185) calling init_process_group:
  backend=xccl, init_method=tcp://127.0.0.1:46163,
  world_size=4, rank=0, is_initialized_before=False

(Worker pid=4186) [DEBUG-DP] init_distributed_environment ENTRY:
  world_size=2, rank=1, local_rank=1,
  distributed_init_method=tcp://127.0.0.1:41965, backend=xccl
(Worker pid=4186) DP adjustment AFTER:
  adjusted_rank=1, adjusted_world_size=4,
  ip=127.0.0.1, port=46163, distributed_init_method=tcp://127.0.0.1:46163
(Worker pid=4186) calling init_process_group:
  backend=xccl, init_method=tcp://127.0.0.1:46163,
  world_size=4, rank=1, is_initialized_before=False

# === EngineCore 1 的 worker ===
(Worker pid=4189) [DEBUG-DP] init_distributed_environment ENTRY:
  world_size=2, rank=0, local_rank=0,           ← local_rank=0 与 PID 4185 冲突!
  distributed_init_method=tcp://127.0.0.1:51907, backend=xccl
(Worker pid=4189) DP adjustment AFTER:
  adjusted_rank=2, adjusted_world_size=4,        ← 全局 rank 正确调整为 2
  ip=127.0.0.1, port=46163, ...
(Worker pid=4189) calling init_process_group:
  backend=xccl, world_size=4, rank=2, is_initialized_before=False

(Worker pid=4190) [DEBUG-DP] init_distributed_environment ENTRY:
  world_size=2, rank=1, local_rank=1,           ← local_rank=1 与 PID 4186 冲突!
  ...
(Worker pid=4190) DP adjustment AFTER:
  adjusted_rank=3, ...                           ← 全局 rank 正确调整为 3
```

### 6.3 关键发现汇总

#### ✅ 发现1：不存在两阶段初始化

所有 worker 均显示 `is_initialized_before=False`。**此前的"两阶段初始化"假设被证伪。**
CCL 日志中 PID 2441 的 finalize 可能是其他模块的短暂初始化（如 torch.xpu 初始化时的内部 communicator），与 vLLM 的 `init_process_group` 无关。

#### ~~🔴 发现2：LOCAL_RANK 冲突确认~~ → 已被 Trace 证伪

> **注意**：以下数据来自早期运行（PID 4185-4190），Trace 验证（第十节）显示最新运行中
> `ZE_AFFINITY_MASK` 已正确设置。两次运行可能使用了不同的启动路径或配置。

| PID | EngineCore | dp_rank | 原始 rank | **调整后 rank** | **local_rank** | **映射 GPU** |
|-----|------------|---------|-----------|----------------|----------------|-------------|
| 4185 | 0 | 0 | 0 | 0 | **0** | xpu:**0** |
| 4186 | 0 | 0 | 1 | 1 | **1** | xpu:**1** |
| 4189 | 1 | 1 | 0 | 2 | **0** | xpu:**0** ← ~~冲突~~ (此次运行无 ZE_AFFINITY_MASK) |
| 4190 | 1 | 1 | 1 | 3 | **1** | xpu:**1** ← ~~冲突~~ (此次运行无 ZE_AFFINITY_MASK) |

DP 调整只修改了全局 `rank`（偏移为 `dp_rank * world_size + rank`）和 `world_size`，
**但 `local_rank` 没有任何调整**，仍然是每个 EngineCore 内部的 0-indexed 值。

#### ~~🔴 发现3：XPU 平台缺少 `ZE_AFFINITY_MASK` 设置~~ → 在最新 Trace 运行中已正确设置

> **注意**：早期运行（PID 4185-4190）中可能未设置 `ZE_AFFINITY_MASK`，但 Trace 运行（PID 6834-6835）
> 显示 `ZE_AFFINITY_MASK=0,1` 和 `ZE_AFFINITY_MASK=2,3` 已正确设置。
> 这表明 `ZE_AFFINITY_MASK` 可能通过 MP（多进程）路径的 `set_device_control_env_var()` 设置，
> 而非 `DPEngineCoreActor._set_visible_devices()`。

在 `vllm/v1/engine/core.py` 的 `DPEngineCoreActor._set_visible_devices()` 中：

```python
def _set_visible_devices(self, vllm_config, local_dp_rank):
    from vllm.platforms import current_platform
    if current_platform.is_xpu():
        pass  # ← XPU 什么都不做！不设置 ZE_AFFINITY_MASK
    else:
        # CUDA 平台会设置 CUDA_VISIBLE_DEVICES
        self._set_cuda_visible_devices(...)
```

**CUDA 平台的处理（正确）：**
- EngineCore 0: `CUDA_VISIBLE_DEVICES=0,1` → `local_rank=0` 映射物理 GPU 0，`local_rank=1` 映射物理 GPU 1
- EngineCore 1: `CUDA_VISIBLE_DEVICES=2,3` → `local_rank=0` 映射物理 GPU 2，`local_rank=1` 映射物理 GPU 3

**XPU 平台的处理（有缺陷）：**
- 所有 EngineCore 看到**全部 4 块 GPU**
- `local_rank=0` 在所有 EngineCore 中都映射到**物理 GPU 0**

> **注意**：在 MP（多进程）路径中（`vllm/v1/engine/utils.py`），`set_device_control_env_var()`
> 会为非 CUDA 平台（包括 XPU）设置 `ZE_AFFINITY_MASK`。但 Ray 路径中
> `DPEngineCoreActor` 的 XPU 分支是 `pass`，不会设置设备亲和性。

---

## 七、TP=4/DP=1 vs DP>1 的代码路径对比

### 7.1 `init_distributed_environment()` 中的差异

**TP=4/DP=1（正常）：**
```python
# data_parallel_size == 1，不进入 DP 调整分支
# rank 保持原值（0, 1, 2, 3）
# world_size 保持原值（4）
# local_rank 分别为 0, 1, 2, 3 → 各自对应不同 GPU
torch.distributed.init_process_group(
    backend="xccl",
    world_size=4,
    rank=rank,        # 0, 1, 2, 3
    init_method=原始方法,
)
```

**DP>1（hang）：**
```python
# data_parallel_size > 1，进入 DP 调整分支
rank = data_parallel_rank * world_size + rank   # 全局 rank 偏移
world_size = world_size_across_dp               # world_size 扩大
ip = data_parallel_master_ip
port = get_next_dp_init_port()                  # 新端口
distributed_init_method = get_distributed_init_method(ip, port)
# ⚠️ local_rank 未调整！仍然是 EngineCore 内的 0-indexed 值

torch.distributed.init_process_group(
    backend="xccl",
    world_size=world_size_across_dp,
    rank=调整后的rank,
    init_method=新的TCP端口,
)
```

### 7.2 关键差异总结

| | TP=4/DP=1 | DP>1 |
|--|-----------|------|
| **worker 来源** | 同一个 EngineCore | 不同 EngineCore |
| **local_rank** | 0, 1, 2, 3（唯一） | 0, 1, 0, 1（数值重复但 ZE_AFFINITY_MASK 隔离） |
| **GPU 映射** | 4 rank → 4 GPU（直接映射） | 4 rank → 4 GPU（通过 ZE_AFFINITY_MASK 虚拟化） |
| **rank 值** | 原始 0-3 | 偏移调整后 0-3 |
| **world_size** | 原始 4 | `TP × DP` |
| **TCP 端口** | 原始端口 | 新端口 |
| **ZE_AFFINITY_MASK** | 不设置（所有 GPU 可见） | ⚠️ 设置 `0,1` 和 `2,3`（分组隔离） |
| **XCCL IPC** | 同一设备命名空间 | ⚠️ 跨 affinity mask 边界 |

---

## 八、CCL Debug 日志分析

通过 `CCL_LOG_LEVEL=debug` 和 `CCL_LOG_FLUSH=1` 收集到的日志信息：

### 8.1 环境与拓扑

| 项目 | 值 |
|------|------|
| **GPU** | Intel(R) Arc(TM) Pro B60 Graphics × 4 |
| **设备族** | family6 |
| **is_single_tile** | 1 |
| **has_all_vertices_connected** | 0（TP=4/DP=1 同样如此且正常） |
| **stream** | gpu, in_order: 1 |
| **WORLD group** | `comm { rank: X, size: 4, id: 1 }` |

### 8.2 allreduce 内核入队（所有 rank 均成功）

```
[每个 rank 的 CCL 执行路径]
→ zeDeviceGetProperties
→ stream: { type: gpu, in_order: 1, device: Arc Pro B60, device_family: family6 }
→ can_use_sycl_kernels: coll allreduce, local_proc_count 4, comm { rank: X, size: 4, id: 1 }
→ selected algo: coll allreduce, algo topo sycl
→ allreduce selects sycl-kernels count: 1, datatype: FLOAT32
→ allreduce_sycl: is_single_node
→ is_single_tile: 1, has_all_vertices_connected: 0
→ invoking allreduce LL256 kernel allreduce_ll_ring, count:1 datatype: FLOAT32
→ invoking allreduce LL256 kernel arc_allreduce, count:1 datatype: FLOAT32
→ invoking allreduce LL256 kernel, count:1 datatype: FLOAT32 done    ← ✅ 入队成功
```

### 8.3 PID → Rank 映射与 hang 确认

| PID | Rank | CCL 内核入队 | 打印 | synchronize() |
|-----|------|-------------|------|---------------|
| 2267 | [0] | ✅ "done" | ✅ "calling synchronize" | 🔴 HANG |
| 2268 | [1] | ✅ "done" | ✅ "calling synchronize" | 🔴 HANG |
| 2271 | [2] | ✅ "done" | ✅ "calling synchronize" | 🔴 HANG |
| 2272 | [3] | ✅ "done" | ✅ "calling synchronize" | 🔴 HANG |

600 秒后 ApiServer 超时退出。

---

## 九、根因结论（已更新）

### ~~之前的假设：LOCAL_RANK 冲突~~（已被 Trace 数据证伪）

之前基于第六节的诊断日志（PID 4185-4190 运行），推断 LOCAL_RANK 冲突是根因。
但第十节的 Trace 验证（PID 6834-6835 运行）显示 `ZE_AFFINITY_MASK` 已正确设置，
每个 rank 映射到唯一的物理 GPU，**不存在 LOCAL_RANK 冲突**。

### ✅ 根因已确认：XCCL 跨 `ZE_AFFINITY_MASK` IPC 通信失败

**`test_xccl_cross_affinity.py` 验证结果（已实测确认）：**

| 模式 | ZE_AFFINITY_MASK 设置 | 结果 |
|------|----------------------|------|
| 模式 A（baseline） | 不设置 — 所有进程看到全部 4 GPU | ✅ 通过 |
| 模式 B（cross_affinity） | Rank 0,1 → `0,1`；Rank 2,3 → `2,3` | 🔴 HANG |
| 模式 C（same_affinity） | 所有进程 → `0,1,2,3` | ✅ 通过 |

**结论：模式 A 和 C 通过，模式 B hang — 直接确认 XCCL/oneCCL 无法在跨 `ZE_AFFINITY_MASK` 边界的进程间正常完成集合通信。**

### 9.1 TP=4/DP=1 vs DP>1 的关键差异

| | TP=4/DP=1（正常） | DP>1（hang） |
|--|-------------------|-------------|
| **ZE_AFFINITY_MASK** | 未设置（所有进程看到全部 4 GPU） | 设置为 `0,1` 和 `2,3`（分组隔离） |
| **Level Zero 设备命名空间** | 统一命名空间，device 0-3 对应物理 GPU 0-3 | 虚拟化命名空间，每个组内 device 0,1 映射到不同物理 GPU |
| **XCCL IPC 句柄** | 所有 rank 在同一 Level Zero driver 实例内 | rank 跨不同 Level Zero driver 实例通信 |

### 9.2 底层原因分析

问题发生在 **Intel Level Zero Runtime 的 IPC（进程间通信）机制** 层面。具体原因链条：

1. **`ZE_AFFINITY_MASK` 改变了 Level Zero 的设备枚举**
   - 不设置 `ZE_AFFINITY_MASK` 时，所有进程看到 4 块 GPU，设备 ID 为 `0,1,2,3`，与物理设备一一对应
   - 设置 `ZE_AFFINITY_MASK=0,1` 时，该进程的 Level Zero runtime 只枚举 2 块 GPU，设备 ID 变为 `0,1`（虚拟 ID），物理设备为 GPU 0 和 GPU 1
   - 设置 `ZE_AFFINITY_MASK=2,3` 时，设备 ID 同样为 `0,1`（虚拟 ID），但物理设备为 GPU 2 和 GPU 3

2. **XCCL/oneCCL 依赖 Level Zero IPC 句柄实现跨进程 GPU 通信**
   - XCCL 的 allreduce ring 算法需要跨所有 4 个 rank 的 GPU 直接通信
   - 这要求进程间交换 Level Zero IPC memory handle（`ze_ipc_mem_handle_t`），使一个进程的 GPU 可以直接读写另一个进程的 GPU 内存

3. **跨 `ZE_AFFINITY_MASK` 边界时 IPC 句柄失效**
   - 当 Rank 0（`ZE_AFFINITY_MASK=0,1`）尝试打开 Rank 2（`ZE_AFFINITY_MASK=2,3`）导出的 IPC 句柄时，由于两个进程的 Level Zero driver 实例管理不同的物理设备子集，IPC 句柄在目标进程的 driver 上下文中无法被正确解析
   - Level Zero 的 `zeMemOpenIpcHandle()` 可能返回错误或返回无效指针，导致 allreduce kernel 在尝试访问远端 GPU 内存时死锁
   - CCL 日志中显示 kernel enqueue "done"（allreduce kernel 已提交到 GPU command queue），但 kernel 在设备端执行时 hang — 正是因为 kernel 尝试通过无效 IPC 映射读取远端数据，触发了设备级死锁

4. **为什么 `has_all_vertices_connected: 0` 在此场景下产生影响**
   - 当设备无全互联拓扑时，CCL 使用 ring 算法（LL256），每个 rank 必须与相邻 rank 的 GPU 直接通信
   - ring 中 rank 0 → rank 1 → rank 2 → rank 3 → rank 0，其中 rank 1 → rank 2 的通信跨越了 `ZE_AFFINITY_MASK` 边界
   - 在全互联拓扑下，CCL 可能使用不依赖 IPC 的算法，或者 IPC 通过不同路径工作，可能不会触发此问题

5. **为什么 `test_xccl.py`（torchrun 模式）不受影响**
   - `torchrun --nproc_per_node=4` 不设置 `ZE_AFFINITY_MASK`，所有进程共享同一个 Level Zero 设备命名空间
   - 每个进程通过全局唯一的 `LOCAL_RANK=0,1,2,3` 绑定设备，IPC 句柄在同一 driver 上下文内交换，正常工作

### 9.3 根因总结

```
ZE_AFFINITY_MASK 分组设置（0,1 vs 2,3）
    ↓
Level Zero Runtime 创建独立的设备枚举命名空间
    ↓
每个进程的 driver 实例只管理自己组内的物理 GPU
    ↓
XCCL/oneCCL 尝试在 4 个 rank 间建立 ring allreduce
    ↓
ring 中跨 affinity 边界的 rank 交换 IPC 句柄
    ↓
接收方的 Level Zero driver 无法解析来自不同 affinity 组的 IPC 句柄
    ↓
zeMemOpenIpcHandle() 失败或返回无效映射
    ↓
allreduce kernel 在 GPU 执行时尝试访问无效远端内存 → 设备级死锁
    ↓
torch.xpu.synchronize() 永久 hang
```

### 9.4 修复方向

| 方案 | 说明 | 可行性 |
|------|------|--------|
| **A. 避免跨 `ZE_AFFINITY_MASK` 通信** | vLLM 的 DP 架构中，TP 组内通信不需要跨 EngineCore，只有跨 TP 组的全局 allreduce（warmup）需要。可以为 warmup 临时移除 `ZE_AFFINITY_MASK` 隔离，或仅在 TP 组内做 warmup allreduce | ⭐⭐⭐ |
| **B. 统一 `ZE_AFFINITY_MASK`** | 所有进程设置 `ZE_AFFINITY_MASK=0,1,2,3`（等同于不设置），然后通过 `torch.xpu.set_device()` 绑定到正确的物理设备 | ⭐⭐⭐ |
| **C. 使用 socket-based IPC** | 设置 `CCL_ZE_IPC_EXCHANGE=sockets` 绕过 Level Zero IPC 句柄交换机制，改用 socket 传输 | ⭐⭐ |
| **D. 向 Intel 报告 Level Zero/oneCCL bug** | Level Zero IPC 应该能跨 `ZE_AFFINITY_MASK` 工作（CUDA 的 `CUDA_VISIBLE_DEVICES` 下 NCCL IPC 可以正常跨设备组通信），这可能是 Level Zero 或 oneCCL 的 bug | ⭐⭐ |

详见第十节的 trace 验证数据和第十一节的跨 affinity 测试详情。

---

## 十、Trace 验证结果 — LOCAL_RANK 冲突假设被证伪

### 10.1 实际 Trace 日志（TP=2, DP=2, 4 GPU）

使用设备级 trace patch 运行后，`grep -E "DEBUG-DP|TRACE" ccl_debug.log` 得到以下关键数据：

```
# === EngineCore 0 的 worker（rank=1）===
(Worker pid=6835) [TRACE][PID=6835] init_device START: rank=1, local_rank=1,
  device_count=2, ZE_AFFINITY_MASK=0,1, ONEAPI_DEVICE_SELECTOR=level_zero:gpu
(Worker pid=6835) [TRACE][PID=6835]   visible xpu:0 = Intel(R) Arc(TM) Pro B60 Graphics, total_memory=24385683456
(Worker pid=6835) [TRACE][PID=6835]   visible xpu:1 = Intel(R) Arc(TM) Pro B60 Graphics, total_memory=24385683456
(Worker pid=6835) [TRACE][PID=6835] device bound: rank=1, local_rank=1,
  self.device=xpu:1, current_device=1, device_name=Intel(R) Arc(TM) Pro B60 Graphics
(Worker pid=6835) [DEBUG-DP][PID=6835] init_distributed_environment ENTRY:
  world_size=2, rank=1, local_rank=1, distributed_init_method=tcp://127.0.0.1:46161, backend=xccl
(Worker pid=6835) [DEBUG-DP][PID=6835] env: RANK=None, LOCAL_RANK=1, WORLD_SIZE=None,
  MASTER_ADDR=None, MASTER_PORT=None
(Worker pid=6835) [DEBUG-DP][PID=6835] DP adjustment BEFORE:
  data_parallel_size=2, data_parallel_rank=0, world_size_across_dp=4,
  tensor_parallel_size=2, original_rank=1, original_world_size=2
(Worker pid=6835) [DEBUG-DP][PID=6835] DP adjustment AFTER:
  adjusted_rank=1, adjusted_world_size=4, ip=127.0.0.1, port=54231,
  distributed_init_method=tcp://127.0.0.1:54231
(Worker pid=6835) [DEBUG-DP][PID=6835] calling init_process_group:
  backend=xccl, init_method=tcp://127.0.0.1:54231, world_size=4, rank=1, is_initialized_before=False

# === EngineCore 1 的 worker（rank=1 → adjusted_rank=3）===
(Worker pid=6834) [TRACE][PID=6834] init_device START: rank=1, local_rank=1,
  device_count=2, ZE_AFFINITY_MASK=2,3, ONEAPI_DEVICE_SELECTOR=level_zero:gpu
(Worker pid=6834) [TRACE][PID=6834]   visible xpu:0 = Intel(R) Arc(TM) Pro B60 Graphics, total_memory=24385683456
(Worker pid=6834) [TRACE][PID=6834]   visible xpu:1 = Intel(R) Arc(TM) Pro B60 Graphics, total_memory=24385683456
(Worker pid=6834) [TRACE][PID=6834] device bound: rank=1, local_rank=1,
  self.device=xpu:1, current_device=1, device_name=...
# (日志在此截断)
```

### 10.2 关键发现：LOCAL_RANK 冲突假设被证伪 ❌

实际 trace 数据与预期的"LOCAL_RANK 冲突"模式**完全不符**：

| 观测项 | 预期（如果 LOCAL_RANK 冲突） | **实际观测** | 结论 |
|--------|--------------------------|-------------|------|
| `ZE_AFFINITY_MASK` | None（未设置） | **`0,1` 和 `2,3`**（已正确设置） | ✅ 设备隔离已生效 |
| `device_count` | 4（看到全部 GPU） | **2**（每个 EngineCore 只看到 2 块） | ✅ 设备可见性正确 |
| 物理 GPU 映射 | 多个 rank 映射到同一物理 GPU | **各 rank 映射到不同物理 GPU** | ✅ 无冲突 |

推导完整的 4-worker 物理 GPU 映射：

| Worker | PID | ZE_AFFINITY_MASK | local_rank | current_device | 物理 GPU |
|--------|-----|-----------------|------------|----------------|---------|
| EC0-W0 | (未显示) | 0,1 | 0 | 0 | **GPU 0** |
| EC0-W1 | 6835 | 0,1 | 1 | 1 | **GPU 1** |
| EC1-W0 | (未显示) | 2,3 | 0 | 0 | **GPU 2** |
| EC1-W1 | 6834 | 2,3 | 1 | 1 | **GPU 3** |

**每个 rank 映射到唯一的物理 GPU — 不存在 LOCAL_RANK 冲突！**

说明：`ZE_AFFINITY_MASK=0,1` 使进程只看到物理 GPU 0 和 1（作为虚拟 xpu:0 和 xpu:1）；
`ZE_AFFINITY_MASK=2,3` 使进程只看到物理 GPU 2 和 3（同样作为虚拟 xpu:0 和 xpu:1）。
因此 `current_device=1` 在不同 EngineCore 中映射到不同的物理 GPU。

### 10.3 与之前诊断日志的差异说明

之前的诊断日志（第六节，PID 4185-4190）中没有观测到 `ZE_AFFINITY_MASK`。
最新的 trace 日志（PID 6834-6835）显示 `ZE_AFFINITY_MASK` 已正确设置。
这可能是因为：
1. 两次运行使用了不同的启动路径（Ray vs MP）
2. 或者在两次运行之间有其他配置变化

无论原因如何，**最新的 trace 数据明确证明：在当前的运行配置下，LOCAL_RANK 冲突不存在，但 allreduce 仍然 hang。**

### 10.4 新的分析方向

既然 LOCAL_RANK 冲突已被排除，需要重新审视可能的根因：

| # | 新假设 | 分析方向 |
|---|--------|---------|
| 1 | **XCCL 跨 ZE_AFFINITY_MASK IPC 通信问题** | 当 4 个进程分属不同 `ZE_AFFINITY_MASK` 组（0,1 vs 2,3）时，XCCL 的 Level Zero IPC 句柄可能无法跨 affinity mask 边界正确工作。TP=4/DP=1 时所有进程看到全部 4 块 GPU，没有 affinity mask 隔离，因此 IPC 正常 |
| 2 | **`has_all_vertices_connected: 0` 在跨 affinity 场景下的影响** | 之前排除此假设时基于"TP=4/DP=1 也是 0 但正常"。但 TP=4/DP=1 时所有 rank 在同一 affinity 组，DP>1 时 rank 跨不同 affinity 组。拓扑不全互联在跨 affinity 场景下可能产生不同影响 |
| 3 | **XCCL communicator 创建时的设备 ID 映射** | 每个进程内 XCCL 使用虚拟设备 ID（0 或 1），但跨进程 IPC 需要用物理设备 ID。XCCL/oneCCL 是否正确处理了 `ZE_AFFINITY_MASK` 虚拟化？ |
| 4 | **CCL 的 `local_proc_count` 检测问题** | CCL 日志显示 `local_proc_count 4`，但实际上每个 affinity 组只有 2 个进程。如果 CCL 错误地认为 4 个进程都在同一组设备上，可能导致通信拓扑构建错误 |

### 10.5 跨 Affinity Mask 验证 — 已确认

使用 `test_xccl_cross_affinity.py` 进行了三组实验，**直接确认 XCCL 跨 `ZE_AFFINITY_MASK` IPC 通信失败是根因**。

#### 实验结果

| 模式 | ZE_AFFINITY_MASK 设置 | 结果 | 分析 |
|------|----------------------|------|------|
| 模式 A（baseline） | 不设置（全部 4 GPU 可见） | ✅ **PASSED** | XCCL 在统一设备命名空间内正常工作 |
| 模式 B（cross_affinity） | Rank 0,1 → `0,1`；Rank 2,3 → `2,3` | 🔴 **HANG** | **直接复现 vLLM DP>1 hang** |
| 模式 C（same_affinity） | 所有进程 → `0,1,2,3` | ✅ **PASSED** | 设置了 `ZE_AFFINITY_MASK` 但组内一致 → 正常 |

#### 结论

- **模式 A 和 C 通过**：XCCL 在 Level Zero IPC 句柄可以正确交换的场景下工作正常
- **模式 B hang**：当进程分属不同的 `ZE_AFFINITY_MASK` 组时，XCCL ring allreduce 的跨组 IPC 通信失败
- **这与 vLLM DP>1 的行为完全一致**：vLLM 为每个 EngineCore 设置不同的 `ZE_AFFINITY_MASK`（`0,1` vs `2,3`），导致跨 EngineCore 的 XCCL allreduce hang

#### 建议的后续验证

1. **尝试 `CCL_ZE_IPC_EXCHANGE=sockets`**：
   ```bash
   CCL_ZE_IPC_EXCHANGE=sockets python test_xccl_cross_affinity.py --mode cross_affinity
   ```
   如果 socket-based IPC 绕过了 Level Zero IPC 句柄的限制，可以作为临时 workaround

2. **向 Intel 报告 bug**：Level Zero IPC 应能跨 `ZE_AFFINITY_MASK` 工作（类比 CUDA 的 `CUDA_VISIBLE_DEVICES` 下 NCCL 可正常跨设备组通信）

---

## 十一、为什么 torchrun 路径（Option B）不触发跨 affinity 通信失败？

### 11.1 问题背景

使用 `torchrun --nproc-per-node=4` 启动 vLLM（Option B），DP=2/TP=2 时可以正常工作。
为什么这种方式没有触发跨 `ZE_AFFINITY_MASK` IPC 通信失败？

### 11.2 根本原因：`torchrun` 不设置 `ZE_AFFINITY_MASK`

**`torchrun` 启动路径不经过 vLLM 的 `CoreEngineProcManager`**，因此不会调用
`set_device_control_env_var()`。所有 4 个进程共享同一个 Level Zero 设备命名空间，
每个进程都能看到全部 4 块 GPU。

| 启动方式 | `ZE_AFFINITY_MASK` | 设备命名空间 | IPC 状态 |
|----------|-------------------|-------------|---------|
| **Option A（multiprocessing）** | 进程 0,1 → `0,1`；进程 2,3 → `2,3` | 分裂为 2 个独立命名空间 | ❌ 跨命名空间 IPC 失败 |
| **Option B（torchrun）** | **未设置**（所有进程看到全部 4 GPU） | 统一命名空间 | ✅ 同一命名空间内 IPC 正常 |
| **TP=4/DP=1** | 未设置 | 统一命名空间 | ✅ 同一命名空间内 IPC 正常 |

### 11.3 代码路径对比

**Option A（multiprocessing）— 触发 hang 的路径：**

```
vllm.LLM() → AsyncLLM → CoreEngineProcManager.__init__()
    → is_dp=True, current_platform.is_xpu()=True（非 cuda_alike）
    → set_device_control_env_var(vllm_config, local_dp_rank)  ← 设置 ZE_AFFINITY_MASK
        → EngineCore 0: ZE_AFFINITY_MASK=0,1
        → EngineCore 1: ZE_AFFINITY_MASK=2,3
    → 每个 EngineCore spawn worker → init_process_group(world_size=4)
    → warmup all_reduce → 跨 affinity 边界 → HANG
```

**Option B（torchrun）— 不触发 hang 的路径：**

```
torchrun → 直接启动 4 个进程（RANK=0,1,2,3, LOCAL_RANK=0,1,2,3）
    → 每个进程执行用户脚本 → vllm.LLM(tensor_parallel_size=2)
    → 不经过 CoreEngineProcManager（torchrun 已管理进程）
    → ZE_AFFINITY_MASK 未设置（所有进程看到全部 4 GPU）
    → init_process_group(world_size=4) → warmup all_reduce
    → 所有 IPC 在同一设备命名空间内 → ✅ 正常
```

### 11.4 关键差异

1. **进程启动方式不同**：Option A 由 vLLM 的 `CoreEngineProcManager` 分组 spawn，每组设置独立的 `ZE_AFFINITY_MASK`；Option B 由 `torchrun` 统一启动，不设置 `ZE_AFFINITY_MASK`

2. **设备隔离机制不同**：Option A 通过 `ZE_AFFINITY_MASK` 进行设备隔离（创建独立的 Level Zero 命名空间）；Option B 不做设备隔离，每个进程通过 `torch.xpu.set_device(LOCAL_RANK)` 绑定到不同 GPU，但所有 GPU 在同一个命名空间内可见

3. **IPC 跨越边界**：Option A 的 allreduce 需要跨 `ZE_AFFINITY_MASK` 边界通信（失败）；Option B 的 allreduce 在同一设备命名空间内通信（正常）

### 11.5 结论

**这进一步确认了根因：问题不在于 DP>1 的逻辑本身，而在于 vLLM 的 multiprocessing 路径为不同 EngineCore 设置了不同的 `ZE_AFFINITY_MASK`，导致 XCCL 跨 affinity 边界 IPC 通信失败。** `torchrun` 路径因为不设置 `ZE_AFFINITY_MASK` 而完全绕过了这个问题。

这也进一步验证了修复方案 B（统一 `ZE_AFFINITY_MASK`）的可行性——Option B 的正常工作证明，在不分裂 Level Zero 设备命名空间的前提下，DP=2/TP=2 的 XCCL 通信可以正常完成。

---

## 十二、环境信息

| 项目 | 值 |
|------|------|
| GPU | Intel(R) Arc(TM) Pro B60 Graphics × 4 |
| 设备族 | family6 |
| 单 tile | 是 |
| 设备互联 | 无全互联（`has_all_vertices_connected: 0`） |

---

## 附录：CCL Debug 日志（关键摘录）

```
# Rank 0 (PID 2267) — 完整 CCL 流程
2267:[0] |CCL_INFO| stream: { type: gpu, in_order: 1, device: Arc Pro B60, device_family: family6 }
2267:[0] |CCL_DEBUG| can_use_sycl_kernels: coll allreduce, local_proc_count 4, comm { rank: 0, size: 4, id: 1 }
2267:[0] |CCL_DEBUG| selected algo: coll allreduce, algo topo sycl
2267:[0] |CCL_DEBUG| allreduce selects sycl-kernels count: 1, datatype: FLOAT32
2267:[0] |CCL_DEBUG| allreduce_sycl: is_single_node
2267:[0] |CCL_DEBUG| is_single_tile: 1, has_all_vertices_connected: 0
2267:[0] |CCL_DEBUG| invoking allreduce LL256 kernel allreduce_ll_ring, count:1 datatype: FLOAT32
2267:[0] |CCL_DEBUG| invoking allreduce LL256 kernel arc_allreduce, count:1 datatype: FLOAT32
2267:[0] |CCL_DEBUG| invoking allreduce LL256 kernel, count:1 datatype: FLOAT32 done
(Worker pid=2267) [======VLLM_DEBUG======] all_reduce done calling synchronize, pid=2267

# Rank 1-3 (PID 2268/2271/2272) — 同样流程，均报告 "done" 后无更多输出

# 600 秒后
(ApiServer_1 pid=1217) TimeoutError: Timed out waiting for engine core processes to start.
```

---

## 十二、修复实现（方案 B：统一设备命名空间）

### 12.1 修复思路

核心思想：**不为不同 DP group 设置不同的 `ZE_AFFINITY_MASK`**，让所有进程看到全部 GPU，通过 `torch.xpu.set_device(adjusted_local_rank)` 控制设备绑定。这与 `torchrun`（Option B）成功工作的方式一致。

### 12.2 代码修改

#### 修改 1：`vllm/v1/engine/utils.py` — 跳过 XPU 的 `ZE_AFFINITY_MASK` 设置

在 `CoreEngineProcManager.__init__` 中，multiprocessing 启动 EngineCore 子进程时，跳过为 XPU 平台设置 `ZE_AFFINITY_MASK`：

```python
# 修改前：
if is_dp and (
    not current_platform.is_cuda_alike()
    or vllm_config.parallel_config.use_ray
):
    device_control_context = set_device_control_env_var(...)

# 修改后：
if is_dp and (
    (not current_platform.is_cuda_alike() and not current_platform.is_xpu())
    or vllm_config.parallel_config.use_ray
):
    device_control_context = set_device_control_env_var(...)
```

**效果**：XPU multiprocessing 路径不再为不同 EngineCore 设置不同的 `ZE_AFFINITY_MASK`，所有进程共享同一 Level Zero 设备命名空间。

#### 修改 2：`vllm/v1/worker/xpu_worker.py` — 添加 DP local_rank 偏移

在 `XPUWorker.init_device()` 中，设备绑定前添加 DP local_rank 调整逻辑（与 `gpu_worker.py` 中 CUDA 的调整逻辑对齐）：

```python
# DP local_rank 调整公式：
# actual_device_index = dp_local_rank * tp_pp_world_size + tp_local_rank
#
# 例如 DP=2/TP=2，4 块 GPU：
# EngineCore 0 (dp_local_rank=0): worker 0 → 0*2+0=GPU0, worker 1 → 0*2+1=GPU1
# EngineCore 1 (dp_local_rank=1): worker 0 → 1*2+0=GPU2, worker 1 → 1*2+1=GPU3
```

### 12.3 预期行为

修复后的进程拓扑（DP=2/TP=2，4 块 GPU）：

| PID | EngineCore | rank | local_rank(原) | dp_local_rank | local_rank(调整后) | ZE_AFFINITY_MASK | 物理 GPU |
|-----|-----------|------|---------------|--------------|-------------------|-----------------|---------|
| A | 0 | 0 | 0 | 0 | 0 | 未设置(全部可见) | GPU 0 |
| B | 0 | 1 | 1 | 0 | 1 | 未设置(全部可见) | GPU 1 |
| C | 1 | 2 | 0 | 1 | 2 | 未设置(全部可见) | GPU 2 |
| D | 1 | 3 | 1 | 1 | 3 | 未设置(全部可见) | GPU 3 |

所有进程共享同一 Level Zero 设备命名空间 → XCCL IPC 句柄在同一 driver 上下文内交换 → allreduce 正常完成。
