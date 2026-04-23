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
9. **诊断日志确认 LOCAL_RANK 冲突** — 不同 EngineCore 的 worker 均使用 `local_rank=0`，多个 XCCL rank 映射到同一物理 GPU

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

### 4.2 确认的根因：LOCAL_RANK 冲突导致多个 XCCL rank 映射到同一物理 GPU

```
                           EngineCore 0                    EngineCore 1
                           ──────────────                  ──────────────
                           Worker 0: local_rank=0 → GPU 0  Worker 2: local_rank=0 → GPU 0 ← 冲突!
                           Worker 1: local_rank=1 → GPU 1  Worker 3: local_rank=1 → GPU 1 ← 冲突!

                           ↓                               ↓
                           XCCL init_process_group(world_size=4) 创建 4-rank 通信组
                           但只使用了 2 块物理 GPU，每块 GPU 上有 2 个 rank
                           ↓
                           allreduce LL256 ring kernel 需要每个 rank 对应唯一设备
                           设备端 IPC 内存交换死锁 → torch.xpu.synchronize() 永久 hang
```

**对比 TP=4/DP=1（正常）：**
所有 4 个 worker 由同一个 EngineCore 生成，`local_rank` 分别为 0, 1, 2, 3，一一对应 4 块不同的物理 GPU。

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

### 5.4 如何确认 LOCAL_RANK 冲突是根因？

确认 LOCAL_RANK 冲突是根因，基于以下**三重证据链**：

**证据1：诊断日志直接观测到 local_rank 重复（第六节详细数据）**

诊断打印明确显示 EngineCore 0 的 worker（PID 4185）和 EngineCore 1 的 worker（PID 4189）都使用 `local_rank=0`，EngineCore 0 的 worker（PID 4186）和 EngineCore 1 的 worker（PID 4190）都使用 `local_rank=1`。4 个 XCCL rank 只映射到 2 块物理 GPU。

**证据2：对比排除法 — 所有其他可能原因均已排除**

| 排除的假设 | 排除依据 |
|-----------|---------|
| `has_all_vertices_connected: 0` 拓扑问题 | TP=4/DP=1 同样 `=0` 且正常工作 |
| SYCL kernel 路径问题 | `CCL_SYCL_KERNELS=0` 仍 hang |
| PyTorch/XCCL 自身缺陷 | `test_xccl.py` 用 `torchrun` 正常工作 |
| 两阶段 init_process_group | 日志显示 `is_initialized_before=False`，无 destroy/rebuild |
| TCP rendezvous 失败 | `init_process_group` 成功完成 |
| allreduce 提交失败 | CCL 报告所有 4 rank 的内核入队 "done" |

所有假设被排除后，**唯一剩余的差异**就是 `local_rank` 分配：TP=4/DP=1 时 `local_rank` 为 0,1,2,3（唯一），DP>1 时为 0,1,0,1（冲突）。

**证据3：代码级根因链条完整**

1. `vllm/v1/engine/core.py` → `DPEngineCoreActor._set_visible_devices()` 中 XPU 分支是 `pass`（不设置 `ZE_AFFINITY_MASK`）
2. `vllm/distributed/parallel_state.py` → `init_distributed_environment()` 中 DP 调整只修改全局 `rank` 和 `world_size`，**不修改 `local_rank`**
3. `vllm/v1/worker/xpu_worker.py` → `XPUWorker.init_device()` 中使用 `local_rank` 调用 `torch.xpu.set_device()`
4. 结果：`local_rank=0` 的两个 worker（rank 0 和 rank 2）都调用 `torch.xpu.set_device(0)` → 都绑定到物理 GPU 0
5. XCCL communicator 创建时，4 个 rank 声称使用 4 块不同设备，但实际只用了 2 块 → 设备拓扑不一致 → allreduce ring kernel 在设备端死锁

这三重证据（直接观测 + 排除法 + 代码链条）共同确认 LOCAL_RANK 冲突是根因。

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

#### 🔴 发现2：LOCAL_RANK 冲突确认

| PID | EngineCore | dp_rank | 原始 rank | **调整后 rank** | **local_rank** | **映射 GPU** |
|-----|------------|---------|-----------|----------------|----------------|-------------|
| 4185 | 0 | 0 | 0 | 0 | **0** | xpu:**0** |
| 4186 | 0 | 0 | 1 | 1 | **1** | xpu:**1** |
| 4189 | 1 | 1 | 0 | 2 | **0** | xpu:**0** ← 冲突 |
| 4190 | 1 | 1 | 1 | 3 | **1** | xpu:**1** ← 冲突 |

DP 调整只修改了全局 `rank`（偏移为 `dp_rank * world_size + rank`）和 `world_size`，
**但 `local_rank` 没有任何调整**，仍然是每个 EngineCore 内部的 0-indexed 值。

#### 🔴 发现3：XPU 平台缺少 `ZE_AFFINITY_MASK` 设置

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
| **local_rank** | 0, 1, 2, 3（唯一） | 0, 1, 0, 1（冲突） |
| **GPU 映射** | 4 rank → 4 GPU | 4 rank → 2 GPU |
| **rank 值** | 原始 0-3 | 偏移调整后 0-3 |
| **world_size** | 原始 4 | `TP × DP` |
| **TCP 端口** | 原始端口 | 新端口 |
| **ZE_AFFINITY_MASK** | 不需要（all GPU visible） | ⚠️ 未设置（应该设置） |

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

## 九、根因结论

### 🎯 确定的根因

**XPU DP>1 时，多个 XCCL rank 映射到同一块物理 GPU 上，导致 allreduce 在设备端死锁。**

具体机制：
1. DP>1 从多个 EngineCore 进程树生成 worker
2. 每个 EngineCore 内部 `local_rank` 从 0 开始分配
3. DP 调整逻辑只修改了全局 `rank` 和 `world_size`，**未调整 `local_rank`**
4. XPU 平台的 `_set_visible_devices()` 不设置 `ZE_AFFINITY_MASK`（是 `pass`）
5. 结果：多个 rank 使用相同的 `local_rank` → `torch.device("xpu:0")` 被多个 rank 共享
6. XCCL communicator 创建了错误的设备拓扑映射
7. allreduce LL256 ring kernel 的 IPC 内存交换在设备端死锁

### 修复方向

需要确保每个 EngineCore 的 worker 使用正确的物理 GPU，有两种可能的方式：

1. **设置 `ZE_AFFINITY_MASK`**：在 `DPEngineCoreActor._set_visible_devices()` 中为 XPU 平台设置 `ZE_AFFINITY_MASK`，限制每个 EngineCore 可见的 GPU 设备
2. **调整 `local_rank`**：在 DP 调整逻辑中，根据 `data_parallel_rank` 偏移 `local_rank`，使其正确映射到不同的物理 GPU

---

## 十、环境信息

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
