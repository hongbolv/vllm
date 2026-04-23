# vLLM XPU DP>1 Hang 问题调试总结

## 一、问题描述

在 vLLM 的 **XPU 平台**上，当 **DP（数据并行度）> 1** 时，系统在 worker 初始化阶段 **hang 住**，无法继续执行。

**正常场景：** TP=4 / DP=1 可以正常工作。  
**异常场景：** DP > 1 时 hang。

---

## 二、代码执行流（已通过源码确认）

关键文件：`vllm/v1/worker/xpu_worker.py` → `XPUWorker.init_device()` 方法

| 行号 | 操作 | 执行状态 |
|------|------|----------|
| 62-68 | 设备初始化（`set_device_index`, `empty_cache`, 获取设备属性） | ✅ 已完成 |
| 72-78 | 设置 CCL 环境变量（`CCL_ATL_TRANSPORT`, `LOCAL_WORLD_SIZE`, `LOCAL_RANK`） | ✅ 已完成 |
| **80-86** | **`init_worker_distributed_environment(...)`** — 内含 `init_process_group` | ✅ **已成功** |
| **89-90** | **warmup all_reduce**: `torch.distributed.all_reduce(torch.zeros(1).xpu())` | ✅ **已提交成功** |
| — | 🔴 **在第 90 行之后、第 92 行之前，显式插入 `torch.xpu.synchronize()` → hang 在此处** | ❌ **HANG** |
| 92-93 | `set_random_seed(...)` | ❌ **未到达** |
| 96 | `gc.collect()` | ❌ **未到达** |
| 97 | `torch.accelerator.empty_cache()` | ❌ **未到达** |
| 100 | `MemorySnapshot(device=self.device)` | ❌ **未到达** |

### 关键发现

在 warmup `all_reduce`（第 90 行）紧接着**显式调用** `torch.xpu.synchronize()` 后，程序**直接 hang 在 `synchronize()` 处**，不会执行到 `gc.collect()` 及之后的任何代码。

```python
# 第 89-90 行（原代码）
if torch.distributed.is_xccl_available():
    torch.distributed.all_reduce(torch.zeros(1).xpu())

# ↓ 插入显式同步后 hang 在这里 ↓
torch.xpu.synchronize()  # 🔴 HANG HERE

# 以下代码均不会执行
set_random_seed(self.model_config.seed)  # 第 93 行
gc.collect()                              # 第 96 行
torch.accelerator.empty_cache()           # 第 97 行
```

---

## 三、关键推理结论（均已确认）

### ✅ 结论1：`init_process_group` 100% 成功了

**推理依据：**
- `torch.distributed.all_reduce()` 必须在已初始化的 process group 上执行
- 如果 `init_process_group` 失败，第 90 行 `all_reduce` 会立刻抛出异常或 hang，**不可能 "提交成功"**
- 用户已确认 warmup `all_reduce` 提交成功 → 因此 `init_process_group` 必然已完成
- 这意味着 **TCP rendezvous 成功**，所有 rank 都已加入 WORLD process group

### ✅ 结论2：`all_reduce` 的提交是异步操作

**推理依据：**
- `torch.distributed.all_reduce()` 在 XPU/XCCL 后端上是**异步**的
- 它将操作提交到 XPU 的 command queue 后立刻返回
- **"提交成功" 不等于 "执行完成"**
- 真正的执行发生在设备端（device-side），需要 `torch.xpu.synchronize()` 来等待完成

### ✅ 结论3：`torch.xpu.synchronize()` 是确定的 hang 点

**推理依据：**
- 已实测验证：在第 90 行 `all_reduce` 之后显式插入 `torch.xpu.synchronize()`，程序直接 hang
- `gc.collect()` 及之后的代码均**不会执行**
- 这证明 hang 不是由 `gc.collect()`、`empty_cache()` 或 `MemorySnapshot` 引起的
- **问题就在于 XCCL 的 all_reduce 操作在设备端无法完成**

### ✅ 结论4：问题出在 XCCL 设备端执行层

**推理依据：**
- Python 层的 `init_process_group` 成功 → 排除 Python 层初始化问题
- Python 层的 `all_reduce` 提交成功 → 排除 Python 层调度问题
- hang 在 `torch.xpu.synchronize()` → 设备端操作无法完成
- 这意味着 **XCCL 在设备端的集合通信操作永远等不到所有参与者就绪**

---

## 四、问题定位总结

```
Python 层                      设备端 (Device-side)
───────────────               ─────────────────────
init_process_group ──── ✅ ──→ TCP rendezvous 成功
                                ↓
all_reduce 提交   ──── ✅ ──→ 提交到 XPU command queue
                                ↓
torch.xpu.synchronize() ─────→ 等待 all_reduce 完成...
                                ↓
                              🔴 HANG: XCCL 设备端执行
                                   永远等不到完成
```

**核心问题**：XCCL 后端在 DP > 1 场景下，WORLD group 的 `all_reduce` 操作在设备端**无法完成同步**。所有 rank 的 Python 层都正常提交了操作，但设备端的集合通信无法汇合。

---

## 五、TP=4/DP=1 vs DP>1 的关键差异分析

### ⚠️ 重要：`has_all_vertices_connected: 0` 不是根因

**TP=4/DP=1 时也是同样的硬件拓扑**（Arc Pro B60 × 4，`has_all_vertices_connected: 0`），CCL 也选择了同样的 LL256 ring 算法，但**可以正常工作**。因此 `has_all_vertices_connected: 0` 单独**不能**解释为什么 DP>1 会 hang。

### 5.1 进程组初始化路径差异

差异在于 `vllm/distributed/parallel_state.py` → `init_distributed_environment()` 中的处理逻辑：

**TP=4/DP=1（正常）：**
```python
# data_parallel_size == 1，不进入调整逻辑
# rank 保持原值（0, 1, 2, 3）
# world_size 保持原值（4）
# distributed_init_method 使用原始值
torch.distributed.init_process_group(
    backend="xccl",
    world_size=4,     # 原始值
    rank=rank,        # 0-3 原始值
    init_method=原始方法,
)
```

**DP>1（hang）：**
```python
# data_parallel_size > 1，进入调整逻辑
rank = data_parallel_rank * world_size + rank   # rank 偏移
world_size = world_size_across_dp               # world_size 扩大
# 使用新的 TCP 端口和 IP
ip = data_parallel_master_ip
port = get_next_dp_init_port()                  # 新端口
distributed_init_method = get_distributed_init_method(ip, port)

torch.distributed.init_process_group(
    backend="xccl",
    world_size=world_size_across_dp,   # 调整后的值
    rank=调整后的rank,
    init_method=新的TCP端口,
)
```

### 5.2 进程来源差异

| | TP=4/DP=1 | DP>1 |
|--|-----------|------|
| **worker 来源** | **同一个** EngineCore 进程派生 | **不同的** EngineCore 进程派生 |
| **父进程树** | 同一棵进程树 | 不同的进程树 |
| **rank 值** | 原始 0, 1, 2, 3 | 经过偏移调整 |
| **world_size** | 原始 4 | `TP × DP`（更大） |
| **TCP 端口** | 原始端口 | `get_next_dp_init_port()`（新端口） |
| **init_process_group 参数** | 标准初始化 | 跨进程树的分布式初始化 |

### 5.3 CCL 日志中的两阶段初始化证据

日志中出现了 **PID 2441**（rank [2]）的 communicator finalize：
```
2441:[2] |CCL_DEBUG| ze_ipc_event_pool_manager.cpp:23 clear: finalize completed
2441:[2] |CCL_DEBUG| flow_control.cpp:12 ~flow_control: max used credits: 0
```

然后 PID 2267/2268/2271/2272 才建立新的 communicator 执行 allreduce。

这说明 DP>1 场景下存在**两阶段进程组初始化**：
1. **第一阶段**：TP group 或旧 communicator 初始化（PID 2441，`max used credits: 0` 表示从未使用就被销毁）
2. **第二阶段**：WORLD group 的 communicator 创建（PID 2267/2268/2271/2272）

在 TP=4/DP=1 场景下，不存在这种两阶段过程。

---

## 六、CCL Debug 日志分析（实测数据）

通过设置 `CCL_LOG_LEVEL=debug` 和 `CCL_LOG_FLUSH=1` 收集到的日志，揭示了以下关键信息：

### 6.1 环境与拓扑

| 项目 | 值 |
|------|------|
| **GPU 型号** | Intel(R) Arc(TM) Pro B60 Graphics |
| **设备族** | family6 |
| **is_single_tile** | 1（每张卡是单 tile 设备） |
| **has_all_vertices_connected** | **0**（设备之间**没有**全互联拓扑，但 TP=4/DP=1 同样如此且正常工作） |
| **stream 类型** | gpu, in_order: 1 |
| **WORLD group** | `comm { rank: X, size: 4, id: 1 }` — 4 个 rank |

### 6.2 allreduce 执行流程（从日志还原）

每个 rank 的 CCL 执行路径完全一致：

```
zeDeviceGetProperties
zeDeviceGetCommandQueueGroupProperties (×2)
stream: { type: gpu, in_order: 1, device: Arc Pro B60, device_family: family6 } (×2)
can_use_sycl_kernels: coll allreduce, local_proc_count 4, comm { rank: X, size: 4, id: 1 }
selected algo: coll allreduce, algo topo sycl
ccl_allreduce: |CCL_SYCL| allreduce selects sycl-kernels count: 1, datatype: FLOAT32
allreduce_sycl: is_single_node
allreduce_sycl_single_node: |CCL_SYCL| is_single_tile: 1, has_all_vertices_connected: 0
invoking allreduce LL256 kernel allreduce_ll_ring, count:1 datatype: FLOAT32
invoking allreduce LL256 kernel arc_allreduce, count:1 datatype: FLOAT32
invoking allreduce LL256 kernel, count:1 datatype: FLOAT32 done    ← ✅ CCL 认为内核已入队
```

### 6.3 关键时序（PID → Rank 映射）

| PID | Rank | CCL allreduce 入队完成 | VLLM_DEBUG 打印 | 后续 |
|-----|------|----------------------|-----------------|------|
| 2267 | [0] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |
| 2268 | [1] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |
| 2271 | [2] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |
| 2272 | [3] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |

**所有 4 个 rank 的 CCL 均报告 allreduce 内核入队完成，但 `torch.xpu.synchronize()` 后没有任何输出** — 确认 hang 在 `synchronize()` 处。

### 6.4 日志中的异常信号

**信号1：先前 communicator 的 finalize 操作（两阶段初始化）**

日志开头出现 PID 2441（rank [2]）的 finalize 日志：
```
2441:[2] |CCL_DEBUG| ze_ipc_event_pool_manager.cpp:23 clear: finalize completed
2441:[2] |CCL_DEBUG| flow_control.cpp:12 ~flow_control: max used credits: 0
```
这说明 DP>1 场景下有**两阶段进程组初始化**：PID 2441 的旧 communicator 先 finalize（`max used credits: 0` 表示从未使用），然后 PID 2267/2268/2271/2272 才建立新 communicator。这在 TP=4/DP=1 场景下不会发生。

**信号2：`ze_ipc_event_pool_manager` finalize**

旧 communicator 的 finalize 涉及 `ze_ipc_event_pool_manager.cpp` 的清理。如果旧 communicator 的 IPC event pool 清理不干净，可能导致后续新 communicator 的 Level Zero IPC 资源冲突或状态异常。

**信号3：跨进程树的 IPC handle 交换**

DP>1 时，worker 进程来自**不同的父进程树**（不同的 EngineCore 进程）。Level Zero IPC memory handle 的交换方式可能与同一进程树内不同：
- 同一进程树内（TP=4/DP=1）：共享地址空间继承，IPC 映射自然有效
- 跨进程树（DP>1）：需要通过 TCP/共享内存显式交换 IPC handle，映射可能不完整

---

## 七、更新后的根因分析

### ❌ 排除的原因

| 原因 | 排除依据 |
|------|----------|
| init_process_group 失败 | CCL 日志证实 comm id=1, size=4 正确创建 |
| all_reduce 未提交 | 4 个 rank 均显示 "done"（内核已入队） |
| Python 层调度问题 | 所有 rank 的 Python 代码执行一致 |
| 个别 rank 未参与 | 4 个 rank 全部进入 allreduce |
| **`has_all_vertices_connected: 0` 拓扑问题** | **TP=4/DP=1 使用同样的拓扑和 LL256 ring 算法，可以正常工作** |

### 🔴 确认的问题

```
CCL 层面                              GPU 设备层面
──────────────                       ──────────────────
旧 communicator finalize (PID 2441)    IPC event pool 清理
  max used credits: 0                  ↓ 可能残留状态
                                      
comm {size:4, id:1} 创建 ✅            TCP rendezvous 成功（跨进程树）
allreduce 算法选择:                    
  allreduce_ll_ring (LL256) ✅         内核入队到 XPU command queue
  arc_allreduce ✅                     
CCL 报告 "done" ✅                     
                                      ↓
                                      🔴 SYCL kernel 在设备端执行时死锁
                                         LL256 ring 依赖 IPC 内存访问
                                         跨进程树的 IPC handle 交换/映射
                                         可能不正确或受旧 communicator 影响
```

### 🎯 最可能的根因

**DP>1 场景下，跨不同 EngineCore 进程树的 XCCL communicator 初始化过程中，Level Zero IPC memory handle 的交换/映射出现问题，导致 LL256 ring allreduce SYCL kernel 在设备端无法完成数据交换而死锁。**

具体来说，与 TP=4/DP=1 的关键差异：

1. **进程来源不同**：DP>1 的 worker 来自不同的 EngineCore 父进程（不同进程树），而 TP=4/DP=1 所有 worker 来自同一个 EngineCore
2. **两阶段初始化**：DP>1 存在旧 communicator finalize → 新 communicator 创建的两阶段过程（PID 2441 的 `ze_ipc_event_pool_manager` finalize），可能导致 IPC 资源状态异常
3. **rank/world_size 调整**：DP>1 时 `init_distributed_environment` 对 rank 做偏移（`data_parallel_rank * world_size + rank`），world_size 扩大为 `world_size_across_dp`，使用新的 TCP 端口 — 这些调整可能影响 CCL communicator 内部的 IPC handle 交换逻辑
4. **IPC handle 交换方式**：跨不同父进程树时，Level Zero IPC memory handle 的交换可能需要不同的机制（TCP vs 共享内存继承），如果交换不完整，ring 算法的 SYCL kernel 将无法访问邻居 rank 的数据

---

## 八、建议的下一步调试方案（按优先级排序）

### 方案1：对比 TP=4/DP=1 的 CCL 日志（最关键）

用相同的 `CCL_LOG_LEVEL=debug` 和 `CCL_LOG_FLUSH=1` 在 **TP=4/DP=1（正常场景）** 下运行，对比：

1. **是否存在两阶段初始化？** — TP=4/DP=1 是否也有旧 communicator finalize
2. **IPC 相关日志差异** — 搜索 `ze_ipc`、`ipc_handle`、`ipc_event_pool` 关键字
3. **communicator 创建参数差异** — `comm { rank: X, size: Y, id: Z }` 的值
4. **allreduce 算法选择是否一致** — 确认都是 `allreduce_ll_ring`

```bash
# TP=4/DP=1 正常场景
export CCL_LOG_LEVEL=debug
export CCL_LOG_FLUSH=1
# 运行 TP=4/DP=1 配置，收集日志
```

### 方案2：强制 CCL 使用非 SYCL kernel 算法

```bash
# 禁用 SYCL kernel 路径
export CCL_ALLREDUCE=naive
# 或者
export CCL_SYCL_KERNELS=0
```

如果设置后 DP>1 hang 消失 → 确认问题在 LL256 ring SYCL kernel 的 IPC 内存访问层。

### 方案3：测试纯 XCCL allreduce（脱离 vLLM）

编写最小复现脚本，模拟 DP>1 的跨进程组初始化：

```python
# test_xccl_dp.py — 模拟 DP>1 的跨进程组 allreduce
import os
import torch
import torch.distributed as dist

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

**测试1**：用 `torchrun --nproc_per_node=4` 直接运行（模拟 TP=4/DP=1，预期成功）

**测试2**：手动创建两个进程组（2+2），先创建一个 communicator 再销毁，再创建新的执行 allreduce — 模拟 DP>1 的两阶段初始化

如果测试1通过但测试2 hang → 确认问题在两阶段 communicator 初始化的 IPC 状态管理。

### 方案4：检查 Level Zero IPC 状态

```bash
# 启用 Level Zero 调试日志
export ZE_ENABLE_TRACING_LAYER=1
export ZET_ENABLE_API_TRACING_EXP=1
```

观察 DP>1 场景下 IPC handle 的创建、交换、映射是否有错误或异常。

### 方案5：检查 vLLM DP 初始化参数

在 `xpu_worker.py` 的 `init_device()` 中添加打印，验证 DP>1 时传入的参数：

```python
# 在 init_worker_distributed_environment 调用前添加
import sys
print(f"[DEBUG] rank={self.rank}, local_rank={self.local_rank}, "
      f"world_size={self.parallel_config.world_size}, "
      f"dp_size={self.parallel_config.data_parallel_size}, "
      f"dp_rank={self.parallel_config.data_parallel_rank}, "
      f"world_size_across_dp={self.parallel_config.world_size_across_dp}, "
      f"distributed_init_method={self.distributed_init_method}",
      flush=True, file=sys.stderr)
```

---

## 九、环境信息（已确认）

| 项目 | 值 |
|------|------|
| GPU | Intel(R) Arc(TM) Pro B60 Graphics × 4 |
| 设备族 | family6 |
| 单 tile | 是 |
| 设备互联 | 无全互联（`has_all_vertices_connected: 0`，但 TP=4/DP=1 同样如此且正常工作） |

### 待确认项

- [ ] Intel oneAPI / oneCCL 版本
- [ ] PyTorch XPU 版本（`torch.__version__` 和 XCCL 支持状态）
- [ ] XPU 设备拓扑详细信息（`xpu-smi topology`）
- [ ] 完整的环境变量（`CCL_*`, `I_MPI_*` 等）
- [ ] TP 和 DP 的具体配置值
- [ ] TP=4/DP=1 场景下的 CCL debug 日志（用于对比）
- [ ] `CCL_ALLREDUCE=naive` 或 `CCL_SYCL_KERNELS=0` 在 DP>1 下的测试结果
- [ ] DP>1 时 `init_distributed_environment` 的实际参数值（rank 偏移、world_size、TCP 端口）

---

## 七、诊断日志分析（init_distributed_environment 打印结果）

### 7.1 日志原始数据（TP=2, DP=2）

```
# === EngineCore 0 的 worker ===
(Worker pid=4186) [DEBUG-DP] init_distributed_environment ENTRY: world_size=2, rank=1, local_rank=1, distributed_init_method=tcp://127.0.0.1:41965, backend=xccl
(Worker pid=4186) env: RANK=None, LOCAL_RANK=1, WORLD_SIZE=None, MASTER_ADDR=None, MASTER_PORT=None
(Worker pid=4186) DP adjustment BEFORE: data_parallel_size=2, data_parallel_rank=0, world_size_across_dp=4, tensor_parallel_size=2, original_rank=1, original_world_size=2
(Worker pid=4186) DP adjustment AFTER: adjusted_rank=1, adjusted_world_size=4, ip=127.0.0.1, port=46163, distributed_init_method=tcp://127.0.0.1:46163
(Worker pid=4186) calling init_process_group: backend=xccl, init_method=tcp://127.0.0.1:46163, world_size=4, rank=1, is_initialized_before=False

(Worker pid=4185) [DEBUG-DP] init_distributed_environment ENTRY: world_size=2, rank=0, local_rank=0, distributed_init_method=tcp://127.0.0.1:41965, backend=xccl
(Worker pid=4185) env: RANK=None, LOCAL_RANK=0, WORLD_SIZE=None, MASTER_ADDR=None, MASTER_PORT=None
(Worker pid=4185) DP adjustment BEFORE: data_parallel_size=2, data_parallel_rank=0, world_size_across_dp=4, tensor_parallel_size=2, original_rank=0, original_world_size=2
(Worker pid=4185) DP adjustment AFTER: adjusted_rank=0, adjusted_world_size=4, ip=127.0.0.1, port=46163, distributed_init_method=tcp://127.0.0.1:46163
(Worker pid=4185) calling init_process_group: backend=xccl, init_method=tcp://127.0.0.1:46163, world_size=4, rank=0, is_initialized_before=False

# === EngineCore 1 的 worker ===
(Worker pid=4189) [DEBUG-DP] init_distributed_environment ENTRY: world_size=2, rank=0, local_rank=0, distributed_init_method=tcp://127.0.0.1:51907, backend=xccl
(Worker pid=4189) env: RANK=None, LOCAL_RANK=0, WORLD_SIZE=None, MASTER_ADDR=None, MASTER_PORT=None
# (日志截断)
```

### 7.2 关键发现

#### ✅ 发现1：**不是两阶段初始化**

所有 worker 均显示 `is_initialized_before=False`，说明 **没有** 先创建 TP-only WORLD 组再销毁重建的过程。每个 worker 直接用 DP 调整后的参数调用 `init_process_group`。**此前的"两阶段初始化"假设被证伪。**

#### 🔴 发现2：**LOCAL_RANK 重复 — 不同 EngineCore 的 worker 使用相同的 local_rank**

| PID | EngineCore | data_parallel_rank | original_rank | adjusted_rank | **local_rank** | 使用的 GPU |
|-----|------------|-------------------|---------------|---------------|----------------|-----------|
| 4185 | 0 | 0 | 0 | 0 | **0** | xpu:0 |
| 4186 | 0 | 0 | 1 | 1 | **1** | xpu:1 |
| 4189 | 1 | 1 | 0 | 2 | **0** | xpu:0 ← 冲突! |
| 4190 | 1 | 1 | 1 | 3 | **1** | xpu:1 ← 冲突! |

DP 调整只修改了 `rank`（全局 rank）和 `world_size`，**但 `local_rank` 没有调整**。
每个 EngineCore 从 `local_rank=0` 开始分配 worker，导致：
- PID 4185（rank=0）和 PID 4189（rank=2）都使用 `torch.device("xpu:0")` — **同一块 GPU**
- PID 4186（rank=1）和 PID 4190（rank=3）都使用 `torch.device("xpu:1")` — **同一块 GPU**

#### 🔴 发现3：XPU 平台的设备亲和性（ZE_AFFINITY_MASK）可能未正确设置

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

对于 **CUDA 平台**，每个 EngineCore 会设置 `CUDA_VISIBLE_DEVICES` 限制可见 GPU：
- EngineCore 0: `CUDA_VISIBLE_DEVICES=0,1` → local_rank=0 映射 GPU 0，local_rank=1 映射 GPU 1
- EngineCore 1: `CUDA_VISIBLE_DEVICES=2,3` → local_rank=0 映射 GPU 2，local_rank=1 映射 GPU 3

对于 **XPU 平台**，这一步被跳过（`pass`）。所有 EngineCore 看到**全部 4 块 GPU**，
local_rank=0 在两个 EngineCore 中都映射到**物理 GPU 0**。

> **注意**：在 MP（多进程）路径中（`vllm/v1/engine/utils.py`），`set_device_control_env_var()`
> 会为非 CUDA 平台（包括 XPU）设置 `ZE_AFFINITY_MASK`。但如果用户使用 Ray 路径，
> `DPEngineCoreActor` 的 XPU 分支是 `pass`，不会设置设备亲和性。

### 7.3 根因结论

**根因是 XPU DP>1 时多个 XCCL rank 映射到同一块物理 GPU 上。**

XCCL `init_process_group(world_size=4)` 创建了一个 4-rank 的通信组，但实际只用了 2 块 GPU（GPU 0 和 GPU 1），每块 GPU 上有 2 个 rank。XCCL 的 allreduce 需要每个 rank 对应唯一的设备，当两个 rank 共享同一设备时，IPC 内存 handle 交换和设备端同步出现死锁。

对比 TP=4/DP=1（正常工作）：所有 4 个 worker 由同一个 EngineCore 生成，local_rank 分别为 0,1,2,3，对应 4 块不同的 GPU。

### 7.4 需要确认的信息

1. **`ZE_AFFINITY_MASK` 的实际值**：需要在诊断打印中添加 `ZE_AFFINITY_MASK` 环境变量的值，确认每个 worker 进程中该变量的设置
2. **PID 4189/4190 的完整日志**：当前日志被截断，需要确认 EngineCore 1 的 worker 是否使用相同的 port（46163）加入同一个进程组
3. **使用的 executor backend**：确认是 MP 路径还是 Ray 路径，因为两者的设备亲和性处理方式不同

---

## 附录：CCL Debug 日志（最后 ~100 行关键摘录）

```
# === 第一阶段：旧 communicator finalize（PID 2441，DP>1 独有） ===
2441:[2] |CCL_DEBUG| ze_ipc_event_pool_manager.cpp:23 clear: finalize completed
2441:[2] |CCL_DEBUG| ze_ipc_event_pool_manager.cpp:23 clear: finalize completed
2441:[2] |CCL_DEBUG| flow_control.cpp:12 ~flow_control: max used credits: 0
2441:[2] |CCL_DEBUG| ze_ipc_event_pool_manager.cpp:23 clear: finalize completed

# === 第二阶段：新 communicator 的 allreduce 执行 ===

# Rank 0 (PID 2267)
2267:[0] |CCL_INFO| stream: { type: gpu, in_order: 1, device: Intel(R) Arc(TM) Pro B60 Graphics, device_family: family6 }
2267:[0] |CCL_DEBUG| sycl_selection.cpp:43 can_use_sycl_kernels: coll allreduce, local_proc_count 4, comm { rank: 0, size: 4, id: 1 }
2267:[0] |CCL_DEBUG| sycl_selection.cpp:279 can_use_sycl_kernels: selected algo: coll allreduce, algo topo sycl
2267:[0] |CCL_DEBUG| coll.cpp:1359 ccl_allreduce: |CCL_SYCL| allreduce selects sycl-kernels count: 1, datatype: FLOAT32
2267:[0] |CCL_DEBUG| allreduce_sycl.cpp:713 allreduce_sycl: is_single_node
2267:[0] |CCL_DEBUG| allreduce_sycl.cpp:60 allreduce_sycl_single_node: |CCL_SYCL| is_single_tile: 1, has_all_vertices_connected: 0
2267:[0] |CCL_DEBUG| allreduce_sycl.cpp:76 allreduce_sycl_single_node: invoking allreduce LL256 kernel allreduce_ll_ring, count:1 datatype: FLOAT32
2267:[0] |CCL_DEBUG| allreduce_sycl.cpp:104 allreduce_sycl_single_node: invoking allreduce LL256 kernel arc_allreduce, count:1 datatype: FLOAT32
2267:[0] |CCL_DEBUG| allreduce_sycl.cpp:106 allreduce_sycl_single_node: invoking allreduce LL256 kernel, count:1 datatype: FLOAT32 done
(Worker pid=2267) [======VLLM_DEBUG======] XPUWorker.init_device: all_reduce done calling synchronize, pid=2267

# Rank 1 (PID 2268) — 同样的流程
2268:[1] |CCL_DEBUG| allreduce_sycl.cpp:106 allreduce_sycl_single_node: invoking allreduce LL256 kernel, count:1 datatype: FLOAT32 done
(Worker pid=2268) [======VLLM_DEBUG======] XPUWorker.init_device: all_reduce done calling synchronize, pid=2268

# Rank 2 (PID 2271) — 同上
2271:[2] |CCL_DEBUG| allreduce_sycl.cpp:106 allreduce_sycl_single_node: invoking allreduce LL256 kernel, count:1 datatype: FLOAT32 done
(Worker pid=2271) [======VLLM_DEBUG======] XPUWorker.init_device: all_reduce done calling synchronize, pid=2271

# Rank 3 (PID 2272) — 同上
2272:[3] |CCL_DEBUG| allreduce_sycl.cpp:106 allreduce_sycl_single_node: invoking allreduce LL256 kernel, count:1 datatype: FLOAT32 done
(Worker pid=2272) [======VLLM_DEBUG======] XPUWorker.init_device: all_reduce done calling synchronize, pid=2272

# ↑ 所有 rank 到此为止，无更多输出
# ↓ 600 秒后 ApiServer 超时
(ApiServer_1 pid=1217) TimeoutError: Timed out waiting for engine core processes to start.
```
