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

## 五、CCL Debug 日志分析（实测数据）

通过设置 `CCL_LOG_LEVEL=debug` 和 `CCL_LOG_FLUSH=1` 收集到的日志，揭示了以下关键信息：

### 5.1 环境与拓扑

| 项目 | 值 |
|------|------|
| **GPU 型号** | Intel(R) Arc(TM) Pro B60 Graphics |
| **设备族** | family6 |
| **is_single_tile** | 1（每张卡是单 tile 设备） |
| **has_all_vertices_connected** | **0**（设备之间**没有**全互联拓扑） |
| **stream 类型** | gpu, in_order: 1 |
| **WORLD group** | `comm { rank: X, size: 4, id: 1 }` — 4 个 rank |

### 5.2 allreduce 执行流程（从日志还原）

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

### 5.3 关键时序（PID → Rank 映射）

| PID | Rank | CCL allreduce 入队完成 | VLLM_DEBUG 打印 | 后续 |
|-----|------|----------------------|-----------------|------|
| 2267 | [0] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |
| 2268 | [1] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |
| 2271 | [2] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |
| 2272 | [3] | ✅ "done" | ✅ "all_reduce done calling synchronize" | 🔴 无更多输出 |

**所有 4 个 rank 的 CCL 均报告 allreduce 内核入队完成，但 `torch.xpu.synchronize()` 后没有任何输出** — 确认 hang 在 `synchronize()` 处。

### 5.4 日志中的异常信号

**信号1：先前 communicator 的 finalize 操作**

日志开头出现 PID 2441（rank [2]）的 finalize 日志：
```
2441:[2] |CCL_DEBUG| ze_ipc_event_pool_manager.cpp:23 clear: finalize completed
2441:[2] |CCL_DEBUG| flow_control.cpp:12 ~flow_control: max used credits: 0
```
这说明在 warmup all_reduce 之前，已有一个旧的 CCL communicator 被销毁（可能来自 TP group 的 init_process_group）。**`max used credits: 0`** 表示该 communicator 从未被使用过。

**信号2：`has_all_vertices_connected: 0`**

Arc Pro B60 是消费级/专业级 GPU，**没有 GPU 间的高速互联**（不像数据中心 GPU 有 NVLink/XELINK）。CCL 检测到设备间没有全互联拓扑。

**信号3：算法选择了 `allreduce_ll_ring`**

尽管 `has_all_vertices_connected: 0`，CCL 仍然选择了 **LL256 ring 算法**（`allreduce_ll_ring` + `arc_allreduce`）。Ring 算法依赖设备间的直接内存访问。如果设备间没有建立正确的 IPC（Inter-Process Communication）通道，ring 算法的 SYCL kernel 将在设备端死锁 — 每个 rank 等待从邻居读取数据，但邻居的数据永远不可达。

---

## 六、更新后的根因分析

### ❌ 排除的原因

| 原因 | 排除依据 |
|------|----------|
| init_process_group 失败 | CCL 日志证实 comm id=1, size=4 正确创建 |
| all_reduce 未提交 | 4 个 rank 均显示 "done"（内核已入队） |
| Python 层调度问题 | 所有 rank 的 Python 代码执行一致 |
| 个别 rank 未参与 | 4 个 rank 全部进入 allreduce |

### 🔴 确认的问题

```
CCL 层面                              GPU 设备层面
──────────────                       ──────────────────
comm {size:4, id:1} 创建 ✅            TCP rendezvous 成功
allreduce 算法选择:                    
  allreduce_ll_ring (LL256) ✅         内核入队到 XPU command queue
  arc_allreduce ✅                     
CCL 报告 "done" ✅                     
                                      ↓
                                      🔴 SYCL kernel 在设备端执行时死锁
                                         Ring 算法等待邻居数据
                                         但 IPC 通道可能未正确建立
                                         (has_all_vertices_connected: 0)
```

### 🎯 最可能的根因

**CCL 的 LL256 ring allreduce SYCL kernel 在 `has_all_vertices_connected: 0` 的拓扑下，IPC 内存映射未正确建立，导致 ring 通信死锁。**

具体来说：
1. CCL 检测到 `is_single_node` = true，`is_single_tile` = 1
2. 但 `has_all_vertices_connected` = 0 — 设备间没有直接互联
3. CCL 仍然选择了依赖设备间直接内存访问的 `allreduce_ll_ring` 算法
4. SYCL kernel 在设备端执行时，尝试通过 Level Zero IPC 读取邻居 rank 的数据
5. 如果 IPC handle 交换或内存映射有问题，kernel 将永远等待 — 表现为 `synchronize()` hang

这也解释了为什么 **TP=4/DP=1 正常但 DP>1 异常**：
- TP=4/DP=1 时 WORLD size = 4，所有 rank 可能使用同一套 IPC 映射
- DP>1 时可能涉及多组 communicator，IPC 映射可能冲突或未正确重建

---

## 七、建议的下一步调试方案

### 方案1：强制 CCL 使用非 IPC 算法（最快验证）

```bash
# 禁用 SYCL kernel 路径，回退到 CPU staging 或其他算法
export CCL_ALLREDUCE=naive
# 或者
export CCL_SYCL_KERNELS=0
```

如果设置后 hang 消失，则确认是 LL256 ring SYCL kernel 的 IPC 问题。

### 方案2：检查 Level Zero IPC 状态

```bash
# 启用 Level Zero 调试日志
export ZE_ENABLE_TRACING_LAYER=1
export ZET_ENABLE_API_TRACING_EXP=1
```

观察 IPC handle 的创建、交换、映射是否成功。

### 方案3：对比 TP=4/DP=1 的 CCL 日志

用相同的 `CCL_LOG_LEVEL=debug` 在 **TP=4/DP=1（正常场景）** 下运行，对比：
- communicator 的创建和 finalize 顺序
- IPC handle 交换日志
- allreduce 算法选择是否一致

### 方案4：验证 IPC 内存映射

在 `xpu_worker.py` 中添加简单的 IPC 测试：

```python
import intel_extension_for_pytorch  # noqa
import torch

# 测试基本的 IPC 内存操作
tensor = torch.zeros(1).xpu()
# 尝试获取 IPC handle
try:
    handle = torch.xpu.ipc_collect()
    logger.info("IPC collect succeeded")
except Exception as e:
    logger.error("IPC collect failed: %s", e)
```

### 方案5：测试纯 XCCL allreduce（脱离 vLLM）

编写最小复现脚本，排除 vLLM 框架的影响：

```python
# test_xccl.py — 用 torchrun --nproc_per_node=4 运行
import os
import torch
import torch.distributed as dist

os.environ["CCL_LOG_LEVEL"] = "debug"
os.environ["CCL_LOG_FLUSH"] = "1"

dist.init_process_group(backend="xccl")
rank = dist.get_rank()

print(f"Rank {rank}: init_process_group done", flush=True)

tensor = torch.zeros(1).xpu(rank)
dist.all_reduce(tensor)
print(f"Rank {rank}: all_reduce submitted", flush=True)

torch.xpu.synchronize()
print(f"Rank {rank}: synchronize done!", flush=True)

dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=4 test_xccl.py
```

如果最小脚本也 hang → 问题在 CCL/Level Zero 层，需要报告给 Intel。
如果最小脚本通过 → 问题在 vLLM 的初始化流程中（可能是多组 communicator 创建的副作用）。

---

## 八、环境信息（已确认）

| 项目 | 值 |
|------|------|
| GPU | Intel(R) Arc(TM) Pro B60 Graphics × 4 |
| 设备族 | family6 |
| 单 tile | 是 |
| 设备互联 | 无全互联（`has_all_vertices_connected: 0`） |

### 待确认项

- [ ] Intel oneAPI / oneCCL 版本
- [ ] PyTorch XPU 版本（`torch.__version__` 和 XCCL 支持状态）
- [ ] XPU 设备拓扑详细信息（`xpu-smi topology`）
- [ ] 完整的环境变量（`CCL_*`, `I_MPI_*` 等）
- [ ] TP 和 DP 的具体配置值
- [ ] `CCL_ALLREDUCE=naive` 或 `CCL_SYCL_KERNELS=0` 测试结果

---

## 附录：CCL Debug 日志（最后 ~100 行关键摘录）

```
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

# Rank 1 (PID 2268) — 同样的流程，最后到达
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
