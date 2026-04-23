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

## 五、可能的根因方向

| 方向 | 说明 | 可能性 |
|------|------|--------|
| **XCCL communicator 跨 DP group 通信链路问题** | `init_process_group` 在 Python 层成功（TCP rendezvous），但 XCCL 底层 communicator 在设备端的通信通道可能未正确建立跨 DP group 的链路 | 高 |
| **不同 DP group 的 rank 提交时序差异** | 某些 rank 的 XPU queue 中可能有大量待执行操作，导致 all_reduce 在设备端执行时等不齐 | 中 |
| **XCCL 在 multi-group/跨节点场景的 bug** | TP=4/DP=1 正常，DP>1 异常 → XCCL 可能在处理更大 world size 或跨组通信时有 bug | 高 |
| **CCL 环境变量配置不匹配** | `CCL_ATL_TRANSPORT` 或 `LOCAL_WORLD_SIZE` 等变量在 DP>1 时可能需要不同配置 | 中 |

---

## 六、建议的下一步调试方案

### 方案1：启用 XCCL 详细日志

```bash
export CCL_LOG_LEVEL=debug
export CCL_LOG_TO_STDOUT=1
```

运行后观察每个 rank 的 CCL 日志，确认：
- 每个 rank 的 communicator 是否正确创建
- 设备端通信通道是否建立
- all_reduce 操作在哪一步等待

### 方案2：逐 rank 检查同步状态

在 `xpu_worker.py` 第 89 行附近添加：

```python
if torch.distributed.is_xccl_available():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logger.info("Rank %d/%d: submitting warmup all_reduce...", rank, world_size)

    # barrier 先确保所有 rank 同时开始
    torch.distributed.barrier()
    logger.info("Rank %d/%d: barrier passed, submitting all_reduce...", rank, world_size)

    torch.distributed.all_reduce(torch.zeros(1).xpu())
    logger.info("Rank %d/%d: all_reduce submitted, synchronizing...", rank, world_size)

    torch.xpu.synchronize()
    logger.info("Rank %d/%d: all_reduce completed!", rank, world_size)
```

### 方案3：测试点对点通信

用 `send/recv` 替代 `all_reduce`，判断是集合通信问题还是所有通信都有问题：

```python
if torch.distributed.is_xccl_available():
    rank = torch.distributed.get_rank()
    if rank == 0:
        torch.distributed.send(torch.zeros(1).xpu(), dst=1)
    elif rank == 1:
        torch.distributed.recv(torch.zeros(1).xpu(), src=0)
    torch.xpu.synchronize()
    logger.info("Rank %d: point-to-point test passed!", rank)
```

### 方案4：缩小 world size 测试

用最小的 world_size=2 测试，排除规模因素：

```bash
# 只用 2 个 XPU 设备
torchrun --nproc_per_node=2 ...
```

---

## 七、环境信息检查清单

调试时请确认以下信息：

- [ ] Intel oneAPI / oneCCL 版本
- [ ] PyTorch XPU 版本（`torch.__version__` 和 XCCL 支持状态）
- [ ] XPU 设备数量和拓扑（`xpu-smi` 输出）
- [ ] 网络配置（如果跨节点）
- [ ] 完整的环境变量（`CCL_*`, `I_MPI_*` 等）
- [ ] TP 和 DP 的具体配置值
- [ ] 每个 rank 的日志输出
