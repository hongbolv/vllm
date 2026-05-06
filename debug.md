# Debug 分析: all_gatherv hang 问题

## 1. 问题概述

- **test_all_gatherv 测试 hang**
- **Qwen3.5 MoE 场景失败（hang）**
- **Qwen3 MoE 场景成功**

## 2. all_gatherv 实现分析

### 2.1 底层实现 (`pynccl.py:211-248`)

```python
def all_gatherv(self, output_tensor, input_tensor, sizes, stream=None):
    self.nccl.ncclGroupStart()
    for root, split_size in enumerate(sizes):
        dst_slice = output_tensor[split_offset : split_offset + split_size]
        sendbuf = input_tensor if root == self.rank else dst_slice
        self.nccl.ncclBroadcast(sendbuf, dst_slice, ...)  # N次广播
    self.nccl.ncclGroupEnd()
```

**关键点**: `all_gatherv` 通过 N 次 `ncclBroadcast`（封装在 `ncclGroupStart/End` 中）实现。这是**集体通信操作**，要求**所有 rank 必须同时参与**。

### 2.2 高层包装 (`cuda_communicator.py:346-395`)

```python
def all_gatherv(self, input_, dim=0, sizes=None):
    # 如果所有 sizes 相同，退化为普通 all_gather
    if sizes is not None and all(s == sizes[0] for s in sizes):
        sizes = None
    # sizes != None 时使用 pynccl.all_gatherv (基于 broadcast)
    # sizes == None 时使用 pynccl.all_gather (使用 ncclAllGather)
```

**重要优化**: 当所有 rank 的 sizes 相同时，退化为标准 `all_gather`。

## 3. test_all_gatherv 测试分析

### 3.1 测试代码 (`tests/distributed/test_pynccl.py:218-249`)

```python
@worker_fn_wrapper
def all_gatherv_worker_fn():
    sizes = [81, 20, 57, 52, 81, 5, 49, 49][:world_size]  # world_size=2 → [81, 20]
    num_elems = sizes[rank]  # rank0: 81个元素, rank1: 20个元素
    tensor = torch.arange(num_elems, ...)
    result = torch.zeros(sum(sizes), ...)  # 101个元素
    pynccl_comm.all_gatherv(result, tensor, sizes=sizes)
```

**测试特点**:
- ✅ **数据量非对称**: rank0 有 81 个元素，rank1 有 20 个元素
- ✅ **所有 rank 都参与调用**: 两个 rank 都执行 `all_gatherv`
- ✅ **所有 rank 共享相同的 sizes 列表**: `sizes = [81, 20]`

**这是一个"数据量非对称、但参与对称"的测试。**

### 3.2 其他 comprehensive 测试

| 测试名 | sizes (2 GPUs) | 特点 |
|---|---|---|
| `all_gatherv_worker_fn` | [81, 20] | 非对称数据量 |
| `equal_sizes` | [64, 64] | 对称（会退化为 all_gather） |
| `single_element` | [1, 1] | 对称（会退化为 all_gather） |
| `large_imbalance` | [1, 1000] | 极度非对称数据量 |
| `float16` | [30, 50] | 非对称 + 不同 dtype |

**所有测试的共同点**: 每个 rank 都参与了 `all_gatherv` 调用。

## 4. Qwen3 MoE vs Qwen3.5 MoE 架构差异

### 4.1 Qwen3 MoE (`qwen3_moe.py`)

- MoE 层: `Qwen3MoeSparseMoeBlock`
- 使用 `SharedFusedMoE`（继承自 `FusedMoE`）
- 路由: 内部 `self.gate` (ReplicatedLinear)
- `all_gatherv` 调用路径: `all2all.py` 中的 `NaiveAll2AllManager.dispatch()` / `dispatch_router_logits()`

### 4.2 Qwen3.5 MoE (`qwen3_5.py` + `qwen3_next.py`)

- MoE 层: `Qwen3NextSparseMoeBlock`（定义在 `qwen3_next.py:81`）
- 同样使用 `SharedFusedMoE`
- 同样有 `self.gate` 和 `self.shared_expert_gate`
- `all_gatherv` 调用路径: 同样通过 `all2all.py`

### 4.3 Transformers MoE 路径 (`transformers/moe.py`)

Qwen3.5 MoE 可能走 transformers 路径，该路径有额外的 `all_gatherv` 调用:

```python
# transformers/moe.py:62
def custom_routing_function(hidden_states, gating_output, topk, renormalize):
    if topk_ids.size(0) != hidden_states.size(0):
        dp_metadata = get_forward_context().dp_metadata
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        (topk_ids,) = dist_group.all_gatherv([topk_ids], 0, sizes)
```

## 5. all_gatherv 在 DP (Data Parallel) 场景中的调用路径

### 5.1 dispatch 路径 (`all2all.py:50-82`)

```python
def dispatch_router_logits(self, hidden_states, router_logits, ...):
    dp_metadata = get_forward_context().dp_metadata
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    gathered_tensors = dist_group.all_gatherv(tensors_to_gather, dim=0, sizes=sizes)
```

### 5.2 dispatch 路径 (`all2all.py:84-122`)

```python
def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    gathered_tensors = dist_group.all_gatherv(tensors_to_gather, dim=0, sizes=sizes)
```

### 5.3 combine 路径 (`all2all.py:124-137`)

```python
def combine(self, hidden_states, ...):
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
```

## 6. 根因分析

### 6.1 "非对称" 的两种含义

| 维度 | 含义 | all_gatherv 是否支持 |
|---|---|---|
| **数据量非对称** | 每个 rank 贡献不同数量的元素 | ✅ 支持，这是 "v" 的设计目的 |
| **参与非对称** | 某些 rank 不调用 all_gatherv | ❌ 不支持，NCCL 集体通信会 hang |

### 6.2 test_all_gatherv hang 的原因

`test_all_gatherv` 测试本身**不应该 hang**（所有 rank 都参与了调用）。如果它 hang 了，可能的原因：

1. **环境问题**: NCCL 通信初始化失败或 GPU 不可用
2. **测试框架问题**: `worker_fn_wrapper` / `distributed_run` 的进程管理有问题
3. **NCCL 版本兼容性**: 某些 NCCL 版本对 `ncclBroadcast` in group 有 bug

### 6.3 Qwen3.5 MoE 失败、Qwen3 MoE 成功的可能原因

**核心假设: DP 场景中的非对称参与**

在 DP (Data Parallel) + EP (Expert Parallel) 场景中:

1. **Qwen3 MoE**: 所有 DP rank 都会进入 MoE forward，都会调用 `all_gatherv` → ✅ 成功
2. **Qwen3.5 MoE**: 由于架构差异（hybrid attention + MoE 混合层、GDN Linear Attention），可能存在某些 DP rank 在特定条件下**跳过 MoE 层**或**提前返回**的情况 → 导致部分 rank 不参与 `all_gatherv` → 💀 hang

**具体可能的差异点**:

#### 假设 A: 模型架构差异导致的控制流分歧

Qwen3.5 使用 `GatedDeltaNetAttention`（Mamba-style attention，见 `qwen3_5.py:43`），这是一种**hybrid 架构**。某些层可能是 attention-only（不含 MoE），如果 DP rank 的 token 分配不均，可能导致某些 rank 在 MoE 层没有 token 要处理。

```python
# qwen3_5.py:156
if config.model_type == "qwen3_5_moe_text":
    self.mlp = Qwen3NextSparseMoeBlock(...)  # MoE 层
elif config.model_type == "qwen3_5_text":
    self.mlp = Qwen3NextMLP(...)  # 普通 MLP 层
```

#### 假设 B: sizes 中包含 0 的情况

如果某个 DP rank 上没有 token（`sizes` 中有 0），那么:

```python
# pynccl.py all_gatherv 中
for root, split_size in enumerate(sizes):
    dst_slice = output_tensor[split_offset : split_offset + split_size]  # split_size=0
    sendbuf = input_tensor if root == self.rank else dst_slice  # input_tensor 可能为空
    self.nccl.ncclBroadcast(sendbuf, dst_slice, dst_slice.numel(), ...)  # numel()=0
```

当 `split_size=0` 时，`ncclBroadcast` 的 count=0。NCCL 对 count=0 的行为可能是未定义的或在某些版本中有 bug。

**这可能是 Qwen3.5 MoE 的关键问题**: Qwen3.5 MoE 的 DP attention 分配机制可能允许某个 rank 分配 0 个 token 到 MoE 层。

#### 假设 C: 不同的 all2all 管理器

Qwen3 MoE 和 Qwen3.5 MoE 可能选择了不同的 all2all 管理器:
- `NaiveAll2AllManager`: 使用 `all_gatherv` + `reduce_scatterv`
- `NaiveAll2AllManagerTorch`: 使用 PyTorch native 实现
- `FlashInferNVLink*`: 使用 FlashInfer 库

如果 Qwen3.5 MoE 选择了 `NaiveAll2AllManager`（基于 `ncclBroadcast`），而 Qwen3 MoE 选择了其他实现，就可能解释差异。

## 7. 建议排查方向

1. **确认 hang 的位置**: 添加日志确认是否 hang 在 `ncclGroupEnd()` 
2. **检查 sizes 值**: 在 `all_gatherv` 入口打印 `sizes`，确认是否有 0 值
3. **检查是否所有 rank 都进入 MoE forward**: 在 `Qwen3NextSparseMoeBlock.forward()` 添加日志
4. **对比 all2all 管理器**: 确认 Qwen3 和 Qwen3.5 使用的是同一个管理器
5. **NCCL count=0 测试**: 写一个 `ncclBroadcast` count=0 的测试，验证是否会 hang
6. **检查 `reduce_scatterv`**: combine 阶段也使用 `reduce_scatterv`，同样可能有 hang 问题

## 8. 相关代码文件

| 文件 | 说明 |
|---|---|
| `vllm/distributed/device_communicators/pynccl.py:211-248` | `all_gatherv` 底层实现 |
| `vllm/distributed/device_communicators/cuda_communicator.py:346-395` | `all_gatherv` 高层包装 |
| `vllm/distributed/device_communicators/all2all.py:50-137` | DP MoE dispatch/combine |
| `vllm/distributed/parallel_state.py:544-552` | `all_gatherv` 入口 |
| `vllm/model_executor/models/qwen3_moe.py` | Qwen3 MoE 模型 |
| `vllm/model_executor/models/qwen3_next.py:81-191` | Qwen3.5 MoE 的 SparseMoeBlock |
| `vllm/model_executor/models/qwen3_5.py` | Qwen3.5 模型主文件 |
| `vllm/model_executor/models/transformers/moe.py:50-63` | Transformers MoE 路由中的 all_gatherv |
| `tests/distributed/test_pynccl.py:218-376` | all_gatherv 测试 |
| `vllm/forward_context.py:113-115` | `get_chunk_sizes_across_dp_rank` |
