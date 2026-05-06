# Debug 分析: test_all_gatherv hang 问题 (XPU/XCCL)

## 1. 问题概述

- **test_all_gatherv 脚本在 XPU 上 hang** — 使用 XCCL 后端的 variable-size `dist.all_gather` 卡住
- **Qwen3.5 MoE (Qwen3.5-35B-A3B) 推理 hang** — DP=2, TP=2, EP=true 场景下卡在 "Processed prompts: 0%"
- **Qwen3 MoE 成功** — 相同配置下可以正常推理

## 2. test_all_gatherv 脚本分析

### 2.1 测试脚本 (独立复现脚本)

```bash
torchrun --nproc-per-node=4 /models/test_all_gatherv.py
```

```python
import torch, torch.distributed as dist

dist.init_process_group(backend="xccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Variable sizes: rank 0 has 3 tokens, rank 1 has 5, rank 2 has 2, rank 3 has 4
sizes = [3, 5, 2, 4]
my_size = sizes[rank]
hidden = 128

t = torch.randn(my_size, hidden, device=f"xpu:{rank}")

# Variable-size all_gather (what EP dispatch does)
all_gather_list = []
for size in sizes:
    all_gather_list.append(
        torch.empty((size, hidden), dtype=t.dtype, device=t.device)
    )
dist.all_gather(all_gather_list, t)
gathered = torch.cat(all_gather_list, dim=0)
print(f"Rank {rank}: variable all_gather succeeded, shape={gathered.shape}")

# Variable-size reduce_scatter (what EP combine does)
input_splits = list(gathered.split(sizes, dim=0))
output = torch.empty((my_size, hidden), dtype=t.dtype, device=t.device)
dist.reduce_scatter(output, input_splits)
print(f"Rank {rank}: variable reduce_scatter succeeded, shape={output.shape}")

dist.destroy_process_group()
```

### 2.2 测试关键特征

| 特征 | 值 |
|---|---|
| 后端 | **XCCL** (Intel XPU 分布式通信后端) |
| 设备 | **XPU** (Intel ARC B60 GPU) |
| world_size | 4 |
| sizes | `[3, 5, 2, 4]` — 每个 rank 贡献**不同数量**的元素 |
| 操作 | `dist.all_gather(list_of_different_sized_tensors, input_tensor)` |

### 2.3 Hang 的位置

脚本在 `dist.all_gather(all_gather_list, t)` 处 hang。这表明 **XCCL 后端不支持 variable-size `all_gather`** — 即 `all_gather_list` 中的 tensor 大小不同时会卡死。

## 3. vLLM 中对应的代码路径

### 3.1 XPU communicator 的 `all_gatherv` 实现 (`xpu_communicator.py:107-159`)

```python
def all_gatherv(self, input_, dim=0, sizes=None):
    # 当所有 sizes 相同时，退化为普通 all_gather (sizes 设为 None)
    if sizes is not None and all(s == sizes[0] for s in sizes):
        sizes = None

    def _all_gather_single(input_, sizes=None):
        if sizes is not None:
            # ⚠️ 关键路径: variable-size all_gather
            all_gather_list = []
            for size in sizes:
                all_gather_list.append(
                    torch.empty((size,) + input_.shape[1:], ...)
                )
            dist.all_gather(all_gather_list, input_, group=self.device_group)
            # ^^^ 这里会 HANG (XCCL 不支持 variable-size all_gather)
            output_tensor = torch.cat(all_gather_list, dim=0)
        else:
            # ✅ equal-size 路径: 使用 all_gather_into_tensor，正常工作
            dist.all_gather([output_tensor], input_, group=self.device_group)
        return output_tensor
```

**`test_all_gatherv` 脚本使用的就是与 `xpu_communicator.py:137-148` 完全相同的模式。**

### 3.2 XPU communicator 的 `reduce_scatterv` 实现 (`xpu_communicator.py:73-105`)

```python
def reduce_scatterv(self, input_, dim=-1, sizes=None):
    if sizes is not None and sizes.count(sizes[0]) != len(sizes):
        # ⚠️ variable-size 路径
        input_splits = list(input_tensor.split(sizes, dim=0))
        dist.reduce_scatter(output, input_splits, group=self.device_group)
        # ^^^ 同样可能 HANG
    else:
        # ✅ equal-size 路径
        dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
```

### 3.3 EP dispatch/combine 调用路径 (`all2all.py`)

XPU 上使用 `AgRsAll2AllManager` (All-Gather / Reduce-Scatter 方式):

```python
# dispatch (EP 分发): 将各 DP rank 的 hidden_states 聚合到所有 rank
def dispatch_router_logits(self, hidden_states, router_logits, ...):
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    gathered_tensors = dist_group.all_gatherv(tensors, dim=0, sizes=sizes)
    # → 调用 xpu_communicator.all_gatherv → dist.all_gather (variable-size)

# combine (EP 合并): 将结果分散回各 DP rank
def combine(self, hidden_states, ...):
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
    # → 调用 xpu_communicator.reduce_scatterv → dist.reduce_scatter (variable-size)
```

## 4. Qwen3 MoE vs Qwen3.5 MoE: 为什么一个成功一个失败

### 4.1 关键差异: DP rank 之间的 token 数量是否相等

当所有 DP rank 的 token 数量**相等**时:
- `sizes = [N, N, N, N]` (所有相同)
- `all_gatherv` 检测到 `all(s == sizes[0] for s in sizes)` → `sizes = None`
- 走 **equal-size 路径** → `dist.all_gather([output_tensor], input_)` → ✅ 正常工作

当 DP rank 的 token 数量**不等**时:
- `sizes = [3, 5, 2, 4]` (不同)
- 走 **variable-size 路径** → `dist.all_gather(list_of_different_sized_tensors, input_)` → 💀 HANG

### 4.2 Qwen3 MoE 成功的原因

Qwen3 MoE 在 DP+EP 场景中，各 DP rank 处理的 token 数量**恰好相等**（或通过 padding 保证相等）:
- `sizes = [N, N]` → 退化为 equal-size `all_gather` → ✅ 成功

### 4.3 Qwen3.5 MoE 失败的原因

Qwen3.5 MoE (Qwen3.5-35B-A3B) 在 DP+EP 场景中，各 DP rank 处理的 token 数量**不相等**:
- `sizes = [M, N]` (M ≠ N) → 走 variable-size `all_gather` → 💀 HANG

**可能导致 token 数量不等的原因:**

1. **Hybrid 架构**: Qwen3.5 使用 `GatedDeltaNetAttention` (Mamba-style linear attention)，与标准 Transformer 不同的 token 处理方式可能导致不同 DP rank 的序列长度不同
2. **Sequence Parallel**: Qwen3.5 MoE 支持 `use_sequence_parallel_moe`，启用时会对 token 进行 chunk，可能产生不均匀的 chunk sizes
3. **不同的 prompt 分配**: DP rank 0 和 rank 1 分别处理不同的 prompt，如果 prompt 长度不同，token 数量就不同

## 5. 根因总结

```
根本原因: XCCL 后端不支持 variable-size dist.all_gather / dist.reduce_scatter

调用链:
  Qwen3.5 MoE inference (DP=2, TP=2, EP=true)
  → EP dispatch (AgRsAll2AllManager.dispatch_router_logits)
  → dist_group.all_gatherv(tensors, sizes=[M, N])  # M ≠ N
  → XpuCommunicator.all_gatherv(sizes=[M, N])
  → sizes are NOT equal → 走 variable-size 路径
  → dist.all_gather(list_of_different_sized_tensors, input_)
  → XCCL backend 不支持此操作 → HANG
```

**而 Qwen3 MoE 恰好 sizes 相等，退化为 equal-size 路径，因此不会触发此 bug。**

## 6. 验证方式

### 6.1 确认 XCCL variable-size all_gather hang

```python
# test_equal_size.py — 应该成功
import torch, torch.distributed as dist
dist.init_process_group(backend="xccl")
rank = dist.get_rank()
t = torch.randn(10, 128, device=f"xpu:{rank}")
out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
dist.all_gather(out, t)  # equal-size → 应该成功
print(f"Rank {rank}: equal-size all_gather succeeded")
dist.destroy_process_group()
```

```python
# test_variable_size.py — 应该 hang
import torch, torch.distributed as dist
dist.init_process_group(backend="xccl")
rank = dist.get_rank()
sizes = [3, 5, 2, 4]
t = torch.randn(sizes[rank], 128, device=f"xpu:{rank}")
out = [torch.empty(s, 128, device=f"xpu:{rank}") for s in sizes]
dist.all_gather(out, t)  # variable-size → 预期 HANG
print(f"Rank {rank}: variable-size all_gather succeeded")
dist.destroy_process_group()
```

### 6.2 确认 Qwen3.5 MoE 的 sizes 不等

在 `xpu_communicator.py:all_gatherv` 入口添加日志:
```python
def all_gatherv(self, input_, dim=0, sizes=None):
    import logging
    logging.warning(f"[XPU all_gatherv] rank={self.rank_in_group} sizes={sizes}")
    ...
```

## 7. 可能的修复方向

### 方案 A: 在 XPU communicator 中用多次 broadcast 替代 variable-size all_gather

```python
# xpu_communicator.py 中的 all_gatherv, sizes != None 时:
if sizes is not None:
    # 不使用 dist.all_gather (XCCL 不支持 variable-size)
    # 而是用 N 次 broadcast 模拟
    all_gather_list = []
    for root, size in enumerate(sizes):
        buf = torch.empty((size,) + input_.shape[1:], dtype=input_.dtype, device=input_.device)
        if root == self.rank_in_group:
            buf.copy_(input_)
        dist.broadcast(buf, src=root, group=self.device_group)
        all_gather_list.append(buf)
    output_tensor = torch.cat(all_gather_list, dim=0)
```

### 方案 B: 将 variable-size 输入 pad 到相同大小

```python
# 将所有 rank 的输入 pad 到 max(sizes)，然后用 equal-size all_gather
max_size = max(sizes)
padded_input = torch.zeros(max_size, *input_.shape[1:], ...)
padded_input[:input_.shape[0]] = input_
# equal-size all_gather
dist.all_gather_into_tensor(gathered, padded_input, group=self.device_group)
# 然后去掉 padding
```

### 方案 C: 修复 XCCL 后端使其支持 variable-size all_gather

这需要在 Intel oneCCL / PyTorch XPU 层面修复，不在 vLLM 范围内。

## 8. 相关代码文件

| 文件 | 说明 |
|---|---|
| `vllm/distributed/device_communicators/xpu_communicator.py:107-159` | XPU `all_gatherv` 实现 (hang 的位置) |
| `vllm/distributed/device_communicators/xpu_communicator.py:73-105` | XPU `reduce_scatterv` 实现 (同样可能 hang) |
| `vllm/distributed/device_communicators/all2all.py:41-137` | `AgRsAll2AllManager` EP dispatch/combine |
| `vllm/distributed/parallel_state.py:544-552` | `all_gatherv` 入口 |
| `vllm/forward_context.py:113-115` | `get_chunk_sizes_across_dp_rank` |
| `vllm/model_executor/models/qwen3_moe.py` | Qwen3 MoE 模型 |
| `vllm/model_executor/models/qwen3_next.py:81-191` | Qwen3.5 MoE 的 SparseMoeBlock |
| `vllm/model_executor/models/qwen3_5.py` | Qwen3.5 模型主文件 |
