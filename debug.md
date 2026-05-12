# XPU EP Hang Diagnosis - Debug Summary

## Problem Statement

vLLM with Expert Parallelism (EP) on XPU hangs during inference when using
Data Parallelism (DP). The hang manifests as a silent deadlock — the process
stops producing output with no error message.

**Config**: Qwen3.5-35B-A3B, TP=2, DP=2, EP=True (4x Intel ARC B60 GPUs).
See `examples/offline_inference/xpu_arc_b60_dp_ep.py` for the reference
configuration (rank mapping, prompt distribution, launch command).

---

## Applied Fixes

### Fix 1 — Force DP padding when EP is enabled

**File**: `vllm/v1/worker/dp_utils.py`

Without DP padding, each DP rank processes a different number of tokens. XCCL
MoE dispatch/combine collectives require equal-size tensors. Forcing DP padding
when EP is active ensures all ranks always have the same token count.

```diff
-    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    should_dp_pad = (synced_cudagraph_mode != 0 or should_ubatch
+                     or parallel_config.enable_expert_parallel)
```

### Fix 2 — Align `pad_attn` with DP padding state

**File**: `vllm/v1/worker/gpu_model_runner.py`

DP padding pads `hidden_states` to the max token count across DP ranks (e.g.,
30), but `num_actual_tokens` in attention metadata remained at the real count
(e.g., 26). The fix sets `pad_attn=True` whenever DP padding is applied.

```diff
-            pad_attn = cudagraph_mode == CUDAGraphMode.FULL
+            dp_padding_applied = num_tokens_padded > num_tokens_unpadded
+            pad_attn = cudagraph_mode == CUDAGraphMode.FULL or dp_padding_applied
```

### Fix 3 — Disable async scheduling for EP+DP

**File**: `vllm/v1/worker/gpu_model_runner.py`

With async scheduling and EP+DP, DP ranks can advance at different speeds,
causing cross-iteration collective mismatch deadlocks.

```diff
+        if (self.use_async_scheduling
+                and self.parallel_config.enable_expert_parallel
+                and self.parallel_config.data_parallel_size > 1):
+            self.use_async_scheduling = False
```

### Fix 6 — Add XCCL barrier before MoE collectives

**File**: `vllm/distributed/device_communicators/all2all.py`

Adds an XCCL barrier before each `all_gatherv` and `reduce_scatterv` call in
`AgRsAll2AllManager` to force all EP ranks to rendezvous before submitting the
collective. Uses `dist.barrier(group=dist_group.device_group)` to issue a
device-level (not CPU-level) barrier.

---

## Attention Output Buffer Fix

**Files**: `vllm/model_executor/layers/attention/attention.py`,
`vllm/model_executor/models/qwen3_next.py`

Changed `torch.empty` → `torch.zeros` (shared attention layer) and
`torch.empty_like` → `torch.zeros_like` (Qwen3.5 model layer) for attention
output buffer allocation. With DP padding, `query.shape[0]` is rounded up
beyond `num_actual_tokens`, and the attention backend only writes
`output[:num_actual_tokens]`. On XPU (BMG), uninitialized memory in bf16/fp16
frequently contains NaN bit patterns. Using `torch.zeros` eliminates NaN
contamination from uninitialized padding rows.

---

## NaN Root Cause Analysis

### Diagnostic setup

Two NaN detection traces are instrumented in `Qwen3NextDecoderLayer.forward()`:
- `[NAN_CHECK_PRE_ATTN]` — before attention call
- `[NAN_CHECK_POST_ATTN]` — after attention call, with actual/padding row
  distinction using `num_actual_tokens` from `attn_metadata`

Both use one-shot class-level flags (first occurrence only, `print()` with
`flush=True`). Both skip warmup/profiling passes (when `attn_metadata` is
`None`).

### Diagnostic log evidence

```
dp_rank=0 POST_ATTN layer_idx=3 shape=[30, 2048] num_actual_tokens=30
  actual_nan_rows=4 actual_nan_elems=8192 padding_nan_rows=0
  nan_row_indices=[26, 27, 28, 29] total_nan_rows=4

dp_rank=0 PRE_ATTN layer_idx=4 shape=[30, 2048]
  nan_row_indices=[26, 27, 28, 29] total_nan_rows=4

dp_rank=1 POST_ATTN layer_idx=1 shape=[4, 2048] num_actual_tokens=4
  actual_nan_rows=1 actual_nan_elems=86 padding_nan_rows=0
  nan_row_indices=[0] total_nan_rows=1

dp_rank=1 PRE_ATTN layer_idx=2 shape=[4, 2048]
  nan_row_indices=[0, 2] total_nan_rows=2
```

### Key findings

1. **dp_rank=0** logs are from the **prefill** stage (shape=[30, 2048], 30
   prompt tokens). dp_rank=1 logs are from the **decode** stage (shape=[4,
   2048], 4 sequences × 1 decode token each). The one-shot flags triggered at
   different inference stages on different ranks.

2. **Prefill precedes decode in time** — vLLM schedules all prefills before
   entering the decode loop, so dp_rank=0's prefill NaN at layer 3 occurred
   first chronologically. However, dp_rank=1's decode NaN at layer 1 was
   independently produced (PRE_ATTN did not trigger at layer 0 or 1, meaning
   the input to layer 1 was clean).

3. **NaN is in actual tokens only** (`padding_nan_rows=0` in all logs). The
   buffer zero-initialization fix eliminated padding NaN, but NaN persists in
   actual token computation.

4. **Two independent NaN sources**:
   - **dp_rank=0, layer 3 (full_attention)**: 4 rows × 2048 elements = 8192
     NaN (entire rows). This is full-row NaN produced by the full_attention
     (softmax-based) computation on XPU.
   - **dp_rank=1, layer 1 (linear_attention/GDN)**: 1 row × 86/2048 elements.
     This is a **partial NaN pattern** characteristic of the GDN delta-net
     kernel (`fused_recurrent_gated_delta_rule`) — specific channels overflow
     during the recurrence computation.

5. **NaN propagation**: Once NaN appears in a layer's output, it propagates
   to subsequent layers via residual-add (NaN + any value = NaN). This is why
   PRE_ATTN reports at layer N+1 match POST_ATTN NaN rows from layer N.

6. **Each log line appears twice** because TP=2: two ranks (tp_rank=0 and
   tp_rank=1) share the same dp_rank and process the same tokens, so both
   report the same NaN event.

### TP=4/DP=1 reference case

The same model (Qwen3.5-35B-A3B) runs successfully with **TP=4, DP=1** on the
same 4x Intel ARC B60 hardware — no NaN, no hang, correct output. This is a
critical observation:

- **TP=4/DP=1**: No DP padding → `num_actual_tokens == query.shape[0]` →
  attention mask parameters (`seq_lens`, `query_start_loc`) correctly reflect
  actual token boundaries → no NaN
- **TP=2/DP=2**: DP padding active → token count padded to max across DP ranks
  → padding changes batch structure → potential attention mask parameter
  corruption → NaN in specific token positions

Since the XPU attention kernels produce correct results without DP padding
(TP=4/DP=1), **the kernels themselves are not the root cause**. The NaN is
caused by how DP padding interacts with attention mask construction.

### ATTN_MASK_CHECK log analysis — definitive confirmation

The `[ATTN_MASK_CHECK]` diagnostic output confirms the DP padding → NaN
connection:

```
[ATTN_MASK_CHECK] dp_rank=0 layer_idx=3 num_actual_tokens=30 seq_lens=[5, 5, 8, 8] query_start_loc=[0, 5, 10, 18, 26] hidden_shape=[30, 2048]
[ATTN_MASK_CHECK] dp_rank=1 layer_idx=3 num_actual_tokens=30 seq_lens=[7, 5, 11, 7] query_start_loc=[0, 7, 12, 23, 30] hidden_shape=[30, 2048]
[NAN_CHECK_POST_ATTN] ERROR dp_rank=0 layer_idx=3 ... nan_row_indices=[26, 27, 28, 29]... total_nan_rows=4
```

**dp_rank=1** (no padding needed):
- `seq_lens=[7, 5, 11, 7]` → sum = **30** = `num_actual_tokens`
- `query_start_loc=[0, 7, 12, 23, 30]` → last boundary = 30 = total tokens
- All 30 rows are covered by valid sequences → **no NaN**

**dp_rank=0** (padding applied):
- `seq_lens=[5, 5, 8, 8]` → sum = **26** real tokens
- `query_start_loc=[0, 5, 10, 18, 26]` → last boundary = 26
- `num_actual_tokens=30`, `hidden_shape=[30, 2048]` → DP padding added **4 extra rows** (26→30)
- Rows 26-29 are **not covered by any sequence** in `seq_lens`/`query_start_loc`
- These 4 rows have no valid attention mask entry → attention backend processes
  them with an **all-masked (all -inf)** attention score → softmax(-inf) = 0/0 = **NaN**
- `nan_row_indices=[26, 27, 28, 29]` — exactly the 4 gap rows

**Root cause confirmed**: DP padding increases `num_actual_tokens` from 26 to 30
for dp_rank=0, but `seq_lens` and `query_start_loc` still only describe the
real 26 tokens. The attention backend treats all 30 rows as actual tokens and
computes attention for rows 26-29, which have no valid sequence assignment. With
no valid attention targets, these rows receive all-`-inf` attention scores,
and softmax produces 0/0 = NaN. The NaN then propagates to all subsequent
layers via residual-add.

### Conclusion

The full_attention NaN is caused by a **mismatch between DP-padded
`num_actual_tokens` and the attention mask parameters** (`seq_lens`,
`query_start_loc`). DP padding extends the token count but does not extend
the mask parameters to cover the padding rows. This is a bug in the DP
padding → attention metadata construction path.

**Fix direction**: When DP padding is applied, the attention mask parameters
must either:
1. **Extend `seq_lens`/`query_start_loc`** to cover padding rows (e.g., add a
   dummy sequence of length `num_padded - num_real` at the end), or
2. **Keep `num_actual_tokens` at the real count** (26) so the attention backend
   only processes real tokens and skips padding rows entirely.

The buffer zero-initialization fix (`torch.empty` → `torch.zeros`) remains
necessary as a defense-in-depth measure to prevent NaN from uninitialized
memory, but the core fix must address the `num_actual_tokens` vs `seq_lens`
mismatch in the DP padding path.
