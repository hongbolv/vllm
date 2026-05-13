# XPU EP Hang Diagnosis - Debug Summary

## Problem Statement

vLLM with Expert Parallelism (EP) on XPU hangs during inference when using
Data Parallelism (DP). The hang manifests as a silent deadlock — the process
stops producing output with no error message. After fixing the hang, the
inference output degrades to mostly `"!!!!"` tokens — caused by NaN
contamination propagated through the model layers.

**Config**: Qwen3.5-35B-A3B, TP=2, DP=2, EP=True (4x Intel ARC B60 GPUs).
See `examples/offline_inference/xpu_arc_b60_dp_ep.py` for the reference
configuration (rank mapping, prompt distribution, launch command).

**Observed symptom after hang fix**: First token(s) are correct but subsequent
decode tokens degrade to `"!!!!"`, e.g.:
```
DP rank 0: 'Hello, my name is' → ' Isha!!!!!!...'
DP rank 0: 'The capital of France is' → ' known as!!!!!!...'
```

---

## All Six Fixes — Status Summary

| Fix | File | Status | Description |
|-----|------|--------|-------------|
| Fix 1 | `dp_utils.py` | ✅ Applied | Force DP padding when EP is enabled |
| Fix 2 | `gpu_model_runner.py` | ✅ Applied | Align `pad_attn` with DP padding state |
| Fix 3 | `gpu_model_runner.py` | ✅ Applied | Disable async scheduling for EP+DP |
| Fix 4 | `xpu_communicator.py` | ❌ Reverted | Fix `all_gather` API usage — confirmed not the issue |
| Fix 5 | `xpu_communicator.py` | ❌ Reverted | Same-dtype tensor batching — confirmed not the issue |
| Fix 6 | `all2all.py` | ✅ Applied | Add XCCL barrier before MoE collectives |

---

## Applied Fixes

### Fix 1 — Force DP padding when EP is enabled

**File**: `vllm/v1/worker/dp_utils.py`

Without DP padding, each DP rank processes a different number of tokens. XCCL
MoE dispatch/combine collectives require equal-size tensors. Forcing DP padding
when EP is active ensures all ranks always have the same token count. Removing
this fix causes immediate hang — confirmed DP padding is required to prevent
EP all-to-all collective deadlock.

```diff
-    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    should_dp_pad = (synced_cudagraph_mode != 0 or should_ubatch
+                     or parallel_config.enable_expert_parallel)
```

### Fix 2 — Align `pad_attn` with DP padding state

**File**: `vllm/v1/worker/gpu_model_runner.py`

DP padding increases the token count, so attention metadata must be built with
padded sizes. The fix sets `pad_attn=True` whenever DP padding is applied, not
only when CUDAGraph FULL mode is active.

```diff
-            pad_attn = cudagraph_mode == CUDAGraphMode.FULL
+            dp_padding_applied = num_tokens_padded > num_tokens_unpadded
+            pad_attn = cudagraph_mode == CUDAGraphMode.FULL or dp_padding_applied
```

### Fix 3 — Disable async scheduling for EP+DP

**File**: `vllm/v1/worker/gpu_model_runner.py`

With async scheduling and EP+DP, DP ranks can advance at different speeds,
causing cross-iteration collective mismatch deadlocks. One rank enters
iteration N+1's `all_reduce` while the other is still in iteration N.

```diff
+        if (self.use_async_scheduling
+                and self.parallel_config.enable_expert_parallel
+                and self.parallel_config.data_parallel_size > 1):
+            self.use_async_scheduling = False
```

### Fix 4 — `all_gather` API fix in `xpu_communicator.py` (Reverted)

**File**: `vllm/distributed/device_communicators/xpu_communicator.py`

Investigated replacing `dist.all_gather([output_tensor], input_, ...)` with
`dist.all_gather_into_tensor(output_tensor, input_, ...)` to fix potential
API misuse. Confirmed via experiment that removing this fix causes no change
in output — the original `all_gather([output_tensor], ...)` with a single-element
list works correctly on XCCL. **Reverted.**

### Fix 5 — Same-dtype tensor batching in `xpu_communicator.py` (Reverted)

**File**: `vllm/distributed/device_communicators/xpu_communicator.py`

Investigated batching same-dtype tensors into a single `all_gatherv` call
(concatenating along dim=1) to reduce collective count. After validating that
tensor shapes were correctly reconstructed (no SHAPE MISMATCH or NOT
CONTIGUOUS errors), the output did not improve and the fix was determined to
be unnecessary. The `!!!!` output persisted, and NaN was traced to an
upstream source independent of this path — specifically, the attention mask
metadata mismatch documented in the [Prefill NaN root cause](#prefill-nan-root-cause--confirmed-fixed) section. **Reverted.**

### Fix 6 — Add XCCL barrier before MoE collectives

**File**: `vllm/distributed/device_communicators/all2all.py`

Adds an XCCL barrier before each `all_gatherv` and `reduce_scatterv` call in
`AgRsAll2AllManager` to force all EP ranks to rendezvous before submitting the
collective. Uses `dist.barrier(group=dist_group.device_group)`.

In the DP=2, TP=2, EP=True configuration, `dist_group.device_group` covers
all 4 ranks when called from `is_sequence_parallel=True` (EP group), or 2 DP
peers when called from `is_sequence_parallel=False`.

```diff
+        dist.barrier(group=dist_group.device_group)
         gathered_tensors = dist_group.all_gatherv(...)
```

---

## Additional Fixes

### Attention output buffer zero-initialization

**Files**: `vllm/model_executor/layers/attention/attention.py`,
`vllm/model_executor/models/qwen3_next.py`

Changed `torch.empty` → `torch.zeros` (shared `Attention` layer) and
`torch.empty_like` → `torch.zeros_like` (Qwen3.5 `Qwen3NextDecoderLayer`)
for attention output buffer allocation. With DP padding, `query.shape[0]` is
rounded up beyond `num_actual_tokens`, and the attention backend only writes
`output[:num_actual_tokens]`. On XPU (BMG), uninitialized memory in bf16/fp16
frequently contains NaN bit patterns. Using `torch.zeros` eliminates NaN
contamination from uninitialized padding rows (defense-in-depth).

### Flash attention k/v contiguous fix

**File**: `vllm/_xpu_ops.py`

Added `.contiguous()` calls for `k` and `v` tensors passed to
`flash_attn_varlen_func`. Non-contiguous tensors can cause incorrect results
or crashes in the XPU flash attention kernel.

```diff
-            k=k,
-            v=v,
+            k=k.contiguous(),
+            v=v.contiguous(),
```

### DP padding NaN root cause fix — attention metadata

**File**: `vllm/v1/worker/gpu_model_runner.py`, `_build_attention_metadata()`

This is the definitive fix for prefill NaN. See [Prefill NaN Root Cause](#prefill-nan-root-cause--confirmed-fixed) below.
Uses **local clones** of `seq_lens` and `query_start_loc` so the DP padding
extension only affects attention metadata; the shared buffers used for KV
cache `slot_mapping` computation remain unmodified (prevents KV cache
corruption that would cause NaN in subsequent decode steps).

---

## NaN Root Cause Analysis

### TP=4/DP=1 reference case

TP=4/DP=1 runs successfully on the same 4x Intel ARC B60 hardware — no NaN,
no hang, correct output. This rules out XPU kernel numerical issues as the
root cause. NaN only appears with DP=2 where DP padding is active.

### Prefill NaN root cause — confirmed, fixed

**`[NAN_CHECK_POST_ATTN]`** first triggered at `dp_rank=0, layer_idx=3`
(first full-attention layer) with `nan_row_indices=[26,27,28,29]`,
`actual_nan_rows=4`, `padding_nan_rows=0`.

**`[ATTN_MASK_CHECK]`** confirmed the root cause:

- **dp_rank=0**: `seq_lens=[5,5,8,8]` (sum=26 real tokens), but
  `num_actual_tokens=30` (DP-padded to match dp_rank=1).
  `query_start_loc=[0,5,10,18,26]` ends at 26 — rows 26–29 are DP padding
  tokens with **no sequence assignment**.
- **dp_rank=1**: `seq_lens=[7,5,11,7]` (sum=30) = `num_actual_tokens=30`.
  No gap, no NaN.

The attention backend processes all 30 rows for dp_rank=0, but rows 26–29
have no valid attention mask. Their attention scores are all-`-inf`, causing
softmax to produce 0/0 = NaN. NaN propagates to all subsequent layers via
residual-add. dp_rank=0 layer 0–2 (GDN linear-attention layers) show no
prefill NaN — the GDN kernel processes only real tokens in prefill.

**Fix applied** (`_build_attention_metadata()`): extend `seq_lens` and
`query_start_loc` to cover DP padding rows by assigning them to the last
request (direction 1). All attention backends see consistent metadata with
no uncovered rows.

When `num_tokens_padded > num_tokens` (DP padding active), the fix:
1. Sets `attn_query_start_loc[num_reqs_padded] = num_tokens_padded` so the
   last entry covers all rows including padding.
2. Adds `padding_len` to `attn_seq_lens[num_reqs_padded - 1]` so the last
   request's causal attention mask covers the padding rows.

Local clones (`attn_query_start_loc_gpu`, `attn_query_start_loc_cpu`,
`attn_seq_lens`) are used so the shared buffers (`self.query_start_loc`,
`self.seq_lens`) used for KV cache `slot_mapping` are not corrupted.

```diff
+        attn_query_start_loc_gpu = self.query_start_loc.gpu[: num_reqs_padded + 1]
+        attn_query_start_loc_cpu = self.query_start_loc.cpu[: num_reqs_padded + 1]
+        attn_seq_lens = self.seq_lens[:num_reqs_padded]
+        if num_tokens_padded > num_tokens:
+            padding_len = num_tokens_padded - num_tokens
+            attn_query_start_loc_gpu = attn_query_start_loc_gpu.clone()
+            attn_query_start_loc_cpu = attn_query_start_loc_cpu.clone()
+            attn_seq_lens = attn_seq_lens.clone()
+            attn_query_start_loc_gpu[num_reqs_padded] = num_tokens_padded
+            attn_query_start_loc_cpu[num_reqs_padded] = num_tokens_padded
+            attn_seq_lens[num_reqs_padded - 1] += padding_len
         cm_base = CommonAttentionMetadata(
-            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
-            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
-            seq_lens=self.seq_lens[:num_reqs_padded],
+            query_start_loc=attn_query_start_loc_gpu,
+            query_start_loc_cpu=attn_query_start_loc_cpu,
+            seq_lens=attn_seq_lens,
             ...
             num_actual_tokens=num_tokens_padded,
```

### Decode NaN — open issue (GDN recurrent state)

**`[NAN_CHECK_POST_ATTN]`** triggered at `dp_rank=1, layer_idx=1` (GDN
linear-attention layer) during the decode phase with:
- `shape=[4, 2048]`, `num_actual_tokens=4` (no DP padding on dp_rank=1)
- `nan_count=86` per affected row (partial NaN — not full-row)
- Different rows affected in consecutive decode steps (`[0]`, then `[2]`)

Key observations:
- dp_rank=1 has no DP padding in prefill (`seq_lens` sum = `num_actual_tokens`),
  so the prefill fix code path **never executes** on dp_rank=1.
- Partial NaN (86/2048 elements) rules out attention mask issues (which
  produce whole-row NaN via softmax on all-`-inf`).
- NaN is newly generated at layer 1 decode — `[NAN_CHECK_PRE_ATTN]` for
  layer 1 does **not** trigger (input is clean).
- The GDN `_forward_core()` CUDA path is not called on XPU — the XPU path
  goes through `forward_xpu()` → `_gdn_attention_core_xpu_impl()` in
  `_xpu_ops.py`, which calls the SYCL kernel directly via
  `torch.ops._xpu_C.gdn_attention`.

**Current hypothesis**: GDN recurrent state (`ssm_state`) written during
prefill may be incorrect in EP+DP scenarios, or `cu_seqlens`
(`non_spec_query_start_loc`) may be computed incorrectly causing the
recurrent computation to operate on wrong token ranges.

The `[GDN_STATE_CHECK]` diagnostic validates:
1. `last_recurrent_state` returned from `chunk_gated_delta_rule`
2. Written `ssm_state` slots
3. `initial_state` read back
4. `cu_seqlens` consistency: starts at 0, monotonically increasing, final
   entry matches expected token count

---

## Active Diagnostic Traces

All diagnostics skip warmup/profiling passes (when `attn_metadata` is `None`).
Each uses a class-level flag to print ERROR only on first occurrence.

| Trace tag | File | Purpose |
|-----------|------|---------|
| `[NAN_CHECK_PRE_ATTN]` | `qwen3_next.py` | Detect NaN in `hidden_states` **before** attention call in `Qwen3NextDecoderLayer` |
| `[NAN_CHECK_POST_ATTN]` | `qwen3_next.py` | Detect NaN in `hidden_states` **after** attention call; reports `actual_nan_rows` vs `padding_nan_rows` |
| `[GDN_STATE_CHECK]` | `_xpu_ops.py` | Validate GDN `ssm_state` after `gdn_attention` kernel, and `cu_seqlens` consistency in prefill and decode |
