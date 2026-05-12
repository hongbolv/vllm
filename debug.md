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

## NaN Root Cause — Conclusion

### TP=4/DP=1 reference case

TP=4/DP=1 runs successfully on the same 4x Intel ARC B60 hardware — no NaN,
no hang, correct output. This rules out XPU kernel numerical issues. NaN only
appears with DP=2 where DP padding is active.

### Root cause confirmed

DP padding causes a **mismatch between `num_actual_tokens` and attention mask
parameters** (`seq_lens`, `query_start_loc`):

- **dp_rank=0**: `seq_lens=[5,5,8,8]` (sum=26 real tokens), but
  `num_actual_tokens=30` (DP-padded). `query_start_loc=[0,5,10,18,26]` ends
  at 26 — rows 26-29 are padding tokens with **no sequence assignment**.
- **dp_rank=1**: `seq_lens=[7,5,11,7]` (sum=30) = `num_actual_tokens=30`.
  No gap, no NaN.

The attention backend processes all 30 rows for dp_rank=0, but rows 26-29
have no valid attention mask. Their attention scores are all-`-inf`, causing
softmax to produce 0/0 = NaN. Observed `nan_row_indices=[26,27,28,29]`
exactly matches the gap rows. NaN propagates to all subsequent layers via
residual-add.

### Fix applied

**File**: `vllm/v1/attention/backends/flash_attn.py`,
`FlashAttentionMetadataBuilder.build()`

The fix is applied at the flash attention backend level rather than in
`CommonAttentionMetadata`. This is because `CommonAttentionMetadata.num_actual_tokens`
is consumed by **both** the flash attention backend and the GDN (GatedDeltaNet)
attention kernel. The GDN kernel requires `num_actual_tokens == hidden_states.size(0)`
(i.e., `num_tokens_padded`), so we cannot change it globally.

Instead, the flash attention backend now clamps `num_actual_tokens` to
`query_start_loc_cpu[-1]` (the actual covered token count) during metadata
construction. When DP padding is active:
- `CommonAttentionMetadata.num_actual_tokens` = `num_tokens_padded` (30) — GDN happy
- `FlashAttentionMetadata.num_actual_tokens` = `query_start_loc[-1]` (26) — flash attention skips padding rows

```diff
     def build(self, ...):
         num_reqs = common_attn_metadata.num_reqs
         num_actual_tokens = common_attn_metadata.num_actual_tokens
+        # When DP padding is applied, num_actual_tokens includes padding rows
+        # but query_start_loc only covers real tokens. Clamp to the actual
+        # covered token count so the attention kernel skips unassigned padding
+        # rows (which would otherwise produce NaN via softmax on all-inf mask).
+        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
+        if query_start_loc_cpu is not None and len(query_start_loc_cpu) > 0:
+            covered_tokens = int(query_start_loc_cpu[-1])
+            if covered_tokens < num_actual_tokens:
+                num_actual_tokens = covered_tokens
```

The attention backend slices Q/output to `[:num_actual_tokens]` (26), so padding
rows (26-29) are never processed through the flash attention kernel. The
`seq_lens`/`query_start_loc` correctly describe all 26 real tokens, eliminating
the mismatch that produced NaN.

The buffer zero-initialization fix (`torch.empty` → `torch.zeros`) remains
as defense-in-depth to prevent NaN from uninitialized memory on XPU.
