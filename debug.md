# XPU EP Hang Diagnosis - Debug Summary

## Problem Statement

vLLM with Expert Parallelism (EP) on XPU hangs during inference when using
Data Parallelism (DP) with DP padding enabled. The hang manifests as a silent
deadlock - the process stops producing output with no error message.

**Config**: Qwen3.5-35B-A3B, TP=2, EP (MoE dispatch/combine over XCCL), DP padding enabled.

---

## Root-Cause Fixes

### Fix 1 - `num_actual_tokens` mismatch when DP padding is active

**File**: `vllm/v1/worker/gpu_model_runner.py`

```diff
-            pad_attn = cudagraph_mode == CUDAGraphMode.FULL
+            # Attention metadata needs padded sizes when CUDAGraph FULL
+            # mode is active, or when DP padding has increased the token
+            # count (e.g. for equal-size EP collectives on XPU).
+            dp_padding_applied = num_tokens_padded > num_tokens_unpadded
+            pad_attn = cudagraph_mode == CUDAGraphMode.FULL or dp_padding_applied
```

DP padding pads `hidden_states` (and thus `core_attn_out`) to the max token
count across DP ranks, but `num_actual_tokens` in attention metadata remained
at the real per-rank count. The XPU GDN kernel asserts
`core_attn_out.size(0) == num_actual_tokens` and fails. The fix ensures
`num_actual_tokens`, slot mappings, and attention metadata all reflect the
padded count. Padding slots get `-1` fill (no KV cache writes);
`logits_indices` already discards padding tokens from output.

### Fix 2 - Force DP padding when Expert Parallelism is enabled

**File**: `vllm/v1/worker/dp_utils.py`

```diff
-    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    # Also force DP padding when expert parallelism is enabled to ensure
+    # equal-size collectives (xccl workaround for unequal-size corruption).
+    should_dp_pad = (synced_cudagraph_mode != 0 or should_ubatch
+                     or parallel_config.enable_expert_parallel)
```

Without DP padding, each DP rank may have a different number of tokens. The
XCCL collectives in MoE dispatch/combine assume all-equal tensor sizes. Forcing
DP padding when EP is enabled ensures all DP ranks always process the same
token count, eliminating this class of XCCL corruption/hang.

---

## Iteration Tracing and Deadlock Risk Checker

### Iteration counter in `GPUModelRunner`

`self._iter_count` is incremented at the start of each `execute_model` call.
All trace prints now include `iter=N` so logs from multiple iterations are
easy to correlate across DP ranks, e.g.:

```
[TRACE dp=0 iter=1] execute_model: model forward complete, type(model_output)=Tensor
[TRACE dp=1 iter=1] execute_model: model forward complete, type(model_output)=Tensor
[TRACE dp=0 iter=2] execute_model: ENTER compute_logits
[TRACE dp=1 iter=1] execute_model: ENTER compute_logits    <- DP1 still on iter 1!
```

A gap like the above would confirm cross-iteration collective mismatch.

### Deadlock risk detection in `_run_ar`

`iter_count` is passed down through `_determine_batch_execution_and_padding` ->
`coordinate_batch_across_dp` -> `_synchronize_dp_ranks` -> `_run_ar` and
included in row 4 of the DP all-reduce tensor. After the all-reduce, `_run_ar`
checks if all DP ranks report the same iteration number:

```python
iter_counts = tensor[4]  # shape: [dp_size]
if int(iter_counts.max().item()) != int(iter_counts.min().item()):
    print(f"[WARN deadlock-risk] dp_rank={dp_rank} iter={iter_count} "
          f"iter_counts_across_dp={iter_counts.tolist()} -- ...")
```

If this warning fires, it means one DP rank has advanced to the next batch
before the other has finished the current one - exactly the condition that
causes a cross-iteration XCCL communicator deadlock.

---

## Chronological Diagnosis

### Step 1 - Initial hypothesis: variable-size XCCL collectives

The original fix (reverted) tried to pad all tensors to the same size before
`all_gather` / `reduce_scatter` in `xpu_communicator.py` and `all2all.py`.
This was reverted to restore original behavior and instead add tracing.

**Files changed**: `vllm/distributed/device_communicators/xpu_communicator.py`,
`vllm/distributed/device_communicators/all2all.py`

Trace prints added around:
- `reduce_scatterv` ENTER/EXIT
- `all_gatherv` ENTER/EXIT
- MoE `dispatch` ENTER/EXIT
- MoE `combine` ENTER/EXIT

### Step 2 - DP padding causes `num_actual_tokens` mismatch

**Log evidence**:
```
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=30, num_actual_tokens=30, match=True   # DP rank 0
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=30, num_actual_tokens=26, match=False  # DP rank 1
```

**Root cause**: DP padding pads `hidden_states` (and thus `core_attn_out`) to
the max token count across DP ranks (30), but `num_actual_tokens` in attention
metadata remained at the real count for rank 1 (26). The XPU GDN kernel asserts
`core_attn_out.size(0) == num_actual_tokens` and fails/hangs.

**Fix**: See Fix 1 above (commit `cd3b791` / `0130002`).

### Step 3 - GDN attention no longer hangs, but system still hangs

After the `pad_attn` fix, `num_actual_tokens` matched and GDN attention exited:

```
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=4, num_actual_tokens=4, match=True
[TRACE] _gdn_attention_core_xpu_impl: EXIT gdn_attention kernel
[TRACE] gdn_linear_attn forward_xpu: hidden_states.shape=torch.Size([4, 2048]), num_tokens=4
```

### Step 4 - Narrowing hang to decoder layer / MoE level

Added trace prints in `Qwen3NextDecoderLayer.forward` and
`Qwen3NextSparseMoeBlock.forward` (commit `f507331`). All attention layers
and all MoE experts blocks for layers 36-39 complete successfully. Hang occurs
**after** all decoder layers finish.

### Step 5 - Hang is after model forward

Added trace prints in `execute_model` (commit `3f17a87`). `execute_model`
completes and returns successfully through all stages (forward -> logits ->
return). The hang is downstream in `collective_rpc` or `sample_tokens`.

### Step 6 - `sample_tokens` completes too

Both `execute_model` and `sample_tokens` complete successfully on the first
iteration for both DP ranks (including async GPU->CPU copy path).

### Step 7 - DP0/DP1 desync across iterations

The second iteration reveals DP0 consistently running ahead of DP1:
- DP0 finishes `compute_logits` and enters `sample_tokens`
- DP1 is still inside its model forward
- DP0 may enter the third iteration's XCCL collective before DP1 finishes the
  second iteration's collective -> cross-iteration communicator deadlock

### Step 8 - Iter=3 hang: complete silence before model forward

**Log evidence** (provided after Fix 2 applied):

```
# iter=1 and iter=2 complete on all 4 processes (dp=0/tp=0, dp=0/tp=1,
# dp=1/tp=0, dp=1/tp=1 — each iter appears twice due to TP=2).
[TRACE dp=0 iter=2] sample_tokens: returning output (async)  # ← appears twice
[TRACE dp=1 iter=2] sample_tokens: returning output (async)  # ← appears twice
# Then: NOTHING. Zero output from any process for iter=3.
```

**Observations**:

1. **Duplicate traces per iteration are expected** — with TP=2, both `tp_rank=0`
   and `tp_rank=1` in each DP group share the same `dp_rank` and both print traces.
   Every iteration therefore prints twice per `dp=X` label.

2. **Complete silence after iter=2 is the anomaly** — all 4 processes produce
   ZERO output for iter=3. The earliest `execute_model` trace fires AFTER the
   model forward completes. The hang is before that point.

3. **No `[WARN deadlock-risk]` output** — the deadlock checker runs inside
   `_run_ar` only after `dist.all_reduce` returns. Since there is no such
   warning, `_run_ar` never completed: it is hanging inside `dist.all_reduce`.

**Root cause hypothesis**: `dist.all_reduce` in `_run_ar` hangs for iter=3.
This can happen if the XCCL communicator is in a corrupted or stalled state
after the iter=2 MoE dispatch/combine collectives, causing the next XCCL
operation (`_run_ar`) to block indefinitely.

**New traces added** to confirm this:
- `[TRACE dp=X iter=N] execute_model: ENTER (before _run_ar / DP all-reduce)`
  fires immediately when `execute_model` is entered, before any collective.
- `[TRACE dp=X iter=N] _run_ar: ENTER dist.all_reduce` / `EXIT dist.all_reduce`
  bracket the all-reduce call directly.

If the next run shows ENTER-execute_model but no ENTER-_run_ar: the hang is
between the two (unlikely, trivial code path). If ENTER-_run_ar appears but no
EXIT-_run_ar: confirmed the all-reduce itself is hanging.

---

### Confirmed Fixed
- **`num_actual_tokens` mismatch**: fixed by Fix 1
- **Unequal XCCL tensor sizes** when EP is enabled: fixed by Fix 2

### Remaining Hang - `dist.all_reduce` in `_run_ar` hangs at iter=3

After Fix 1 and Fix 2, iter=1 and iter=2 complete on all 4 processes. But
iter=3 never starts (no output from any process). The hypothesis is that
`dist.all_reduce` in `_run_ar` hangs for all processes at the start of iter=3,
caused by a corrupted or stalled XCCL communicator state left over from the
iter=2 MoE dispatch/combine collectives. New traces added in Step 8 will
confirm whether the hang is inside `dist.all_reduce` or elsewhere.

---

## Files Modified (Trace Infrastructure)

| File | Changes |
|------|---------|
| `vllm/_xpu_ops.py` | ENTER/EXIT around `gdn_attention` kernel; match check for `core_attn_out.size(0)` vs `num_actual_tokens` |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `hidden_states.shape` / `num_tokens` at `forward_xpu` entry |
| `vllm/model_executor/models/qwen3_next.py` | ENTER/EXIT around attn and MLP in `Qwen3NextDecoderLayer`; ENTER/EXIT around FusedMoE experts in `Qwen3NextSparseMoeBlock` |
| `vllm/v1/worker/gpu_model_runner.py` | `execute_model` and `sample_tokens` traces with `dp=` and `iter=`; early ENTER trace before `_run_ar`; **Fix 1**; **iteration counter** `_iter_count`; pass `iter_count` to `_determine_batch_execution_and_padding` |
| `vllm/v1/worker/dp_utils.py` | **Fix 2**: `should_dp_pad` includes EP; `_run_ar` extends tensor to 5 rows with `iter_count` in row 4; **deadlock risk checker** prints `[WARN deadlock-risk]` if iteration counts mismatch; ENTER/EXIT around `dist.all_reduce` |
| `vllm/distributed/device_communicators/xpu_communicator.py` | ENTER/EXIT around `reduce_scatterv` and `all_gatherv` |
| `vllm/distributed/device_communicators/all2all.py` | ENTER/EXIT around MoE `dispatch` and `combine` |

---

## Recommended Next Steps

1. ~~**Confirm Fix 2 resolves the hang**~~ **✓ CONFIRMED (iter=1 and iter=2)**:
   With `should_dp_pad` always True when EP is enabled, all DP ranks process
   the same number of tokens. iter=1 (prefill) and iter=2 (1st decode) complete
   on all 4 processes. The hang moved from iter=1 to iter=3.

2. **Confirm iter=3 hang location with new traces**: Re-run with the new
   ENTER-execute_model and ENTER/EXIT-_run_ar traces. Expected outcome:
   ```
   [TRACE dp=0 iter=3] execute_model: ENTER (before _run_ar / DP all-reduce)
   [TRACE dp=0 iter=3] _run_ar: ENTER dist.all_reduce
   # <-- hangs here, no EXIT line
   ```
   If this pattern appears, `dist.all_reduce` itself is the blocking call,
   pointing to a corrupted/stalled XCCL communicator after iter=2's MoE
   dispatch/combine.

3. **If hang is confirmed inside `dist.all_reduce`**: Investigate whether the
   iter=2 MoE all2all collectives leave the XCCL communicator in a bad state.
   A possible workaround is to insert an explicit `torch.xpu.synchronize()` (or
   XCCL barrier) after each MoE dispatch/combine and before `_run_ar`.

4. **Disable async output path** as a fallback: Set `use_async_output=False` to
   force synchronous GPU->CPU copies. If this resolves the hang, the async path
   is causing one rank to advance into iter=3's collective before the other
   finishes iter=2's MoE collective.

5. **Long-term**: Add a barrier in the executor so that all DP ranks must
   complete `sample_tokens` before any rank receives the next `execute_model`
   dispatch. This would definitively prevent cross-iteration collective
   mismatches.
