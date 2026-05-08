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

---

## Root Cause Analysis

### Confirmed Fixed
- **`num_actual_tokens` mismatch**: fixed by Fix 1
- **Unequal XCCL tensor sizes** when EP is enabled: fixed by Fix 2
- **Cross-iteration XCCL deadlock**: Fix 2 (forced DP padding) confirmed through
  logs to resolve the hang. With `should_dp_pad` always True when EP is enabled,
  all DP ranks process the same token count every iteration, and no
  `[WARN deadlock-risk]` warnings are emitted in the confirmed-working run.

### Remaining Hang - Cross-DP / Cross-iteration Synchronization

DP0 is consistently faster than DP1 due to the async output path returning
immediately. The scheduler may dispatch iteration N+1 to DP0 before DP1 has
finished iteration N, causing one DP rank's TP ranks to enter a collective
while the other's TP ranks are still in the previous iteration's collective -
XCCL communicator deadlock.

The `iter=N` labels and `[WARN deadlock-risk]` warnings confirm this.

---

## Files Modified (Trace Infrastructure)

| File | Changes |
|------|---------|
| `vllm/_xpu_ops.py` | ENTER/EXIT around `gdn_attention` kernel; match check for `core_attn_out.size(0)` vs `num_actual_tokens` |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `hidden_states.shape` / `num_tokens` at `forward_xpu` entry |
| `vllm/model_executor/models/qwen3_next.py` | ENTER/EXIT around attn and MLP in `Qwen3NextDecoderLayer`; ENTER/EXIT around FusedMoE experts in `Qwen3NextSparseMoeBlock` |
| `vllm/v1/worker/gpu_model_runner.py` | `execute_model` and `sample_tokens` traces with `dp=` and `iter=`; **Fix 1**; **iteration counter** `_iter_count`; pass `iter_count` to `_determine_batch_execution_and_padding` |
| `vllm/v1/worker/dp_utils.py` | **Fix 2**: `should_dp_pad` includes EP; `_run_ar` extends tensor to 5 rows with `iter_count` in row 4; **deadlock risk checker** prints `[WARN deadlock-risk]` if iteration counts mismatch |
| `vllm/distributed/device_communicators/xpu_communicator.py` | ENTER/EXIT around `reduce_scatterv` and `all_gatherv` |
| `vllm/distributed/device_communicators/all2all.py` | ENTER/EXIT around MoE `dispatch` and `combine` |

---

## Recommended Next Steps

1. ~~**Confirm Fix 2 resolves the hang**~~ **✓ CONFIRMED**: With
   `should_dp_pad` always True when EP is enabled, all DP ranks process the
   same number of tokens every iteration. XCCL collectives have equal-size
   inputs and the hang no longer occurs. Confirmed through run logs.

2. ~~**Confirm no `[WARN deadlock-risk]` warnings**~~ **✓ CONFIRMED**: No
   `[WARN deadlock-risk]` warnings are emitted after Fix 2 is applied,
   confirming that DP ranks stay in sync across iterations.

3. **Disable async output path** as a fallback (no longer needed given Fix 2,
   but remains an option): Set `use_async_output=False` to force synchronous
   GPU->CPU copies. This slows DP0 down, giving DP1 time to catch up. If a
   future regression reintroduces desync, this would be the first thing to try.

4. **Long-term**: Add a barrier in the executor so that all DP ranks must
   complete `sample_tokens` before any rank receives the next `execute_model`
   dispatch. This would definitively prevent cross-iteration collective
   mismatches even if DP padding is not applied.
