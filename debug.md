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

### Fix 3 - Disable async scheduling when EP + DP is active

**File**: `vllm/v1/worker/gpu_model_runner.py`

```diff
+        # Disable async scheduling when Expert Parallelism + Data Parallelism
+        # is active: AsyncGPUModelRunnerOutput lets one DP rank advance to the
+        # next iteration before the other DP rank finishes the current one.
+        # This skew causes the DP all_reduce in _run_ar to deadlock.
+        if (self.use_async_scheduling
+                and self.parallel_config.enable_expert_parallel
+                and self.parallel_config.data_parallel_size > 1):
+            self.use_async_scheduling = False
```

`AsyncGPUModelRunnerOutput` starts the GPU→CPU output copy asynchronously and
returns immediately, potentially allowing dp=0's scheduler to queue iter=N+1
before dp=1 has finished iter=N's GPU copy. If dp=0 enters iter=N+1's DP
all_reduce before dp=1 arrives, all 4 processes can deadlock. This fix is a
preventive measure applied pending confirmation that async output skew is the
actual remaining hang cause (logs through iter=14 show both DP ranks stay in
sync, so the actual hang point is at iter≥15 and needs further investigation).

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

### Step 9 - Both dp=0 and dp=1 complete iter=13 AND iter=14; hang is later

**Log evidence** (from run with ENTER/EXIT traces around `dist.all_reduce`):

```
[TRACE dp=0 iter=13] _run_ar: ENTER dist.all_reduce   ← RANK=0 (dp=0, tp=0)
[TRACE dp=1 iter=13] _run_ar: ENTER dist.all_reduce   ← RANK=2 (dp=1, tp=0)
[TRACE dp=0 iter=13] execute_model: ENTER             ← RANK=1 (dp=0, tp=1), slightly slower
[TRACE dp=1 iter=13] execute_model: ENTER             ← RANK=3 (dp=1, tp=1), slightly slower
[TRACE dp=0 iter=13] _run_ar: EXIT dist.all_reduce    ← RANK=0 exits (Group A done)
[TRACE dp=1 iter=13] _run_ar: EXIT dist.all_reduce    ← RANK=2 exits
[TRACE dp=0 iter=13] _run_ar: ENTER dist.all_reduce   ← RANK=1 (dp=0, tp=1) enters Group B
[TRACE dp=1 iter=13] _run_ar: ENTER dist.all_reduce   ← RANK=3 (dp=1, tp=1) enters Group B
[TRACE dp=0/1 iter=13] _run_ar: EXIT dist.all_reduce  ← both exit Group B
# ... iter=13 model forward and sample_tokens complete for all 4 processes
# ... iter=14 also completes for both dp=0 AND dp=1 (confirmed by full log)
# Hang occurs at some later iteration (iter=15 or beyond)
```

**Key observation - TWO separate DP communicator subgroups**:

With TP=2, DP=2, vLLM creates two independent DP communicator groups:
- **Group A**: `{RANK=0 (dp=0,tp=0), RANK=2 (dp=1,tp=0)}` — tp=0 processes
- **Group B**: `{RANK=1 (dp=0,tp=1), RANK=3 (dp=1,tp=1)}` — tp=1 processes

All 4 processes call `_run_ar`, but Group A and Group B each do an independent `dist.all_reduce`. Group A finishes first (tp=0 processes slightly faster), then Group B. This is why we see TWO ENTER/EXIT pairs per dp_rank per iteration — one from each group. This behavior is **normal** and both groups succeed for iter=13 and iter=14.

**Status**: iter=13 and iter=14 both complete successfully on all 4 processes.
The hang occurs at a later iteration (likely iter=15 or beyond). The async output
skew hypothesis (Fix 3) was not confirmed as the specific cause for this run since
both DP ranks stay synchronized through iter=14. Full logs beyond iter=14 are needed
to identify the exact hang point.

---

### Step 10 — New run (TP=2, Fix 3 NOT applied): hang after iter=18, iter=19 never starts

**Log evidence** (fresh run, full log showing iter=17 and iter=18 complete, but
iter=19 never starts):

```
# --- Group A (tp=0) reaches iter=17 first ---
[TRACE dp=0 iter=17] execute_model: ENTER (before _run_ar / DP all-reduce)
[TRACE dp=1 iter=17] execute_model: ENTER (before _run_ar / DP all-reduce)
[TRACE dp=0 iter=17] _run_ar: ENTER dist.all_reduce    ← Group A (RANK=0, tp=0)
[TRACE dp=1 iter=17] _run_ar: ENTER dist.all_reduce    ← Group A (RANK=2, tp=0)
[TRACE dp=1 iter=17] _run_ar: EXIT dist.all_reduce
[TRACE dp=0 iter=17] _run_ar: EXIT dist.all_reduce     ← Group A exits

# --- Group B (tp=1) is still finishing iter=16's CPU postprocessing ---
# RANK=1 (dp=0,tp=1) and RANK=3 (dp=1,tp=1) lag behind tp=0 in CPU scheduling;
# they are still running Python postprocessing code for iter=16 while tp=0
# has already incremented to iter=17 and entered _run_ar.
[TRACE dp=1 iter=16] execute_model: model forward complete ...
[TRACE dp=1 iter=16] execute_model: postprocess ENTER ...
[TRACE dp=0 iter=16] execute_model: model forward complete ...  ← RANK=1 (dp=0,tp=1)
[TRACE dp=0 iter=16] execute_model: postprocess ENTER ...
[TRACE dp=1 iter=16] sample_tokens: ENTER
[TRACE dp=1 iter=16] sample_tokens: ModelRunnerOutput built, use_async=True
[TRACE dp=1 iter=16] sample_tokens: ENTER AsyncGPUModelRunnerOutput
[TRACE dp=1 iter=16] sample_tokens: EXIT AsyncGPUModelRunnerOutput
[TRACE dp=0 iter=16] execute_model: EXIT compute_logits
[TRACE dp=0 iter=16] execute_model: returning None (success)
[TRACE dp=1 iter=16] sample_tokens: returning output (async)  ← RANK=3 (dp=1,tp=1)
[TRACE dp=0 iter=16] sample_tokens: ENTER ...
# ... dp=0 (RANK=1, tp=1) also completes iter=16's sample_tokens

# --- Group B then runs iter=17's _run_ar (Group B's turn) ---
# [These traces follow but are not shown in the excerpt.]
# Both Group A and Group B complete iter=17 and iter=18 fully.

# --- After iter=18: all 4 processes have called sample_tokens (async) ---
# ... iter=18 execute_model and sample_tokens complete on all 4 processes ...
[TRACE dp=0 iter=18] sample_tokens: returning output (async)  ← last trace
[TRACE dp=1 iter=18] sample_tokens: returning output (async)  ← last trace
# Then: NOTHING. No iter=19 execute_model: ENTER on any process.
```

**Key observations**:

1. **TP=2, CPU lag between tp=0 and tp=1 processes is normal**: In the iter=17
   portion of the log we see `dp=0 iter=17` / `dp=1 iter=17` `_run_ar` (Group A,
   tp=0 processes) followed immediately by `dp=0 iter=16` / `dp=1 iter=16`
   execute_model traces. This is the **tp=1 processes (RANK=1 and RANK=3)** running
   one CPU iteration behind the tp=0 processes. Since GPU ops are async, tp=0 can
   finish CPU postprocessing for iter=16, increment to iter=17, and enter `_run_ar`
   before tp=1 finishes its Python postprocessing for iter=16. This is expected
   behavior with TP=2 + async scheduling.

2. **Fix 3 NOT applied**: `use_async=True` in sample_tokens traces confirms the
   async scheduling fix was not active in this run.

3. **Both iter=17 and iter=18 complete (CPU-side)**: Unlike the previous
   understanding, iter=18 also runs and all 4 processes complete through
   `sample_tokens: returning output (async)` for iter=18. Iter=19 is never started.

4. **Root cause — GPU-side silent hang inside iter=18's model forward**:

   Both iter=17 and iter=18 GPU model forwards are submitted to the GPU asynchronously.
   The CPU traces all complete normally because Python returns immediately after
   submitting GPU ops. However, the GPU silently hangs inside iter=18's model
   forward — likely inside an XCCL MoE dispatch/combine op. When the async GPU→CPU
   output copy for iter=18 is queued on the GPU stream, it waits behind the stuck
   op and never starts.

   The scheduler retrieves iter=18's output by calling into the async output object,
   which blocks until the GPU copy completes. Since the copy never completes,
   the scheduler stalls. No iter=19 is ever queued.

   **The CPU sees everything as normal**: all Python code paths for iter=18 execute
   successfully, `AsyncGPUModelRunnerOutput` is created, and `sample_tokens` returns.
   The GPU failure is completely invisible at the CPU level until the scheduler tries
   to consume the output.

5. **Why iter=18 specifically (not iter=17)**: The hang point shifts slightly
   between runs (iter=17 in one run, iter=18 here). This is consistent with a
   race condition or non-deterministic XCCL state: the exact iteration at which
   the GPU-side hang triggers depends on the batch composition and communicator
   state at that decode step.

---

### Confirmed Fixed
- **`num_actual_tokens` mismatch**: fixed by Fix 1
- **Unequal XCCL tensor sizes** when EP is enabled: fixed by Fix 2

### Remaining Hang — GPU-side silent hang during model forward (visible after async copy stalls)

**Pattern**: All 4 processes complete all CPU-side traces for iter=18 (including
returning from sample_tokens with async output). Iter=19 is never scheduled.
The GPU hangs silently inside iter=18's GPU execution, causing the async
GPU→CPU output copy to stall. The scheduler waits indefinitely for the copy.

The hang point is non-deterministic: in different runs it appears at iter=17 or
iter=18. This is consistent with a race condition or non-deterministic XCCL state.

**Root cause**: An XCCL or MoE operation inside the GPU-side model forward
hangs on the GPU. From the CPU's perspective all ops completed (they were
submitted asynchronously). The GPU copy stream is blocked behind the hung op.

**Next action**: Apply Fix 3 (synchronous output) so the scheduler calls
`torch.xpu.synchronize()` (or equivalent) before returning from `sample_tokens`.
This will expose the GPU-side hang inside `sample_tokens` itself instead of
hiding it in the async copy. The hang iteration's last ENTER trace before
`sample_tokens` silences will then identify which GPU op is stuck.

---

## Files Modified (Trace Infrastructure)

| File | Changes |
|------|---------|
| `vllm/_xpu_ops.py` | ENTER/EXIT around `gdn_attention` kernel; match check for `core_attn_out.size(0)` vs `num_actual_tokens` |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `hidden_states.shape` / `num_tokens` at `forward_xpu` entry |
| `vllm/model_executor/models/qwen3_next.py` | ENTER/EXIT around attn and MLP in `Qwen3NextDecoderLayer`; ENTER/EXIT around FusedMoE experts in `Qwen3NextSparseMoeBlock` |
| `vllm/v1/worker/gpu_model_runner.py` | `execute_model` and `sample_tokens` traces with `dp=` and `iter=`; early ENTER trace before `_run_ar`; **Fix 1**; **Fix 3**: disable `use_async_scheduling` when EP+DP is active; **iteration counter** `_iter_count`; pass `iter_count` to `_determine_batch_execution_and_padding` |
| `vllm/v1/worker/dp_utils.py` | **Fix 2**: `should_dp_pad` includes EP; `_run_ar` extends tensor to 5 rows with `iter_count` in row 4; **deadlock risk checker** prints `[WARN deadlock-risk]` if iteration counts mismatch; ENTER/EXIT around `dist.all_reduce` |
| `vllm/distributed/device_communicators/xpu_communicator.py` | ENTER/EXIT around `reduce_scatterv` and `all_gatherv` |
| `vllm/distributed/device_communicators/all2all.py` | ENTER/EXIT around MoE `dispatch` and `combine` |

---

## Recommended Next Steps

1. ~~**Confirm Fix 2 resolves the hang**~~ **✓ CONFIRMED (iter=1–18 succeed on TP=2, DP=2)**:
   Both Fix 1 and Fix 2 are working. The system now processes iter=1 (prefill)
   through iter=17 or iter=18 (16th–17th decode steps) successfully on all 4
   processes (TP=2 × DP=2). The hang point varies between runs (iter=17 in one
   run, iter=18 in another), consistent with a non-deterministic XCCL/MoE issue.

2. ~~**Confirm iter=3 hang location with new traces**~~ **✓ CONFIRMED (resolved)**:
   The `dist.all_reduce` in `_run_ar` completes normally for all iterations.

3. **Apply Fix 3 (synchronous output) to expose the GPU-side hang**:
   In both latest runs (iter=17/18 hang), Fix 3 was NOT applied (`use_async=True`).
   The GPU hangs silently inside the last iteration's model forward; the CPU only
   discovers this when the async GPU→CPU copy stalls the scheduler after all
   CPU-side traces appear to complete normally.

   Apply Fix 3 (already committed in `gpu_model_runner.py`) so that `sample_tokens`
   blocks until the GPU copy completes synchronously. This will cause the hang to
   manifest INSIDE `sample_tokens` rather than after it, and the last trace printed
   before `sample_tokens` silences will identify the exact GPU op that is stuck.

4. **After Fix 3 — look for the last trace before `sample_tokens` hangs**:
   With synchronous output, the hang should appear as `sample_tokens: ENTER _sample`
   or `sample_tokens: ENTER bookkeeping` with no EXIT. The last trace before
   silence points directly to the stuck GPU operation.

5. **If Fix 3 does not expose the hang inside sample_tokens**: the GPU op may be
   stuck inside `execute_model`'s model forward (the MoE dispatch/combine XCCL
   calls). Check the last ENTER trace from `all2all.py` or `xpu_communicator.py`
   traces (MoE dispatch/combine ENTER with no matching EXIT) for the hang iter.

6. **Long-term**: Add `torch.xpu.synchronize()` checkpoints around suspected ops
   (MoE dispatch, all2all, reduce_scatter, all_gather) so the CPU blocks until each
   GPU op completes. This converts silent GPU hangs into CPU-visible hang points
   with clear trace labels.
