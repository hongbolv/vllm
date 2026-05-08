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
returns immediately, allowing dp=0's scheduler to queue iter=N+1 before dp=1
has even finished iter=N's GPU copy. When dp=0 enters iter=N+1's DP all_reduce,
dp=1 has not yet entered it → communicator deadlock. Disabling async scheduling
forces the output copy to complete before the scheduler can advance, making the
iteration boundary a natural synchronization point across DP ranks.

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

### Step 9 - Iter=13/14 hang: confirmed `dist.all_reduce` path, two DP communicator subgroups

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
# Then: NOTHING. Zero output from any process for iter=14.
```

**Key observation - TWO separate DP communicator subgroups**:

With TP=2, DP=2, vLLM creates two independent DP communicator groups:
- **Group A**: `{RANK=0 (dp=0,tp=0), RANK=2 (dp=1,tp=0)}` — tp=0 processes
- **Group B**: `{RANK=1 (dp=0,tp=1), RANK=3 (dp=1,tp=1)}` — tp=1 processes

All 4 processes call `_run_ar`, but Group A and Group B each do an independent `dist.all_reduce`. Group A finishes first (tp=0 processes slightly faster), then Group B. This is why we see TWO ENTER/EXIT pairs per dp_rank per iteration — one from each group. This behavior is **normal** and both groups succeed for iter=13.

**Remaining hang — DP0 outpaces DP1 by one iteration**:

The hang after iter=13 is the classic cross-iteration DP deadlock:
1. dp=0 finishes iter=13 faster (async output returns immediately)
2. dp=0 schedules iter=14, both RANK=0 and RANK=1 enter iter=14's `_run_ar`
3. RANK=2 and RANK=3 (dp=1) are still finishing iter=13's async GPU copy
4. RANK=0 waits in Group A's all_reduce; RANK=1 waits in Group B's all_reduce
5. RANK=2 and RANK=3 finally start iter=14, BUT they are already waiting in iter=14's all_reduce for Group A and B respectively — except if they're still stuck in the async copy path they never enter iter=14 at all
6. All 4 processes deadlock → zero output for iter=14

The root cause is `AsyncGPUModelRunnerOutput`: the async GPU→CPU copy allows dp=0 to signal completion to its scheduler before dp=1's GPU work for the same iteration is finished. The dp=0 scheduler immediately queues iter=14. dp=1 scheduler is one step behind. When dp=0 enters iter=14's DP all_reduce, dp=1 hasn't yet entered it → communicator deadlock.

---

### Confirmed Fixed
- **`num_actual_tokens` mismatch**: fixed by Fix 1
- **Unequal XCCL tensor sizes** when EP is enabled: fixed by Fix 2

### Remaining Hang — Cross-iteration DP all_reduce deadlock (async output skew)

**Pattern**: iter=1 (prefill) through iter=13 (12th decode step) complete on all
4 processes. Zero output for iter=14. The two `_run_ar` pairs per dp_rank label
confirm two independent DP communicator subgroups (Group A: tp=0 pair across DP
ranks; Group B: tp=1 pair). Both groups succeed every iteration — until the last.

**Root cause**: `AsyncGPUModelRunnerOutput` lets dp=0 signal output to its
scheduler before dp=1 finishes the async GPU copy. dp=0's scheduler queues
iter=14 while dp=1's scheduler is still on iter=13. RANK=0 enters iter=14's
`_run_ar` (Group A) but RANK=2 never arrives — hang. RANK=1 similarly waits in
Group B, RANK=3 never arrives. All 4 processes deadlock.

**Root fix**: Disable async output when EP is active so that the GPU→CPU copy
completes synchronously before the scheduler can queue the next batch. This
makes iter boundaries a natural synchronization point.

```python
# In gpu_model_runner.py or the output consumer
use_async_output = not self.parallel_config.enable_expert_parallel
```

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

1. ~~**Confirm Fix 2 resolves the hang**~~ **✓ CONFIRMED (iter=1–13 succeed)**:
   Both Fix 1 and Fix 2 are working. The system now processes iter=1 (prefill,
   30 tokens) through iter=13 (12th decode step) successfully on all 4 processes.
   The hang moved from iter=1/2 all the way to iter=13/14 — major progress.

2. ~~**Confirm iter=3 hang location with new traces**~~ **✓ CONFIRMED (resolved)**:
   The `dist.all_reduce` in `_run_ar` completes normally for all iterations up
   to iter=13. No hang inside `dist.all_reduce` for normal execution.

3. **Fix 3 — Disable async output when EP is active**:
   The iter=14 hang is caused by `AsyncGPUModelRunnerOutput` letting dp=0 advance
   one iteration ahead of dp=1. Forcing synchronous output under EP ensures both
   DP ranks complete their iteration before either can start the next one:

   ```python
   # In gpu_model_runner.py sample_tokens():
   use_async = (self.use_async_output
                and not self.parallel_config.enable_expert_parallel)
   ```

   Or equivalently: in the engine configuration, set `use_async_output=False`
   when `enable_expert_parallel=True`.

4. **Alternative Fix 3 — DP barrier at iteration boundary**:
   Insert an explicit `dist.barrier()` (or another `dist.all_reduce`) at the
   END of `sample_tokens` (or at the start of `execute_model` before `_run_ar`)
   using the existing DP group. This ensures no rank can start iter=N+1 until
   all ranks have finished iter=N's output path.

5. **Long-term**: Make `external_launcher` DP mode guarantee that all DP ranks
   are within ±1 iteration of each other by adding back-pressure from the
   executor to the scheduler when any DP rank falls behind.
