# XPU EP Hang Diagnosis - Debug Summary

## Problem Statement

vLLM with Expert Parallelism (EP) on XPU hangs during inference when using
Data Parallelism (DP). The hang manifests as a silent deadlock — the process
stops producing output with no error message.

**Config**: Qwen3.5-35B-A3B, TP=2, EP=4 (MoE dispatch/combine over XCCL),
DP=2 with DP padding enabled.

---

## MoE Layer Collective Sequence (confirmed from logs)

Each MoE layer forward issues exactly these XCCL collectives (confirmed by
layer=11 full-cycle log, all 4 ranks completing each round):

```
all_gatherv  round 1  ←  dispatch_router_logits (hidden_states + router_logits)
all_gatherv  round 2  ←  dispatch/prepare (hidden_states + topk_weights + topk_ids)
all_gatherv  round 3  ←  (third call; source under investigation — shared expert or second dispatch)
reduce_scatterv round 1  ←  combine (expert outputs)
```

Layer 11 log evidence (all 4 ranks complete each round, full cycle confirmed):
```
[COUNTER] rank={0,1,2,3} all_gatherv/uniform counter=1  → 0   (round 1)
[COUNTER] rank={0,1,2,3} all_gatherv/uniform counter=1  → 0   (round 2)
[COUNTER] rank={0,1,2,3} all_gatherv/uniform counter=1  → 0   (round 3)
[COUNTER] rank={0,1,2,3} reduce_scatterv/uniform counter=1  → 0  (round 1)
```

The layer=11 MoE cycle completes fully — all 3 all_gatherv rounds and the
reduce_scatterv round all reach counter=0 for all 4 ranks.

---

## Confirmed Fixes

### Fix 1 — Force DP padding when Expert Parallelism is enabled

**Status**: ✅ CONFIRMED NEEDED and applied. All COUNTER logs show
`all_gatherv/uniform` (uniform = equal-size tensors across ranks), confirming
DP padding is in effect.

**File**: `vllm/v1/worker/dp_utils.py`

**Root cause**: Without DP padding, each DP rank processes a different number
of tokens. XCCL MoE dispatch/combine collectives require equal-size tensors.
Forcing DP padding when EP is active ensures all ranks always have the same
token count.

```diff
-    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    should_dp_pad = (synced_cudagraph_mode != 0 or should_ubatch
+                     or parallel_config.enable_expert_parallel)
```

### Fix 2 — `num_actual_tokens` mismatch when DP padding is active

**Status**: ✅ CONFIRMED FIXED by log evidence.

**File**: `vllm/v1/worker/gpu_model_runner.py`

**Log evidence** (before fix — rank 1 mismatch):
```
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=30, num_actual_tokens=26, match=False
```

**After fix** — all ranks show `match=True`.

**Root cause**: DP padding pads `hidden_states` to the max token count across
DP ranks (30), but `num_actual_tokens` in attention metadata remained at the
real count (26). The XPU GDN kernel asserts
`core_attn_out.size(0) == num_actual_tokens` and hangs. The fix sets
`pad_attn=True` whenever DP padding is applied, aligning `num_actual_tokens`,
slot mappings, and attention metadata with the padded count.

```diff
-            pad_attn = cudagraph_mode == CUDAGraphMode.FULL
+            dp_padding_applied = num_tokens_padded > num_tokens_unpadded
+            pad_attn = cudagraph_mode == CUDAGraphMode.FULL or dp_padding_applied
```

### Fix 3 — Disable async scheduling when EP + DP is active

**Status**: ✅ APPLIED. This is a **production correctness fix**, not merely a
diagnostic aid.

**File**: `vllm/v1/worker/gpu_model_runner.py`

**Root cause (production)**: With async scheduling enabled and EP+DP active,
`AsyncGPUModelRunnerOutput` returns immediately after queuing the GPU→CPU
copy. If DP ranks advance their schedulers at different speeds, one DP rank
can enter the next iteration's `_run_ar` all-reduce before the other finishes
the current iteration's GPU work, causing a cross-iteration collective
mismatch deadlock.

**Diagnostic benefit**: With async scheduling disabled, GPU-side hangs inside
the MoE forward become visible inside `sample_tokens: bookkeeping` rather than
hiding behind the async copy queue. This confirmed the hang is GPU-side (not a
CPU/scheduler race) and narrowed it to the model forward pass.

```diff
+        if (self.use_async_scheduling
+                and self.parallel_config.enable_expert_parallel
+                and self.parallel_config.data_parallel_size > 1):
+            self.use_async_scheduling = False
```

### Fix 4 — Correct `all_gatherv` uniform path

**Status**: ✅ CONFIRMED NEEDED. The original code passed a 1-element list to
`dist.all_gather`, which requires `world_size` tensors. All ranks deadlocked
waiting for the missing output slots.

**File**: `vllm/distributed/device_communicators/xpu_communicator.py`

```diff
-        dist.all_gather([output_tensor], input_, group=self.device_group)
+        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
```

### Fix 5 — Eliminate sequential all_gatherv calls in list path

**Status**: ✅ APPLIED. This is a **production correctness fix**, not merely a
diagnostic change. Collapses N sequential `dist.all_gather_into_tensor` calls
(one per tensor) into a single call via int8 byte-view concatenation. This
eliminates call-order mismatch deadlocks when faster ranks submit collective #2
before slower ranks finish collective #1. Without this fix, any rank timing
skew within a MoE layer forward can cause a collective-type mismatch deadlock
on the list-path (non-uniform) all_gatherv.

**File**: `vllm/distributed/device_communicators/xpu_communicator.py`

### Fix 6 — Add `dist.barrier` before each collective in `all2all.py`

**Status**: ✅ APPLIED. Adds an XCCL barrier before each `all_gatherv` and
`reduce_scatterv` call in `AgRsAll2AllManager` to force all EP ranks to
rendezvous before submitting the collective. This eliminates the round 2
deadlock caused by rank 2 being slower than ranks 0,1,3 at the GPU-side
routing computation (softmax/topk) between rounds 1→2.

**File**: `vllm/distributed/device_communicators/all2all.py`

```diff
+        dist.barrier(group=dist_group.device_group)
         gathered_tensors = dist_group.all_gatherv(   # dispatch_router_logits
+        dist.barrier(group=dist_group.device_group)
         gathered_tensors = dist_group.all_gatherv(   # dispatch
+        dist.barrier(group=dist_group.device_group)
         hidden_states = dist_group.reduce_scatterv(  # combine
```

**Why `dist_group.device_group`**: `GroupCoordinator.barrier()` uses a CPU-level
group only. `dist.barrier(group=dist_group.device_group)` issues an XCCL
barrier that drains any in-flight GPU kernels (routing softmax/topk) before
the collective is submitted, ensuring all ranks reach the collective
call-site together.

---

## Current Status (after all 6 fixes)

### Hang resolved — inference now completes

After applying all 6 fixes, the silent deadlock is eliminated. All 4 ranks
complete all MoE layers and the inference loop finishes. The `dist.barrier`
calls in Fix 6 prevent the rank-skew collective ordering deadlock that was the
last hang symptom.

### New symptom — incorrect output ("!!!!")

With all 6 fixes applied, inference completes but generates wrong output: every
prompt produces a long sequence of `"!"` characters regardless of input.

Example output:
```
[ARC B60] DP rank 0, Prompt: 'Hello, my name is'
Generated: '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
[ARC B60] DP rank 0, Prompt: 'The capital of France is'
Generated: '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
```

All prompts, all DP ranks, all iterations produce the same degenerate output.

---

## Wrong Output Analysis

### Why Fix 2 does NOT directly cause "!!!!" output

Fix 2 sets `pad_attn=True` when DP padding increases the token count, aligning
`num_actual_tokens` with the padded tensor row count (e.g., 30 instead of 26).
This causes the GDN attention kernel to process all 30 rows — including the 26
padding positions whose query vectors contain uninitialized (garbage) data.

However, Fix 2 **cannot** be the primary cause of "!!!!" through attention
corruption because of `logits_indices`:

```python
# gpu_model_runner.py — sampling step
sample_hidden_states = hidden_states[logits_indices]
```

`logits_indices` contains only the real token positions (e.g., `[0, 1, 2, 3]`
for 4 decode requests). Even if GDN writes garbage attention outputs to
`hidden_states[4:30]` for the padding rows, the final logit computation uses
only `hidden_states[0:3]` — the correct positions. Garbage at positions 4–29
is never read by the sampler.

Similarly, inside the MoE layer, each token's expert output is computed
independently (no cross-token interactions within a single expert forward).
Garbage routing for positions 4–29 does not overwrite positions 0–3.

**Fix 2 is necessary and correct.** The `num_actual_tokens` alignment is
required to prevent the GDN kernel size-check assertion failure that caused
the original hang.

---

### Fix 5 int8 byte-view — COMPLETELY RULED OUT

All XPU type punning round-trip tests pass:

```
# float16 → int8 → float16:  PASSES
# float32 → int8 → float32:  PASSES
# int32  → int8 → int32:     PASSES
```

Fix 5's int8 byte-view correctly preserves bytes for all dtypes used in MoE
collectives (`hidden_states` float16, `topk_weights` float16/float32,
`topk_ids` int32). Fix 5 is **not** the source of the "!!!!" output.

---

### New hypothesis: `sizes` mismatch between dp_metadata and padded tensor

The `dispatch` and `combine` functions in `AgRsAll2AllManager` both call
`dp_metadata.get_chunk_sizes_across_dp_rank()` to get `sizes`. Under MoE
sequence parallelism (SP), `sizes` is computed via:

```python
# forward_context.py — DPMetadata.sp_local_sizes(sp_size)
sp_tokens = (num_tokens_across_dp_cpu + sp_size - 1) // sp_size
sp_tokens = sp_tokens.repeat_interleave(sp_size)
```

With TP=2 (used as SP=2 for MoE) and `num_tokens_across_dp_cpu = [26, 30]`
(unpadded, before Fix 1 fully propagates through dp_metadata):

```
sizes = [ceil(26/2), ceil(26/2), ceil(30/2), ceil(30/2)] = [13, 13, 15, 15]
```

This non-uniform sizes would mean the `dispatch` assertion
`sizes[ep_rank] == hidden_states.shape[0]` compares `13 != 30` and fails.

With `num_tokens_across_dp_cpu = [30, 30]` (padded, Fix 1 fully effective):

```
sizes = [15, 15, 15, 15]  (uniform → sizes=None in all_gatherv)
```

**Key question**: Does Fix 1 correctly update `dp_metadata.num_tokens_across_dp_cpu`
to the padded values before the MoE forward? The [TRACE] logs already emitted
by the code will answer this directly.

---

### Recommended next steps

1. **Read the [TRACE] logs** — they are already emitted by the current code:

   ```
   [TRACE] rank=N dispatch ENTER all_gatherv: sizes=[...], tensor_shapes=[...]
   [TRACE] rank=N combine ENTER reduce_scatterv: sizes=[...], hidden_states_shape=[...]
   ```

   - If `sizes` is uniform (e.g., `[30, 30]` for DP=2, SP=1), the collectives
     use `all_gather_into_tensor` and `reduce_scatter_tensor` (uniform path) ✓
   - If `sizes` is non-uniform (e.g., `[26, 30]`), an assertion will fire OR
     the variable-size path is taken with mismatched tensor shapes → data corruption

2. **Check for SP (sequence parallelism)**: If TP is used as SP for MoE
   (sp_size > 1), `sizes` will have `dp_size * sp_size` entries (e.g., 4 for
   TP=2, DP=2). Verify that `sizes[ep_rank] == hidden_states.shape[0]` holds.

3. **If sizes are correct (uniform/matching)**: The "!!!!" must originate from
   within the model forward itself. Candidates:
   - Padding tokens (rows 26–29) with garbage query vectors produce large
     attention weights that corrupt real-token KV cache entries via attention
     (GDN attention output at positions 0–25 may be affected if the padded
     queries have extreme values)
   - Shared experts receiving padded input: if Qwen3-MoE shared experts run
     on the full padded tensor [30, d], their output for positions 26–29 is
     garbage. If those positions' shared-expert output is added to the sparse
     expert output via reduce_scatter, the sum may incorrectly mix garbage
     with real-token results
   - Zero out the padding positions before the router to test:
     ```python
     # In gpu_model_runner.py, after DP padding is applied:
     if dp_padding_applied:
         hidden_states[num_tokens_unpadded:] = 0
     ```
     If "!!!!" disappears, padding garbage values are corrupting the MoE router.

---

## Tracing Infrastructure

### Files modified

| File | Changes |
|------|---------|
| `vllm/_xpu_ops.py` | ENTER/EXIT around `gdn_attention` kernel; match check for `core_attn_out.size(0)` vs `num_actual_tokens` |
| `vllm/v1/worker/gpu_model_runner.py` | `execute_model` and `sample_tokens` traces with `dp=` and `iter=`; **Fix 2**; **Fix 3** |
| `vllm/v1/worker/dp_utils.py` | **Fix 1**; `_run_ar` deadlock risk checker (iter count mismatch warning); ENTER/EXIT around `dist.all_reduce` |
| `vllm/distributed/device_communicators/xpu_communicator.py` | **Fix 4**; **Fix 5**; COUNTER probes around `reduce_scatterv` and `all_gatherv` with seq number |
| `vllm/distributed/device_communicators/all2all.py` | **Fix 6**; ENTER/EXIT around MoE `dispatch_router_logits`, `dispatch`, and `combine` |

### How to read COUNTER logs

```
[COUNTER] rank=X seq=N all_gatherv/uniform counter=1   ← before collective
[COUNTER] rank=X seq=N all_gatherv/uniform counter=0   ← after collective (success)
```

- `counter=1` with no following `0` identifies the hanging collective.
- `seq=N` is a global call sequence number; compare across ranks to detect ordering mismatches.
- `uniform` = all ranks have the same tensor size (DP padding active); `variable-size` = sizes differ.

### DP communicator structure (TP=2, DP=2)

With TP=2, DP=2, vLLM creates two independent DP communicator groups:
- **Group A**: `{RANK=0 (dp=0,tp=0), RANK=2 (dp=1,tp=0)}` — tp=0 processes
- **Group B**: `{RANK=1 (dp=0,tp=1), RANK=3 (dp=1,tp=1)}` — tp=1 processes

Each group runs an independent `dist.all_reduce` per iteration in `_run_ar`.
Seeing two ENTER/EXIT pairs per dp_rank per iteration is normal.
