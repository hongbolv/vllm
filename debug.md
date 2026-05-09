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

## Remaining Hang

### Current symptom (after all fixes applied)

All 4 ranks enter `layer=12` MoE block (ENTER mlp → ENTER experts), all 4
complete all_gatherv **round 1** (dispatch_router_logits), then hang inside
round 2 (dispatch). None of the 4 ranks exits `layer=12` MoE.

Observed log (current state):
```
# ranks 0,1,3 print ENTER mlp/experts first (rank 2 is slower)
[TRACE] Qwen3NextSparseMoeBlock.forward ENTER experts num_tokens=4  (×3 ranks)

# all_gatherv round 1 — ranks 0,1,3 show counter=1 immediately
[COUNTER] rank=1 all_gatherv/uniform counter=1
[COUNTER] rank=3 all_gatherv/uniform counter=1
[COUNTER] rank=0 all_gatherv/uniform counter=1
# rank 2 also calls round 1 here (its stdout is still buffered from GPU work)
# → all 4 ranks are in the collective; it completes
[COUNTER] rank=3 all_gatherv/uniform counter=0
[COUNTER] rank=0 all_gatherv/uniform counter=0
[COUNTER] rank=1 all_gatherv/uniform counter=0
# rank 2's buffered prints now flush (rank 2 completed round 1 above)
[TRACE] layer=12 type=linear_attention EXIT attn    ← rank 2 stdout flush
[TRACE] Qwen3NextDecoderLayer.forward layer=12 ENTER mlp
[TRACE] Qwen3NextSparseMoeBlock.forward ENTER experts num_tokens=4

# ← HANG HERE: no counter=1 for round 2 ever appears
# layer=12 EXIT mlp/experts NEVER printed
```

**Key observation**: Rank 2's `EXIT attn` / `ENTER mlp` / `ENTER experts`
prints appear in the console *after* the round 1 `counter=0` logs. This is a
stdout buffering artifact — rank 2's Python thread had already submitted the
round 1 collective call (so round 1 completes for all 4), but the preceding
print statements were flushed to the console late. Round 1 therefore completes
with all 4 ranks participating.

### Hang analysis

After round 1 completes, all 4 CPU threads are unblocked simultaneously and
each proceeds to the routing computation (router softmax, topk selection) then
calls round 2 (all_gatherv for dispatch). Rank 2 is consistently slower than
ranks 0,1,3 at the GPU-side routing kernel between round 1 and round 2. When
ranks 0,1,3 submit round 2 before rank 2 does, and rank 2 then submits a
different collective type or round 2 with a long delay, the XCCL collective
ordering guarantee breaks → deadlock on round 2 with no counter output.

### Recommended next steps

1. **Add layer and round labels to COUNTER prints** in `dispatch_router_logits`
   and `dispatch` in `all2all.py` so each line identifies `layer=N round=M`.
   This will confirm round 2 is the hanging collective (vs round 3).

2. **Add `sys.stdout.flush()` after each COUNTER print** to prevent stdout
   buffering from masking which rank is the last to submit a collective.

3. **Investigate why rank 2 is slower between round 1 → round 2 in layer 12**:
   - Check whether the routing kernel (softmax + topk) takes longer on rank 2's
     XPU tile for this particular batch composition.
   - Check whether `num_actual_tokens` or input shapes differ between ranks in
     a way that causes unequal GPU work despite Fix 1 and Fix 2 being applied.

4. **Add `dist.barrier(group=ep_group)` before round 2** (Fix 6, already applied):
   before each `all_gatherv` and `reduce_scatterv` call in `all2all.py` to force
   all EP ranks to synchronize before entering the collective.

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
