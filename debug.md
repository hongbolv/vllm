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

Layer 11 log evidence (all 4 ranks complete each round):
```
[COUNTER] rank={0,1,2,3} all_gatherv/uniform counter=1  → 0   (round 1)
[COUNTER] rank={0,1,2,3} all_gatherv/uniform counter=1  → 0   (round 2)
[COUNTER] rank={0,1,2,3} all_gatherv/uniform counter=1  → 0   (round 3)
[COUNTER] rank={2,...}   reduce_scatterv/uniform counter=1     (round 1, log cut off)
```

---

## Confirmed Fixes

### Fix 1 — `num_actual_tokens` mismatch when DP padding is active

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

### Fix 2 — Force DP padding when Expert Parallelism is enabled

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

### Fix 3 — Disable async scheduling when EP + DP is active

**Status**: ✅ APPLIED. Confirmed effective for diagnosis: with async scheduling
disabled, the hang becomes visible inside `sample_tokens: bookkeeping` rather
than hiding behind the async GPU→CPU copy.

**File**: `vllm/v1/worker/gpu_model_runner.py`

**Root cause**: `AsyncGPUModelRunnerOutput` returns immediately after queuing
the GPU→CPU copy asynchronously. The GPU hangs inside the MoE forward, the
queued copy never completes, and the scheduler stalls waiting for the copy
with no visible error. Disabling async output forces the CPU to block until
the copy completes, making GPU-side hangs visible in `sample_tokens`.

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

**Status**: ✅ APPLIED. Collapses N sequential `dist.all_gather_into_tensor`
calls (one per tensor) into a single call via int8 byte-view concatenation.
This eliminates call-order mismatch deadlocks when faster ranks submit
collective #2 before slower ranks finish collective #1.

**File**: `vllm/distributed/device_communicators/xpu_communicator.py`

### Fix 6 — Remove `torch.xpu.synchronize()` from around XCCL collectives

**Status**: ✅ CONFIRMED NEEDED. Local-only GPU drains caused faster ranks to
immediately submit the next collective before slower ranks finished the current
one on the GPU side, producing XCCL call-order mismatch deadlocks.

**File**: `vllm/distributed/device_communicators/xpu_communicator.py`

---

## Remaining Hang

### Current symptom (after all fixes applied)

All 4 ranks enter `layer=12` MoE block (ENTER mlp → ENTER experts), complete
all_gatherv **round 1** successfully, then hang. None of the 4 ranks exits
`layer=12` MoE (no EXIT experts or EXIT mlp is ever printed).

Layer 12 log evidence:
```
[TRACE] Qwen3NextSparseMoeBlock.forward ENTER experts num_tokens=4   (×4 ranks)
[COUNTER] rank={1,3,0,2} all_gatherv/uniform counter=1  → 0   (round 1 — all 4 complete)
# ← hang here: no round 2 all_gatherv counter ever appears
[TRACE] layer=12 EXIT mlp   ← NEVER printed
```

### Hang analysis

All 4 ranks complete all_gatherv round 1 together (all show counter=1→0).
After round 1 all CPUs unblock simultaneously and proceed to the next
collective (all_gatherv round 2). The hang occurs during round 2 — no
`counter=1` print appears for round 2.

**Why layer 11 completes but layer 12 does not:**

Layer 11 all three all_gatherv rounds complete with all 4 ranks synchronized.
Layer 12 completes round 1 with all 4 ranks, but hangs on round 2. This
indicates an asymmetry between the ranks that develops between round 1 and
round 2: likely rank 2's GPU is slower to finish the routing computation
(e.g., router softmax / topk selection submitted to the GPU queue after round
1), so when rounds proceed on the GPU side, rank 2's GPU enters round 2 late
relative to the other 3, causing a GPU-side XCCL collective ordering mismatch.

**Key question still open**: What makes layer 12 different from layer 11?
Possible explanations:
- Rank 2's GPU computation between round 1 and round 2 of layer 12 is
  significantly slower than in layer 11 (e.g., a different batch composition
  or a GPU kernel stall that accumulates over layers).
- There is a GPU-level ordering issue that does not yet appear in layer 11 but
  triggers at layer 12.

### Recommended next steps

1. **Add COUNTER prints with layer number** in `dispatch_router_logits` and
   `dispatch` in `all2all.py` so each COUNTER line identifies which round
   (round 1 / round 2 / round 3) and which layer it belongs to. This will
   pinpoint whether the hang is in round 2 or round 3 of layer 12.

2. **Add `sys.stdout.flush()` / explicit flush** after each COUNTER print to
   ensure no output is buffered when the hang occurs.

3. **Identify why round 2 hangs for layer 12 but not layer 11** by comparing
   timing between rounds across layers. If rank 2's GPU routing on layer 12 is
   genuinely slower, adding a `dist.barrier(group=ep_group)` before the round
   2 dispatch (in `all2all.py`'s `dispatch` method) will force all ranks to
   synchronize before entering the collective, eliminating the ordering race.

---

## Tracing Infrastructure

### Files modified

| File | Changes |
|------|---------|
| `vllm/_xpu_ops.py` | ENTER/EXIT around `gdn_attention` kernel; match check for `core_attn_out.size(0)` vs `num_actual_tokens` |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `hidden_states.shape` / `num_tokens` at `forward_xpu` entry |
| `vllm/model_executor/models/qwen3_next.py` | ENTER/EXIT around attn and MLP in `Qwen3NextDecoderLayer`; ENTER/EXIT around FusedMoE experts in `Qwen3NextSparseMoeBlock` |
| `vllm/v1/worker/gpu_model_runner.py` | `execute_model` and `sample_tokens` traces with `dp=` and `iter=`; **Fix 1**; **Fix 3** |
| `vllm/v1/worker/dp_utils.py` | **Fix 2**; `_run_ar` deadlock risk checker (iter count mismatch warning); ENTER/EXIT around `dist.all_reduce` |
| `vllm/distributed/device_communicators/xpu_communicator.py` | **Fix 4**; **Fix 5**; **Fix 6**; COUNTER probes around `reduce_scatterv` and `all_gatherv` with seq number |
| `vllm/distributed/device_communicators/all2all.py` | ENTER/EXIT around MoE `dispatch_router_logits`, `dispatch`, and `combine` |

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
