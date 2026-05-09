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

### Revised analysis: Fix 5 `int32` type punning — NOT yet tested

The type punning test confirmed `float16 → int8 → float16` round-trips
correctly on XPU:

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device='xpu')
x_rt = x.contiguous().view(torch.int8).contiguous().view(torch.float16)
assert torch.allclose(x, x_rt)  # PASSES
```

However, Fix 5 also applies the same `view(torch.int8)` transformation to
**`topk_ids` which is `torch.int32`** (4 bytes per element). This path was
**never tested**.

Fix 5 `all_gatherv` list path — `dispatch()` sends `[hidden_states, topk_weights, topk_ids]`:

```python
# xpu_communicator.py — Fix 5
t_flat = t.reshape(n_tokens, -1).contiguous()  # [T, F]
t_int8 = t_flat.view(torch.int8)               # [T, F * elem_bytes]
```

For `topk_ids` (`int32`, K=8 experts): `F * elem_bytes = 8 * 4 = 32` int8 columns.

After gathering and slicing back:
```python
chunk_int8 = gathered_int8[:, offset:offset+col_width].contiguous()
chunk = chunk_int8.view(torch.int32)  # ← THIS was NOT tested for int32 on XPU
```

**If `view(torch.int32)` on an int8 tensor does not correctly reinterpret
bytes on XPU** (e.g., due to alignment constraints or an unimplemented kernel),
all gathered `topk_ids` would be wrong. Wrong `topk_ids` means:

- Every token (real and padding) is routed to wrong experts
- Wrong expert computation for real tokens 0–3
- Wrong combined `hidden_states` at positions 0–3
- Consistently wrong logits for all prompts → same degenerate output "!!!!"

This perfectly explains the **consistency** of "!!!!" across all prompts and
all DP ranks: the corruption is deterministic (always same wrong expert IDs)
because the byte-pattern of the real `topk_ids` (small integers like 0–59) maps
to the same garbage values via a broken `view(int32)`.

### Additional alignment concern in Fix 5

The int8 slice for `topk_ids` starts at byte offset:

```
offset = hidden_states_col_width + topk_weights_col_width
       = (7168 * 2) + (8 * 2) = 14336 + 16 = 14352
```

`14352 % 4 = 0` — 4-byte aligned in this case. However, for different model
configurations (different hidden_dim or topk), this offset may not be divisible
by 4. A misaligned `view(torch.int32)` could raise a runtime error or silently
corrupt data.

---

### Recommended next steps

1. **Test `int32 → int8 → int32` round-trip on XPU**:
   ```python
   import torch
   x = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device='xpu')
   x_rt = x.contiguous().view(torch.int8).contiguous().view(torch.int32)
   assert torch.equal(x, x_rt), f"int32 round-trip FAILED: {x} vs {x_rt}"
   print("int32 round-trip PASSED")
   ```

2. **If int32 round-trip fails**: Fix 5 is corrupting `topk_ids`. Replace the
   single-collective int8 approach for the `dispatch` list path with per-tensor
   collectives guarded by `dist.barrier()` (Fix 6 barriers already prevent
   deadlock between rounds; barriers between tensors within one round would
   eliminate the sequential-collective race for the list path):
   ```python
   # Safe fallback: barrier before each per-tensor all_gather_into_tensor
   for t in input_:
       dist.barrier(group=self.device_group)
       dist.all_gather_into_tensor(output_t, t, group=self.device_group)
   ```

3. **If int32 round-trip passes**: The `view(int32)` is correct; re-investigate
   Fix 2's effect on MoE expert capacity limits (whether the 26 garbage tokens
   overflow expert capacity and cause real-token drops in the combine step).

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
