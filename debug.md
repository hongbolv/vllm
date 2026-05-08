# XPU EP Hang Diagnosis — Debug Summary

## Problem Statement

vLLM with Expert Parallelism (EP) on XPU hangs during inference when using
Data Parallelism (DP) with DP padding enabled. The hang manifests as a silent
deadlock — the process stops producing output with no error message.

**Config**: Qwen3.5-35B-A3B, TP=2, EP (MoE dispatch/combine over XCCL), DP padding enabled.

---

## Chronological Diagnosis

### Step 1 — Initial hypothesis: variable-size XCCL collectives

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

### Step 2 — DP padding causes `num_actual_tokens` mismatch

**Log evidence**:
```
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=30, num_actual_tokens=30, match=True   # DP rank 0
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=30, num_actual_tokens=26, match=False  # DP rank 1
```

**Root cause**: DP padding pads `hidden_states` (and thus `core_attn_out`) to
the max token count across DP ranks (30), but `num_actual_tokens` in attention
metadata remained at the real count for rank 1 (26). The XPU GDN kernel asserts
`core_attn_out.size(0) == num_actual_tokens` and fails/hangs.

**Fix** (commit `cd3b791` / `0130002`): In `gpu_model_runner.py`,
`pad_attn=True` is now set whenever DP padding increases the token count:

```python
dp_padding_applied = num_tokens_padded > num_tokens_unpadded
pad_attn = cudagraph_mode == CUDAGraphMode.FULL or dp_padding_applied
```

This ensures `num_actual_tokens = num_tokens_padded`, slot mappings are sized
for the padded count (with `-1` fill for padding slots), and attention metadata
uses the padded count.

**Masking for padding tokens** (not an issue for standard attention):
- Slot mappings: padding slots filled with `-1` → no KV cache writes
- `query_start_loc`: only accounts for real tokens
- `logits_indices`: selects only real tokens' hidden states for output

**Concern for GDN/Mamba layers**: GDN kernel processes tokens sequentially and
updates SSM/conv state. Padding tokens could introduce noise into state if they
are processed. This requires further investigation if incorrect outputs are
observed after fixing the crash.

### Step 3 — GDN attention no longer hangs, but system still hangs

After the `pad_attn` fix, `num_actual_tokens` matched and GDN attention exited
successfully:

```
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=4, num_actual_tokens=4, match=True
[TRACE] _gdn_attention_core_xpu_impl: EXIT gdn_attention kernel
[TRACE] gdn_linear_attn forward_xpu: hidden_states.shape=torch.Size([4, 2048]), num_tokens=4
```

Added ENTER/EXIT prints around the `gdn_attention` kernel call in `_xpu_ops.py`
(commit `9a12beb`) to confirm kernel completion.

### Step 4 — Narrowing hang to decoder layer / MoE level

Added trace prints in `Qwen3NextDecoderLayer.forward` and
`Qwen3NextSparseMoeBlock.forward` (commit `f507331`):

**Log evidence**:
```
[TRACE] Qwen3NextDecoderLayer.forward layer=37 type=linear_attention ENTER attn
[TRACE] gdn_linear_attn forward_xpu: hidden_states.shape=torch.Size([4, 2048]), num_tokens=4
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=4, num_actual_tokens=4, match=True
[TRACE] _gdn_attention_core_xpu_impl: EXIT gdn_attention kernel
[TRACE] Qwen3NextDecoderLayer.forward layer=37 type=linear_attention EXIT attn
[TRACE] Qwen3NextDecoderLayer.forward layer=37 ENTER mlp (Qwen3NextSparseMoeBlock)
[TRACE] Qwen3NextSparseMoeBlock.forward ENTER experts num_tokens=4
[TRACE] Qwen3NextSparseMoeBlock.forward EXIT experts
[TRACE] Qwen3NextDecoderLayer.forward layer=37 EXIT mlp
```

**Finding**: All attention layers (both `linear_attention`/GDN and
`full_attention`) and all MoE experts blocks for layers 36-39 complete
successfully. Hang occurs **after** all decoder layers finish.

### Step 5 — Hang is after model forward, in execute_model postprocess

Added trace prints in `execute_model` (commit `3f17a87`):

**Log evidence**:
```
[TRACE] execute_model: model forward complete, type(model_output)=Tensor
[TRACE] execute_model: postprocess ENTER, hidden_states.shape=torch.Size([4, 2048])
[TRACE] execute_model: ENTER logits_indices gather
[TRACE] execute_model: ENTER compute_logits
[TRACE] execute_model: EXIT compute_logits
[TRACE] execute_model: setting execute_model_state
[TRACE] execute_model: returning None (success)
```

**Finding**: `execute_model` completes and returns successfully. The hang is
downstream of `execute_model` — either in the executor's `collective_rpc`
handling or in `sample_tokens`.

### Step 6 — `sample_tokens` completes too

Added trace prints in `sample_tokens` (commit `3da5558`):

**Log evidence** (after adding DP rank info in commit `f516e32`):
```
[TRACE dp=0] sample_tokens: ENTER
[TRACE dp=0] sample_tokens: ENTER _sample
[TRACE dp=0] sample_tokens: EXIT _sample
[TRACE dp=0] sample_tokens: ENTER bookkeeping
[TRACE dp=0] sample_tokens: EXIT bookkeeping
[TRACE dp=0] sample_tokens: building ModelRunnerOutput
[TRACE dp=0] sample_tokens: ModelRunnerOutput built, use_async=True
[TRACE dp=0] sample_tokens: ENTER AsyncGPUModelRunnerOutput
[TRACE dp=0] sample_tokens: EXIT AsyncGPUModelRunnerOutput
[TRACE dp=0] sample_tokens: returning output (async)
```

Added granular prints inside `ModelRunnerOutput` and
`AsyncGPUModelRunnerOutput` construction (commit `13e3880`).

**Finding**: DP0's `sample_tokens` completes the first iteration successfully,
including the async GPU→CPU copy path.

### Step 7 — DP0/DP1 desync across iterations

**Final log evidence** (full log with DP rank labels):
```
# --- First iteration (both DP ranks complete) ---
[TRACE dp=0] execute_model: model forward complete, type(model_output)=Tensor
[TRACE dp=0] execute_model: postprocess ENTER, hidden_states.shape=torch.Size([4, 2048])
[TRACE dp=0] execute_model: ENTER logits_indices gather
[TRACE dp=0] execute_model: ENTER compute_logits
[TRACE dp=0] execute_model: EXIT compute_logits
[TRACE dp=0] execute_model: setting execute_model_state
[TRACE dp=0] execute_model: returning None (success)
[TRACE dp=0] sample_tokens: ENTER
[TRACE dp=0] sample_tokens: ENTER _sample
[TRACE dp=0] sample_tokens: EXIT _sample
[TRACE dp=0] sample_tokens: ENTER bookkeeping
[TRACE dp=0] sample_tokens: EXIT bookkeeping
[TRACE dp=0] sample_tokens: building ModelRunnerOutput
[TRACE dp=0] sample_tokens: ModelRunnerOutput built, use_async=True
[TRACE dp=0] sample_tokens: ENTER AsyncGPUModelRunnerOutput
[TRACE dp=0] sample_tokens: EXIT AsyncGPUModelRunnerOutput
[TRACE dp=0] sample_tokens: returning output (async)

# --- Second iteration (DP0 ahead of DP1 - desync detected) ---
[TRACE dp=0] execute_model: model forward complete, type(model_output)=Tensor
[TRACE dp=0] execute_model: postprocess ENTER, hidden_states.shape=torch.Size([4, 2048])
[TRACE dp=0] execute_model: ENTER logits_indices gather
[TRACE dp=0] execute_model: ENTER compute_logits
[TRACE] gdn_linear_attn forward_xpu: hidden_states.shape=torch.Size([4, 2048]), num_tokens=4  # DP1 still in model forward!
[TRACE dp=1] execute_model: model forward complete, type(model_output)=Tensor
[TRACE dp=1] execute_model: postprocess ENTER, hidden_states.shape=torch.Size([4, 2048])
[TRACE dp=0] execute_model: EXIT compute_logits
[TRACE dp=1] execute_model: ENTER logits_indices gather
[TRACE dp=0] execute_model: setting execute_model_state
[TRACE dp=0] execute_model: returning None (success)
[TRACE dp=1] execute_model: ENTER compute_logits
[TRACE dp=0] sample_tokens: ENTER
[TRACE dp=0] sample_tokens: ENTER _sample
[TRACE] _gdn_attention_core_xpu_impl: core_attn_out.size(0)=4, num_actual_tokens=4, match=True
[TRACE] _gdn_attention_core_xpu_im...   ← LOG TRUNCATED / HANG
```

---

## Root Cause Analysis

### Confirmed Fixed
- **`num_actual_tokens` mismatch** when DP padding is active: fixed in `0130002`
  by setting `pad_attn=True` when `num_tokens_padded > num_tokens_unpadded`.

### Remaining Hang — Cross-DP / Cross-iteration Synchronization

The log shows a **timing desync between DP0 and DP1** across iterations:

1. **First iteration**: Both DP ranks complete successfully.
2. **Second iteration**:
   - DP0 finishes `compute_logits` and enters `sample_tokens: ENTER _sample`
   - DP1 is still inside its model forward (GDN attention at layer ~37+)
   - DP0 may advance to its **third iteration's** model forward, entering a
     collective (MoE dispatch or TP all-reduce) while DP1 is still in the
     second iteration's collective
   - This causes a **collective operation mismatch** between iterations → hang

**Evidence for collective mismatch**:
- The log is truncated at `_gdn_attention_core_xpu_im...` (DP1 still in layer
  37 GDN attention during what appears to be a third-iteration forward)
- DP0 has already moved to `_sample` of the second iteration
- If DP0's TP ranks start the third iteration's TP/EP collectives before DP1's
  TP ranks finish the second iteration's, the XCCL communicators deadlock

### Hypothesis

DP0 is consistently faster than DP1 due to the async output path:
`AsyncGPUModelRunnerOutput` uses a non-blocking GPU→CPU copy. DP0's output
rank returns immediately while the copy completes in the background, allowing
the scheduler to immediately dispatch the next batch to DP0. DP1 may be slower
to return, causing the scheduler to dispatch N+1 to DP0 before DP1 finishes N.

Since TP ranks within each DP group share XCCL communicators, if DP0's TP
ranks start iteration N+1's collective while DP1's TP ranks are still in
iteration N, the collective ordering is violated.

---

## Files Modified (Trace Infrastructure)

| File | Changes |
|------|---------|
| `vllm/_xpu_ops.py` | ENTER/EXIT prints around `gdn_attention` kernel; match check for `core_attn_out.size(0)` vs `num_actual_tokens` |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `hidden_states.shape` and `num_tokens` print in `forward_xpu` |
| `vllm/model_executor/models/qwen3_next.py` | ENTER/EXIT around attn and MLP in `Qwen3NextDecoderLayer.forward`; ENTER/EXIT around FusedMoE experts in `Qwen3NextSparseMoeBlock.forward` |
| `vllm/v1/worker/gpu_model_runner.py` | `execute_model` stage traces (forward complete → logits → return); `sample_tokens` traces (ENTER → _sample → bookkeeping → ModelRunnerOutput → async output); **fix**: `pad_attn=True` when DP padding applied |
| `vllm/distributed/device_communicators/xpu_communicator.py` | ENTER/EXIT around `reduce_scatterv` and `all_gatherv` |
| `vllm/distributed/device_communicators/all2all.py` | ENTER/EXIT around MoE `dispatch` and `combine` |

---

## Recommended Next Steps

1. **Investigate scheduler/executor dispatch timing**: Add traces in the
   executor's `collective_rpc` dispatch to confirm whether DP0 starts a new
   `execute_model` before DP1's previous one completes.

2. **Check TP collective ordering across iterations**: The XCCL communicator
   for TP all-reduces is shared. If DP0 starts iteration N+1's TP all-reduce
   while DP1 is still in iteration N's TP all-reduce, a deadlock occurs.

3. **Synchronize DP rank outputs before next dispatch**: The executor should
   wait for all DP ranks to return from `execute_model` / `sample_tokens`
   before dispatching the next batch. Check if `unique_reply_rank` in
   `collective_rpc` causes the scheduler to return early and re-dispatch.

4. **Disable async output path** as a workaround: Set `use_async_output=False`
   to force synchronous GPU→CPU copies. This slows DP0 down, giving DP1 time
   to catch up, and may eliminate the desync. If this fixes the hang, the
   async path needs proper barrier synchronization before the next dispatch.
