# XPU EP Hang Diagnosis - Debug Summary

## Problem Statement

vLLM with Expert Parallelism (EP) on XPU hangs during inference when using
Data Parallelism (DP). The hang manifests as a silent deadlock — the process
stops producing output with no error message. After fixing the hang, the
inference output degrades to mostly `"!!!!"` tokens — caused by NaN
contamination propagated through the model layers.

**Config**: Qwen3.5-35B-A3B, TP=2, DP=2, EP=True (4x Intel ARC B60 GPUs).
See `examples/offline_inference/xpu_arc_b60_dp_ep.py` for the reference
configuration (rank mapping, prompt distribution, launch command).

**Observed symptom after hang fix**: First token(s) are correct but subsequent
decode tokens degrade to `"!!!!"`, e.g.:
```
DP rank 0: 'Hello, my name is' → ' Isha!!!!!!...'
DP rank 0: 'The capital of France is' → ' known as!!!!!!...'
```

---

## Applied Fixes — Status Summary

| Fix | File | Status | Description |
|-----|------|--------|-------------|
| Hongbo Fix 1 | `dp_utils.py` | ✅ Applied | Force DP padding when EP is enabled |
| Hongbo Fix 2 | `gpu_model_runner.py` | ✅ Applied | Align `pad_attn` with DP padding state |
| Hongbo Fix 3 | `gpu_model_runner.py` | ✅ Applied | Disable async scheduling for EP+DP |
| Hongbo Fix 4 | `gpu_model_runner.py` | ✅ Applied | Fix attention metadata for DP padding rows (prefill NaN fix) |
| Hongbo Fix 6 | `all2all.py`, `qwen3_next.py`, `attention.py` | ✅ Applied | Add XCCL barrier + zero-initialize attention output buffers |

---

## Applied Fixes

### Hongbo Fix 1 — Force DP padding when EP is enabled

**File**: `vllm/v1/worker/dp_utils.py`

Without DP padding, each DP rank processes a different number of tokens. XCCL
MoE dispatch/combine collectives require equal-size tensors. Forcing DP padding
when EP is active ensures all ranks always have the same token count. Removing
this fix causes immediate hang — confirmed DP padding is required to prevent
EP all-to-all collective deadlock.

```diff
-    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    should_dp_pad = (synced_cudagraph_mode != 0 or should_ubatch
+                     or parallel_config.enable_expert_parallel)
```

### Hongbo Fix 2 — Align `pad_attn` with DP padding state

**File**: `vllm/v1/worker/gpu_model_runner.py`

DP padding increases the token count, so attention metadata must be built with
padded sizes. The fix sets `pad_attn=True` whenever DP padding is applied, not
only when CUDAGraph FULL mode is active.

```diff
-            pad_attn = cudagraph_mode == CUDAGraphMode.FULL
+            dp_padding_applied = num_tokens_padded > num_tokens_unpadded
+            pad_attn = cudagraph_mode == CUDAGraphMode.FULL or dp_padding_applied
```

### Hongbo Fix 3 — Disable async scheduling for EP+DP

**File**: `vllm/v1/worker/gpu_model_runner.py`

With async scheduling and EP+DP, DP ranks can advance at different speeds,
causing cross-iteration collective mismatch deadlocks. One rank enters
iteration N+1's `all_reduce` while the other is still in iteration N.

```diff
+        if (self.use_async_scheduling
+                and self.parallel_config.enable_expert_parallel
+                and self.parallel_config.data_parallel_size > 1):
+            self.use_async_scheduling = False
```

### Hongbo Fix 4 — Fix attention metadata for DP padding rows (prefill NaN fix)

**File**: `vllm/v1/worker/gpu_model_runner.py`, `_build_attention_metadata()`

When DP padding is applied (`num_tokens_padded > num_tokens`), `query_start_loc`
ends at the real token count but the attention backend processes
`num_tokens_padded` rows. Rows beyond `num_tokens` have no sequence
assignment, causing softmax on all-`-inf` scores → NaN. Fix: create local
copies of `query_start_loc` and `seq_lens` and extend them to cover the
padding rows.

### Hongbo Fix 6 — Add XCCL barrier before MoE collectives

**File**: `vllm/distributed/device_communicators/all2all.py`

Adds an XCCL barrier before each `all_gatherv` and `reduce_scatterv` call in
`AgRsAll2AllManager` to force all EP ranks to rendezvous before submitting the
collective. Uses `dist.barrier(group=dist_group.device_group)`.

In the DP=2, TP=2, EP=True configuration, `dist_group.device_group` covers
all 4 ranks when called from `is_sequence_parallel=True` (EP group), or 2 DP
peers when called from `is_sequence_parallel=False`.

Also includes zero-initialization of attention output buffers
(`vllm/model_executor/layers/attention/attention.py`,
`vllm/model_executor/models/qwen3_next.py`): changed `torch.empty` →
`torch.zeros` and `torch.empty_like` → `torch.zeros_like`. With DP padding,
uninitialized padding rows on XPU (BMG) frequently contain NaN bit patterns;
using `torch.zeros` eliminates this contamination.

```diff
+        dist.barrier(group=dist_group.device_group)
         gathered_tensors = dist_group.all_gatherv(...)
```

---

## Additional Fixes

### Flash attention k/v contiguous fix

**File**: `vllm/_xpu_ops.py`

Added `.contiguous()` calls for `k` and `v` tensors passed to
`flash_attn_varlen_func`. Non-contiguous tensors can cause incorrect results
or crashes in the XPU flash attention kernel.

```diff
-            k=k,
-            v=v,
+            k=k.contiguous(),
+            v=v.contiguous(),
```

---

## NaN Root Cause Analysis

### TP=4/DP=1 reference case

TP=4/DP=1 runs successfully on the same 4x Intel ARC B60 hardware — no NaN,
no hang, correct output. This rules out XPU kernel numerical issues as the
root cause. NaN only appears with DP=2 where DP padding is active.

### Prefill NaN root cause — confirmed, fixed

**`[NAN_CHECK_POST_ATTN]`** first triggered at `dp_rank=0, layer_idx=3`
(first full-attention layer) with `nan_row_indices=[26,27,28,29]`,
`actual_nan_rows=4`, `padding_nan_rows=0`.

**`[ATTN_MASK_CHECK]`** confirmed the root cause:

- **dp_rank=0**: `seq_lens=[5,5,8,8]` (sum=26 real tokens), but
  `num_actual_tokens=30` (DP-padded to match dp_rank=1).
  `query_start_loc=[0,5,10,18,26]` ends at 26 — rows 26–29 are DP padding
  tokens with **no sequence assignment**.
- **dp_rank=1**: `seq_lens=[7,5,11,7]` (sum=30) = `num_actual_tokens=30`.
  No gap, no NaN.

The attention backend processes all 30 rows for dp_rank=0, but rows 26–29
have no valid attention mask. Their attention scores are all-`-inf`, causing
softmax to produce 0/0 = NaN. NaN propagates to all subsequent layers via
residual-add. dp_rank=0 layer 0–2 (GDN linear-attention layers) show no
prefill NaN — the GDN kernel processes only real tokens in prefill.

**Hongbo Fix 4 applied** (`_build_attention_metadata()`): extend `seq_lens` and
`query_start_loc` to cover DP padding rows by assigning them to the last
request (direction 1). All attention backends see consistent metadata with
no uncovered rows.

When `num_tokens_padded > num_tokens` (DP padding active), the fix:
1. Sets `attn_query_start_loc[num_reqs_padded] = num_tokens_padded` so the
   last entry covers all rows including padding.
2. Adds `padding_len` to `attn_seq_lens[num_reqs_padded - 1]` so the last
   request's causal attention mask covers the padding rows.

Local clones (`attn_query_start_loc_gpu`, `attn_query_start_loc_cpu`,
`attn_seq_lens`) are used so the shared buffers (`self.query_start_loc`,
`self.seq_lens`) used for KV cache `slot_mapping` are not corrupted.

```diff
+        attn_query_start_loc_gpu = self.query_start_loc.gpu[: num_reqs_padded + 1]
+        attn_query_start_loc_cpu = self.query_start_loc.cpu[: num_reqs_padded + 1]
+        attn_seq_lens = self.seq_lens[:num_reqs_padded]
+        if num_tokens_padded > num_tokens:
+            padding_len = num_tokens_padded - num_tokens
+            attn_query_start_loc_gpu = attn_query_start_loc_gpu.clone()
+            attn_query_start_loc_cpu = attn_query_start_loc_cpu.clone()
+            attn_seq_lens = attn_seq_lens.clone()
+            attn_query_start_loc_gpu[num_reqs_padded] = num_tokens_padded
+            attn_query_start_loc_cpu[num_reqs_padded] = num_tokens_padded
+            attn_seq_lens[num_reqs_padded - 1] += padding_len
         cm_base = CommonAttentionMetadata(
-            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
-            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
-            seq_lens=self.seq_lens[:num_reqs_padded],
+            query_start_loc=attn_query_start_loc_gpu,
+            query_start_loc_cpu=attn_query_start_loc_cpu,
+            seq_lens=attn_seq_lens,
             ...
             num_actual_tokens=num_tokens_padded,
```

### Decode NaN — open issue (GDN recurrent state)

**`[NAN_CHECK_POST_ATTN]`** triggered at `dp_rank=1, layer_idx=1` (GDN
linear-attention layer) during the decode phase with:
- `shape=[4, 2048]`, `num_actual_tokens=4` (no DP padding on dp_rank=1)
- `nan_count=86` per affected row (partial NaN — not full-row)
- Different rows affected in consecutive decode steps (`[0]`, then `[2]`)

Key observations:
- dp_rank=1 has no DP padding in prefill (`seq_lens` sum = `num_actual_tokens`),
  so the prefill fix code path **never executes** on dp_rank=1.
- Partial NaN (86/2048 elements) rules out attention mask issues (which
  produce whole-row NaN via softmax on all-`-inf`).
- NaN is newly generated at layer 1 decode — `[NAN_CHECK_PRE_ATTN]` for
  layer 1 does **not** trigger (input is clean).
- The GDN `_forward_core()` CUDA path is not called on XPU — the XPU path
  goes through `forward_xpu()` → `_gdn_attention_core_xpu_impl()` in
  `_xpu_ops.py`, which calls the SYCL kernel directly via
  `torch.ops._xpu_C.gdn_attention`.

**Current hypothesis**: GDN recurrent state (`ssm_state`) written during
prefill may be incorrect in EP+DP scenarios, or `cu_seqlens`
(`non_spec_query_start_loc`) may be computed incorrectly causing the
recurrent computation to operate on wrong token ranges.

The `[GDN_STATE_CHECK]` diagnostic validates:
1. `last_recurrent_state` returned from `chunk_gated_delta_rule`
2. Written `ssm_state` slots
3. `initial_state` read back
4. `cu_seqlens` consistency: starts at 0, monotonically increasing, final
   entry matches expected token count

---

## Active Diagnostic Traces

All diagnostics skip warmup/profiling passes (when `attn_metadata` is `None`).
Each uses a class-level flag to print ERROR only on first occurrence.

| Trace tag | File | Purpose |
|-----------|------|---------|
| `[NAN_CHECK_PRE_ATTN]` | `qwen3_next.py` | Detect NaN in `hidden_states` **before** attention call in `Qwen3NextDecoderLayer` |
| `[NAN_CHECK_POST_ATTN]` | `qwen3_next.py` | Detect NaN in `hidden_states` **after** attention call; reports `actual_nan_rows` vs `padding_nan_rows` |
| `[GDN_STATE_CHECK]` | `gdn_linear_attn.py` | Validate `cu_seqlens` (`non_spec_query_start_loc`) consistency before `gdn_attention` kernel (starts at 0, monotone, final entry = `num_actual_tokens`), and validate `ssm_state` slots for NaN/Inf after the kernel |
| `[SSM_TENSOR_CHECK]` | `gdn_linear_attn.py` | One-shot per layer: print `ssm_state.data_ptr()` and `id(ssm_state)` so cross-rank tensor sharing can be detected. If both DP ranks show the same address, they share physical ssm_state memory → NaN from mutual state overwrite. |
| `[SSM_PRE_KERNEL_CHECK]` | `gdn_linear_attn.py` | One-shot per layer on first decode batch: check if `ssm_state` target slots are already NaN/Inf **before** the kernel runs. If `status=NaN_BEFORE_KERNEL`, the corruption occurred upstream (during prefill or a prior decode step). |
| `[BLOCK_TABLE_CHECK]` | `gdn_attn.py` | Every batch: check (1) DUPLICATE\_SLOTS — two requests sharing the same state slot, (2) NULL\_SLOTS\_FOR\_REAL\_REQS — real sequence mapped to NULL slot 0, (3) SLOT\_CHANGED — slot assignment changed unexpectedly between prefill and decode. |

---

## Kernel Call Flow: Qwen3.5 Prefill → First Decode Token

This section documents all compute kernels invoked for a single forward pass,
from the first prefill batch through the first decode step, on the XPU (SYCL)
execution path. The model is **Qwen3.5-35B-A3B** (qwen3_next.py architecture),
configured with TP=2, DP=2, EP=True on 4× Intel ARC B60 GPUs.

### Model Architecture Overview

- **40 decoder layers** total
- **Layer type pattern** (controlled by `full_attention_interval=4`):
  - Layers 0, 1, 2: `linear_attention` (GDN / Gated Delta Network)
  - Layer 3: `full_attention` (Flash Attention)
  - Layers 4, 5, 6: `linear_attention`
  - Layer 7: `full_attention`
  - ... (repeating pattern: 3 GDN + 1 FlashAttn)
- **MLP type**: MoE (Mixture of Experts) for every layer that satisfies
  `(layer_idx + 1) % decoder_sparse_step == 0` when `num_experts > 0`.
  For Qwen3.5-MoE the default is all non-mlp-only layers use MoE.
- **Parallelism**: TP splits Q/K/V/O projections and expert matrices.
  EP distributes expert weights across DP ranks.

### Phase 1: Prefill (prompt processing, T tokens per sequence)

```
Qwen3NextForCausalLM.forward()
└─ Qwen3NextModel.forward()
   │
   ├─ [Embedding]  embed_tokens(input_ids)
   │     Kernel: XPU gather / embedding lookup
   │     Output: hidden_states [T, hidden_size]
   │
   └─ For layer_idx in [0 .. 39]:
      │
      ├─ [RMSNorm]  input_layernorm(hidden_states, residual)
      │     Kernel: fused_rms_norm (XPU element-wise)
      │
      ├─ [IF linear_attention layer (GDN)]
      │   │
      │   ├─ 1. Input Projection (forward_xpu)
      │   │     in_proj_qkvz(hidden_states)  → projected_states_qkvz [T, qkvz_dim]
      │   │     Kernel: ColumnParallelLinear matmul (XPU GEMM, TP-sharded)
      │   │     in_proj_ba(hidden_states)    → projected_states_ba   [T, 2*num_v_heads]
      │   │     Kernel: ReplicatedLinear matmul (XPU GEMM)
      │   │
      │   ├─ 2. Core Attention  — torch.ops._xpu_C.gdn_attention  (SYCL kernel)
      │   │     Inputs:
      │   │       projected_states_qkvz, projected_states_ba
      │   │       conv_state  [num_slots, num_k_heads, head_k_dim, conv_width-1]
      │   │       ssm_state   [num_slots, num_v_heads, head_v_dim, head_k_dim]
      │   │       conv_weights, conv_bias, A_log, dt_bias
      │   │       num_prefills, num_decodes, has_initial_state
      │   │       non_spec_query_start_loc (cu_seqlens), non_spec_state_indices_tensor
      │   │     Operations inside SYCL kernel:
      │   │       a) causal_conv1d (1-D depthwise convolution over qkv)
      │   │       b) gated delta rule recurrent scan (chunked, updates ssm_state)
      │   │     Outputs:
      │   │       core_attn_out [T, num_v_heads, head_v_dim]
      │   │       z             [T, num_v_heads, head_v_dim]   (gate)
      │   │       ssm_state written in-place (persistent KV cache slot)
      │   │       conv_state written in-place (persistent conv cache slot)
      │   │
      │   ├─ 3. Gate-Norm  norm(core_attn_out, z)
      │   │     Kernel: fused gate × RMSNorm (XPU element-wise)
      │   │
      │   └─ 4. Output Projection
      │         out_proj(core_attn_out) → hidden_states [T, hidden_size]
      │         Kernel: RowParallelLinear matmul + TP all_reduce (XPU GEMM + XCCL)
      │
      ├─ [IF full_attention layer (Flash Attention)]
      │   │
      │   ├─ 1. QKV Projection
      │   │     qkv_proj(hidden_states) → qkv [T, (num_heads*(1+gate)+2*kv_heads)*head_dim]
      │   │     Kernel: QKVParallelLinear matmul (XPU GEMM, TP-sharded)
      │   │
      │   ├─ 2. Split gate / q_norm / k_norm
      │   │     q_norm(q), k_norm(k)
      │   │     Kernel: per-head RMSNorm (XPU element-wise)
      │   │
      │   ├─ 3. RoPE  rotary_emb(positions, q, k) → q_rot, k_rot
      │   │     Kernel: rotary position embedding (XPU element-wise)
      │   │
      │   ├─ 4. Flash Attention  attn(q_rot, k_rot, v)
      │   │     Kernel: flash_attn_varlen_func (XPU SYCL flash attention)
      │   │     Writes K, V to paged KV cache (slot_mapping)
      │   │     Reads full prefix KV from paged cache during prefill
      │   │     Output: attn_output [T, num_heads*head_dim]
      │   │
      │   ├─ 5. Attention Output Gate (if enabled)
      │   │     gate = sigmoid(gate_slice)
      │   │     attn_output = attn_output * gate
      │   │     Kernel: XPU element-wise sigmoid + multiply
      │   │
      │   └─ 6. Output Projection
      │         o_proj(attn_output) → hidden_states [T, hidden_size]
      │         Kernel: RowParallelLinear matmul + TP all_reduce (XPU GEMM + XCCL)
      │
      ├─ [RMSNorm]  post_attention_layernorm(hidden_states, residual)
      │     Kernel: fused_rms_norm (XPU element-wise), also fuses residual add
      │
      └─ [MLP / MoE]
          │
          ├─ [IF MoE layer]
          │   ├─ Router  gate(hidden_states) → router_logits [T, num_experts]
          │   │     Kernel: ReplicatedLinear matmul (XPU GEMM)
          │   │
          │   ├─ EP Dispatch  (if enable_expert_parallel)
          │   │     all_gatherv()   — XCCL gather tokens across EP ranks
          │   │     Barrier before collective (Hongbo Fix 6)
          │   │
          │   ├─ FusedMoE  experts(hidden_states, router_logits)
          │   │     Top-K token→expert routing
          │   │     gate_up_proj  matmul per expert  (XPU GEMM, TP-sharded col-wise)
          │   │     SiLU activation
          │   │     down_proj     matmul per expert  (XPU GEMM, TP-sharded row-wise)
          │   │     Shared expert: gate_up_proj → SiLU → down_proj
          │   │
          │   ├─ EP Combine
          │   │     reduce_scatterv() — XCCL scatter-reduce outputs back to home rank
          │   │     Barrier before collective (Hongbo Fix 6)
          │   │
          │   └─ TP all_reduce / all_gather (if TP>1 and not sequence_parallel)
          │
          └─ [IF dense MLP]
                gate_up_proj(hidden_states) [T, 2*intermediate_size]
                Kernel: ColumnParallelLinear matmul (XPU GEMM)
                SiLU+multiply (fused gated activation, XPU element-wise)
                down_proj → hidden_states [T, hidden_size]
                Kernel: RowParallelLinear matmul + TP all_reduce (XPU GEMM + XCCL)

   ├─ [Final RMSNorm]  norm(hidden_states, residual)
   │     Kernel: fused_rms_norm (XPU element-wise)
   │
   └─ LogitsProcessor  lm_head(hidden_states) → logits [T, vocab_size]
         Kernel: VocabParallelEmbedding / ParallelLMHead matmul (XPU GEMM + XCCL)

Sampler  → next_token_ids [batch_size]
```

### Phase 2: First Decode Step (1 token per active sequence)

The decode step repeats the same `Qwen3NextModel.forward()` call, but with
`num_actual_tokens = batch_size` (one new token per sequence).  The key
differences per layer type are:

#### GDN Linear Attention Layer (decode)

On **CUDA**:
```
in_proj_qkvz, in_proj_ba  [same as prefill, but shape [B, dim]]

causal_conv1d_update(mixed_qkv, conv_state, ...)
  → single-step conv, updates conv_state[state_index] in-place

fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv, a, b, A_log, dt_bias,
    initial_state=ssm_state, ssm_state_indices=...) [packed B-sequence update]
  → updates ssm_state[state_index] in-place, outputs core_attn_out [B, 1, v_dim]

norm + out_proj  [same as prefill]
```

On **XPU** (current debug target):
```
in_proj_qkvz, in_proj_ba  [same as prefill]

torch.ops._xpu_C.gdn_attention(
    core_attn_out, z,
    projected_states_qkvz, projected_states_ba,
    ...
    num_prefills=0, num_decodes=B,
    non_spec_query_start_loc=[0, 1, 2, ..., B],
    non_spec_state_indices_tensor=[slot_0, slot_1, ..., slot_{B-1}])
  → same SYCL kernel as prefill, decode mode:
     causal_conv1d single-step update per sequence
     delta rule single-step recurrent update per sequence
     writes conv_state[slot_i], ssm_state[slot_i] in-place

norm + out_proj  [same as prefill]
```

#### Full Attention Layer (decode)

```
qkv_proj, q_norm, k_norm, rotary_emb  [same projections, shape [B, dim]]

flash_attn_varlen_func  (or paged attention kernel)
  Reads all T_i prefix KV from paged cache for each sequence i
  Appends current K, V to paged cache at next slot
  Computes attention over [T_i+1] keys per sequence
  Output: attn_output [B, num_heads*head_dim]

o_proj  [same as prefill]
```

#### MoE / MLP Layer (decode)

Structurally identical to prefill; only the token dimension changes from
`T_total` (prefill) to `B` (decode batch size).

### Key State Tensors (persistent across prefill→decode)

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `conv_state` | `[num_slots, num_k_heads, head_k_dim, conv_width-1]` | Causal conv sliding window state per sequence slot |
| `ssm_state` | `[num_slots, num_v_heads, head_v_dim, head_k_dim]` | GDN recurrent (delta-rule) hidden state per sequence slot |
| KV cache | `[num_blocks, block_size, num_kv_heads, head_dim]` × 2 | Paged K/V cache for full-attention layers |

`num_slots` is determined by `max_model_len // block_size`.  In DP=2, each
DP rank allocates its own `num_slots` pool independently.  Both ranks start
their allocators from the same base index, which means they assign the same
slot IDs to different sequences — a root-cause candidate for shared-state NaN
when `ssm_state` is physically shared across DP ranks (see `[SSM_TENSOR_CHECK]`).

### Kernel Summary Table

| Phase | Layer type | Kernel | Backend |
|-------|-----------|--------|---------|
| Both | All | `rms_norm` (input/post layernorm) | XPU element-wise |
| Both | linear_attn | `in_proj_qkvz` GEMM | XPU GEMM (TP col-split) |
| Both | linear_attn | `in_proj_ba` GEMM | XPU GEMM (replicated) |
| Prefill | linear_attn | `gdn_attention` (causal_conv1d + delta-rule chunk) | XPU SYCL |
| Decode | linear_attn | `gdn_attention` (causal_conv1d_update + delta-rule recurrent) | XPU SYCL |
| Both | linear_attn | gate-norm (RMSNorm × sigmoid gate) | XPU element-wise |
| Both | linear_attn | `out_proj` GEMM + all_reduce | XPU GEMM + XCCL |
| Both | full_attn | `qkv_proj` GEMM | XPU GEMM (TP col-split) |
| Both | full_attn | q_norm, k_norm RMSNorm | XPU element-wise |
| Both | full_attn | RoPE | XPU element-wise |
| Both | full_attn | `flash_attn_varlen_func` (paged KV) | XPU SYCL flash attention |
| Both | full_attn | `o_proj` GEMM + all_reduce | XPU GEMM + XCCL |
| Both | MoE | router `gate` GEMM | XPU GEMM |
| Both | MoE | EP `all_gatherv` / `reduce_scatterv` | XCCL |
| Both | MoE | `FusedMoE` expert GEMMs (gate_up + down per expert) | XPU GEMM |
| Both | dense MLP | `gate_up_proj` GEMM | XPU GEMM (TP col-split) |
| Both | dense MLP | SiLU gated activation | XPU element-wise |
| Both | dense MLP | `down_proj` GEMM + all_reduce | XPU GEMM + XCCL |
| End | — | `lm_head` GEMM | XPU GEMM + XCCL |
