# vLLM Attention 测试用例梳理 (attn_valid.md)

本文档对 vLLM 仓库中与 **Attention** 相关的测试用例进行系统梳理，覆盖 kernel 级单元测试、v1 后端集成测试、后端选择/注册测试、编译融合测试以及端到端测试。文档按目录与功能分组，便于在做 Attention 相关改动时快速定位需要回归的测试集合。

---

## 1. 总览

vLLM 中的 Attention 测试主要分布在以下几个目录：

| 目录 | 关注点 |
| ---- | ------ |
| `tests/kernels/attention/` | Attention 算子 / kernel 级单元测试（FlashAttention、FlashInfer、Triton、FlashMLA、CUTLASS MLA、CPU、ROCm/AITER、KV cache 操作等） |
| `tests/v1/attention/` | v1 引擎下 Attention 后端集成测试（与 reference SDPA 对比的 backend correctness、metadata 构建、batch 重排、splitting 等） |
| `tests/v1/e2e/general/test_cascade_attention.py` | 端到端的 cascade attention 测试 |
| `tests/v1/spec_decode/test_tree_attention.py` | 投机解码下 tree attention 的正确性 |
| `tests/test_attention_backend_registry.py` | Attention backend 注册机制 |
| `tests/kernels/test_flex_attention.py` | FlexAttention 后端 |
| `tests/compile/passes/test_fusion_attn.py`<br>`tests/compile/passes/test_mla_attn_quant_fusion.py`<br>`tests/compile/passes/test_qk_norm_rope_fusion.py`<br>`tests/compile/passes/test_rope_kvcache_fusion.py`<br>`tests/compile/passes/test_fuse_mla_dual_rms_norm.py` | Torch compile 阶段对 Attention/MLA/RoPE/KV-cache 的融合优化 pass |
| `tests/compile/silly_attention.py`、`tests/compile/fullgraph/...` | full-graph 编译路径中使用的 attention stub 与端到端编译测试 |
| `tests/kernels/core/test_vit_fp8_attn.py`、`tests/models/multimodal/generation/test_vit_backend_functionality.py` | ViT (encoder) 多模态 Attention 后端 |

---

## 2. Kernel 级 Attention 测试 (`tests/kernels/attention/`)

### 2.1 PagedAttention v1 / v2 — `test_attention.py`
- `test_paged_attention`：对照 `ref_masked_attention` 校验自定义 PagedAttention v1/v2 在解码阶段的输出。
  - 主要参数化：`NUM_GEN_SEQS=[7]`、`NUM_HEADS=[(40,40),(64,8)]`（含 GQA）、`HEAD_SIZES=[32,80,128,256]`、`USE_ALIBI=[False,True]`、`BLOCK_SIZES=[16,32]`、`DTYPES=[bf16]`、`KV_CACHE_DTYPE=["auto","fp8"]`、多 CUDA 设备。
- `test_num_heads_not_divisible_by_num_kv_heads`：`Attention` / `MMEncoderAttention` 在 `num_heads` 不是 `num_kv_heads` 整数倍时应抛出错误。

### 2.2 FlashAttention 系列
- `test_flash_attn.py::test_varlen_with_paged_kv`：FA varlen + paged KV，对比 `ref_paged_attn`。
  - 参数化：`use_out`、`NUM_HEADS=[(4,4),(8,2)]`、`HEAD_SIZES=[40,72,80,128,256]`、`BLOCK_SIZES=[16]`、`SLIDING_WINDOWS=[None,256]`、`SOFT_CAPS=[None]`、`fa_version=[2,3]`、`q_dtype∈{None, fp8_e4m3}`、`NUM_BLOCKS∈{32768, 2048}`（2048 是为了触发索引溢出）。
- `test_cascade_flash_attn.py`
  - `test_merge_kernel`：`merge_attn_states` 的 Triton kernel。`num_tokens∈{1,39,16912}`。
  - `test_cascade`：cascade attention 与无 cascade 路径在共享前缀下的等价性，覆盖 `fa_version∈{2,3}`、`soft_cap∈{None,50}` 等。
- `test_aiter_flash_attn.py::test_varlen_with_paged_kv`：ROCm AITER FA 版本的 varlen + paged KV，参数与 FA 版本类似（含 `q_dtype` fp8）。

### 2.3 FlashInfer 系列
- `test_flashinfer.py`
  - `test_fast_decode_plan_importable`：检查 fast plan API 可导入。
  - `test_fast_plan_decode_warmup_uses_full_plan`：warmup 阶段必须落到 full plan。
  - `test_fast_plan_decode_matches_full_plan`：fast plan 与 full plan 输出一致。
  - `test_flashinfer_decode_with_paged_kv`：解码 + paged KV，对比 ref。
  - `test_flashinfer_prefill_with_paged_kv`：prefill + paged KV。
  - 共同参数：`NUM_HEADS=[(32,8),(6,1)]`、`HEAD_SIZES=[128,256]`、`BLOCK_SIZES=[16,32]`、`SOFT_CAPS=[None,30.0]`、`SLIDING_WINDOWS=[None,64]`。
- `test_flashinfer_mla_decode.py::test_flashinfer_mla_decode`：FlashInfer MLA decode（`bs∈{1,2,4,16}`，`block_size∈{32,64}`）。
- `test_flashinfer_trtllm_attention.py`
  - `test_flashinfer_trtllm_decode_with_baseline`
  - `test_flashinfer_trtllm_prefill_with_baseline`
  - 参数包含 `quant_dtypes`（含 NVFP4/FP8）、`kv_layout`、`window_left`、`soft_cap`、`has_sinks`，覆盖 SM100 上的 TRT-LLM 路径。
- `test_use_trtllm_attention.py`：`use_trtllm_attention` 决策函数的纯单元测试。
  - 平台/版本/heads/批大小/spec-decode/FP8 query/sinks/auto KV 等多分支。代表性用例：
    - `test_supports_sm100_with_artifactory` / `test_supports_non_sm100_platform` / `test_supports_sm100_without_artifactory`
    - `test_can_use_*`、`test_use_force_off`、`test_use_dcp_fallback`、`test_use_platform_unsupported`
    - `test_use_incompatible_heads(_force_on_still_false)`、`test_use_spec_decode_enables`
    - `test_use_fp8_query_forces_trtllm`、`test_use_sinks_force_trtllm`
    - `test_use_auto_prefill_kv_auto/fp8`、`test_use_auto_decode_small_batch`
    - `test_supports_batch_invariant_disables`
- `test_trtllm_kvfp8_dequant.py`：TRT-LLM KV FP8 dequant kernel。涵盖：
  - `test_trtllm_kvfp8_dequant`、`test_block_tables_with_zero_pages`、`test_all_zero_block_tables`、`test_different_k_v_scales`、`test_single_page_per_seq`、`test_large_page_indices`、`test_large_block_size`、`test_cross_layer_many_layers`。

### 2.4 FlashMLA / CUTLASS MLA / Sparse MLA
- `test_flashmla.py::test_flash_mla`：FlashMLA decode kernel（`s_q∈{1,2}`, `mean_sk∈{4096,8192,16384}`, `h_q∈{16,32,64,128}`, `varlen∈{False,True}`）。
- `test_cutlass_mla_decode.py::test_cutlass_mla_decode`：CUTLASS MLA decode（与 FlashMLA 类似的形状空间，`d=576, dv=512`）。
- `test_flashmla_sparse.py`：稀疏 MLA 三个 smoke 测试 — `metadata`、`decode`、`prefill`。
- `test_xpu_mla_sparse.py::test_bf16_triton_sparse_mla`：XPU 上 Triton 稀疏 MLA。
- `test_mla_decode_cpu.py::test_mla_decode_cpu`：CPU MLA decode 路径。

### 2.5 Triton Attention kernels
- `test_triton_decode_attention.py`
  - `test_decode_attention`：Triton decode（B、L、`H_Q`、`H_KV`、`D_QK`、`D_V`、`CACHE_SIZE`、`PAGE_SIZE` 多组参数）。
  - `test_decode_attention_fp8`：FP8 KV cache 版本。
- `test_triton_prefill_attention.py`
  - `test_context_attention`、`test_context_attention_sliding_window`。
- `test_triton_unified_attention.py`
  - `test_triton_unified_attn`：统一 prefill+decode kernel 与 ref 对比。
  - `test_triton_unified_attn_fp16_input_fp8_output`：fp16 输入 / fp8 输出路径。
- `test_prefix_prefill.py`：分块 prefill / context attention。
  - `test_contexted_kv_attention` 及其 `_alibi`、`_f32`、`_alibi_f32` 变体。
  - `test_qwen3_nonstandard_block_size`：覆盖 Qwen3 非标准 block size。
  - `OPS=[chunked_prefill_paged_decode, context_attention_fwd]`，`SLIDING_WINDOW=[0,16,2048]`，`KV_CACHE_DTYPES=["auto","fp8","fp8_e5m2"]`。
- `test_merge_attn_states.py::test_merge_attn_states`：合并多段 attention 状态的 CUDA/Triton 实现一致性，含 `use_fp8` 与 prefill+context 路径。
- `test_pack_unpack_triton.py`：序列 pack/unpack Triton kernel 的 FP8 单元测试（基本/自定义 padding/默认 -inf padding/边界/不同 block size/round-trip 等）。

### 2.6 Lightning Attention
- `test_lightning_attn.py`
  - `test_linear_decode_forward_triton`、`test_linear_decode_forward_triton_with_padding`、`test_lightning_attention_reference`。

### 2.7 KV Cache 操作
- `test_cache.py`
  - `test_reshape_and_cache`、`test_reshape_and_cache_flash`（含 `kv_cache_dtype` 扩展到 `nvfp4`、多 layout、多 implementation、多 scale 类型）。
  - `test_swap_blocks`、`test_copy_blocks`（通过 `direction`、`num_mappings` 等参数化）。
  - `test_concat_and_cache_mla`、`test_concat_and_cache_ds_mla`、`test_indexer_k_quant_and_cache` 等 MLA / DeepSeek indexer 相关 KV 写入。
  - `test_cp_gather_*`、`test_cp_gather_indexer_*`（在仓库内还有 `tests/kernels/test_cp_gather_fp8.py` 等关联测试）。

### 2.8 Attention Backend 选择器
- `test_attention_selector.py`
  - `test_backend_selection`（参数化 device/name/use_mla/block_size 全集合）。
  - `test_fp32_fallback`：FP32 应回退到默认 backend。
  - `test_flash_attn`：FA 默认/版本相关分支。
  - `test_invalid_backend`、`test_auto_backend_string`、`test_auto_backend_selection_behavior`。
  - `test_per_head_quant_scales_backend_selection`：per-head quant scales 限制下的选择。
  - `test_non_causal_backend_selection` / `test_non_causal_autoselect_backend`。
- `test_rocm_attention_selector.py::test_selector`：ROCm 平台下 backend 选择。
- `test_mha_attn.py`：原生 MHA 层（vLLM `MultiHeadAttention`）。
  - `test_mha_attn_platform`：平台 → backend 选择。
  - `test_mha_attn_forward`、`test_mha_attn_varlen_forward`、`test_mha_attn_varlen_forward_flashinfer`。

### 2.9 DeepGemm Attention
- `test_deepgemm_attention.py`
  - `test_deepgemm_fp8_mqa_logits`（`clean_logits∈{True,False}`）。
  - `test_deepgemm_fp8_fp4_paged_mqa_logits`。

### 2.10 CPU Attention
- `test_cpu_attn.py`：CPU varlen + paged KV（vec、AMX、vec16、sliding window、ALiBi、sink、FP8 KV cache 等）。
  - `test_varlen_with_paged_kv_normal_vec`
  - `test_varlen_with_paged_kv_normal_amx`
  - `test_varlen_with_paged_kv_vec16`
  - 以及对应的 `_sliding_window`、`_alibi`、`_sink`、`_fp8_*` 等变体（按文件中定义的 isa / dtype / sliding_window / soft_cap / use_alibi / use_sink 参数组合矩阵展开）。

---

## 3. v1 引擎 Attention 后端测试 (`tests/v1/attention/`)

### 3.1 标准 Attention Backend 正确性 — `test_attention_backends.py`
- 测试集合 `BACKENDS_TO_TEST = {FLASH_ATTN, FLASHINFER, FLEX_ATTENTION, TRITON_ATTN, TREE_ATTN, "FLEX_ATTENTION_SLOW"}`（FlashInfer 不可用时自动剔除）。
- 用例：
  - `test_causal_backend_correctness`
  - `test_sliding_window_backend_correctness`
  - `test_sliding_window_encoder_backend_correctness`
  - `test_non_causal_backend_correctness`
- 通过 `BatchSpec`/`create_common_attn_metadata` 生成多种批组合（small/large prefill、decode、mixed），与 SDPA 参考实现对比。

### 3.2 MLA Backend 正确性 — `test_mla_backends.py`
- `BACKENDS_TO_TEST = {CUTLASS_MLA, FLASHMLA, FLASH_ATTN_MLA, FLASHINFER_MLA, TRITON_MLA}`（按硬件能力裁剪）。
- `test_backend_correctness`：覆盖各 MLA backend 在 prefill / decode / mixed batch 下与参考的等价性。

### 3.3 Sparse MLA Backend — `test_sparse_mla_backends.py`
- `test_sparse_backend_decode_correctness`
- `test_triton_convert_req_index_to_global_index_decode_only`
- `test_triton_convert_req_index_to_global_index_with_prefill_workspace`
- `test_split_prefill_chunks`、`test_split_indexer_prefill_chunks`、`test_split_indexer_prefill_chunks_single_request_overflow`
- `test_triton_convert_returns_valid_counts`

### 3.4 后端选择 (v1)
- `test_attention_backends_selection.py`
  - `test_mamba_layers_get_attn_backend`
  - `test_mamba_layers_have_unified_interface`
- `test_rocm_attention_backends_selection.py`
  - `test_standard_attention_backend_selection`
  - `test_mla_backend_selection`
  - `test_aiter_fa_requires_mi3xx`
  - `test_sparse_not_supported`
- `test_mla_prefill_selector.py`：包含若干测试类
  - `TestGetMLAPrefillBackend`
  - `TestAutoSelectMLAPrefillBackend`
  - `TestBackendValidation`
  - `TestMLAPrefillBackendParsing`
  - `TestDeprecatedFlagMigration`

### 3.5 Metadata / Batch / Splitting
- `test_attention_splitting.py`：`query_start_loc` 切片、`split_decodes_and_prefills`（uniform / non-uniform / 全 decode / 全 prefill / 混合 / padded）、`split_attn_metadata`、`prefill_split_across_ubatches` 等。
- `test_batch_reordering.py::test_reorder_batch_to_split_decodes_and_prefills`：基于 `ReorderTestCase` 的 batch 重排。
- `test_chunked_local_attention.py::test_local_attention_virtual_batches`：chunked local attention 的虚拟 batch 切分。
- `test_kv_head_stride_canonicalization.py::TestCanonicalizeSingletonDimStrides`：KV head stride 的 canonical 化。
- `test_gdn_metadata_builder.py`
  - `test_gdn_build_classification`
  - `test_has_initial_state_after_reclassification`
- `test_indexer_deepseek_v4_slot_mapping.py::test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size`
- `test_mamba_update_block_table.py::test_update_block_table_copies_block_idx_to_persistent_buffers`

### 3.6 TRT-LLM 集成 — `test_trtllm_attention_integration.py`
- `test_trtllm_gen_full_attention_integration`
- `test_trtllm_gen_nvfp4_kv_integration`
- 通过 `MockAttentionLayer` + 多 `BatchSpec` 验证 TRT-LLM gen 路径的端到端集成。

---

## 4. Attention Backend 注册 — `tests/test_attention_backend_registry.py`
- `test_custom_is_not_alias_of_any_backend`
- `test_register_custom_backend_with_class_path`
- `test_mamba_custom_is_not_alias_of_any_backend`
- `test_register_custom_mamba_backend_with_class_path`

确认外部插件可通过 class path 注册自定义 attention/mamba backend，且 `CUSTOM` 不与现有 backend 别名冲突。

---

## 5. Flex Attention — `tests/kernels/test_flex_attention.py`
- `test_flex_attention_full_cudagraphs`：FlexAttention 后端在 full CUDA graph 下端到端 OK。
- `test_flex_attention_vs_default_backend`：与默认 backend 输出对齐。
- `test_encoder_flex_attention_vs_default_backend`：encoder 路径对齐。
- `test_block_mask_direct_vs_slow_path`：block mask 直接路径 vs slow path。
- `test_physical_to_logical_mapping_handles_reused_blocks`：复用块的物理→逻辑映射。
- `test_block_sparsity_hint_prunes_blocks`：稀疏 hint 应正确裁剪 block。

---

## 6. Compile / Fusion Pass 中的 Attention

| 文件 | 主要测试 | 说明 |
| ---- | -------- | ---- |
| `tests/compile/passes/test_fusion_attn.py` | `test_attention_quant_pattern` | Attention + quant 融合 pass |
| `tests/compile/passes/test_mla_attn_quant_fusion.py` | `test_mla_attention_quant_pattern` | MLA + quant 融合 |
| `tests/compile/passes/test_qk_norm_rope_fusion.py` | QK-Norm + RoPE 融合 | 与 attention 紧邻的算子融合 |
| `tests/compile/passes/test_rope_kvcache_fusion.py` | RoPE + KV cache 写入融合 | 与 attention metadata 协作 |
| `tests/compile/passes/test_fuse_mla_dual_rms_norm.py` | MLA dual RMSNorm 融合 | MLA prefill 路径融合 |
| `tests/compile/silly_attention.py` + `tests/compile/fullgraph/*` | full-graph 编译路径中的 silly attention stub 与端到端编译测试 | 验证编译/cudagraph 与 attention 的兼容性 |

---

## 7. 端到端 / Spec-Decode 中的 Attention
- `tests/v1/e2e/general/test_cascade_attention.py::test_cascade_attention`
  - 在 v1 引擎上端到端测试 cascade attention，参数化 `attn_backend`，与共享 system message 的多并发请求场景配合。
- `tests/v1/e2e/general/test_correctness_sliding_window.py`：sliding window attention 端到端正确性。
- `tests/v1/spec_decode/test_tree_attention.py::test_tree_attn_correctness`
  - 投机解码下 tree attention（多分支 token 树）的输出正确性。
- `tests/v1/cudagraph/test_cudagraph_dispatch.py`、`tests/v1/cudagraph/test_cudagraph_mode.py`：与 attention backend 协作的 CUDA Graph 调度。
- `tests/distributed/test_context_parallel.py`、`tests/distributed/test_dcp_a2a.py`：长上下文并行（Context Parallel / DCP）下 attention 行为。

---

## 8. 多模态 / 编码器 Attention
- `tests/kernels/core/test_vit_fp8_attn.py`、`tests/kernels/core/test_vit_fp8_scaling.py`：ViT 中的 FP8 attention/scaling kernel。
- `tests/models/multimodal/generation/test_vit_backend_functionality.py`：ViT attention backend 功能性。
- `tests/kernels/attention/test_mha_attn.py::test_mha_attn_*`：encoder-style `MultiHeadAttention`（含 FlashInfer 路径）。

---

## 9. 测试运行建议（按改动范围）

| 改动范围 | 建议执行的测试 |
| -------- | -------------- |
| 修改 PagedAttention / KV cache | `tests/kernels/attention/test_attention.py`, `test_cache.py`, `test_merge_attn_states.py` |
| 修改 FlashAttention / 引入新 FA 版本 | `test_flash_attn.py`, `test_cascade_flash_attn.py`, `test_aiter_flash_attn.py`, `tests/v1/attention/test_attention_backends.py` |
| 修改 FlashInfer 集成 | `test_flashinfer*.py`, `test_use_trtllm_attention.py`, `test_trtllm_kvfp8_dequant.py`, `tests/v1/attention/test_trtllm_attention_integration.py` |
| 修改 MLA / Sparse MLA | `test_flashmla*.py`, `test_cutlass_mla_decode.py`, `test_xpu_mla_sparse.py`, `test_mla_decode_cpu.py`, `tests/v1/attention/test_mla_backends.py`, `test_sparse_mla_backends.py`, `test_mla_prefill_selector.py` |
| 修改 Triton attention kernels | `test_triton_decode_attention.py`, `test_triton_prefill_attention.py`, `test_triton_unified_attention.py`, `test_prefix_prefill.py`, `test_pack_unpack_triton.py` |
| 修改 backend 选择 / 注册 | `test_attention_selector.py`, `test_rocm_attention_selector.py`, `test_attention_backend_registry.py`, `tests/v1/attention/test_attention_backends_selection.py`, `test_rocm_attention_backends_selection.py` |
| 修改 v1 metadata / batch 切分 | `tests/v1/attention/test_attention_splitting.py`, `test_batch_reordering.py`, `test_chunked_local_attention.py`, `test_kv_head_stride_canonicalization.py`, `test_gdn_metadata_builder.py` |
| 修改 compile / fusion | `tests/compile/passes/test_fusion_attn.py`, `test_mla_attn_quant_fusion.py`, `test_qk_norm_rope_fusion.py`, `test_rope_kvcache_fusion.py`, `test_fuse_mla_dual_rms_norm.py` |
| 修改端到端调度 / cascade / spec decode | `tests/v1/e2e/general/test_cascade_attention.py`, `tests/v1/e2e/general/test_correctness_sliding_window.py`, `tests/v1/spec_decode/test_tree_attention.py` |
| 修改 CPU / XPU / ROCm 路径 | `test_cpu_attn.py`, `test_mla_decode_cpu.py`, `test_xpu_mla_sparse.py`, `test_aiter_flash_attn.py`, `test_rocm_attention_selector.py`, `tests/v1/attention/test_rocm_attention_backends_selection.py` |

---

## 10. 如何运行这些用例

### 10.1 环境准备

vLLM 使用 `pytest` 驱动所有测试，运行 attention 测试前需要先按官方文档安装测试依赖（参见 `docs/contributing/README.md`）：

```bash
# 1) 安装与 CI 一致的依赖（CUDA 环境）
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# 2) 安装 tests/conftest.py 所需的测试依赖（按硬件平台选择其一）
#    —— 这一步非常关键。tests/conftest.py 第 7 行 `from tblib import pickling_support`
#       会要求 tblib 等测试依赖；缺失时整个 tests/ 目录都会以
#       `ImportError while loading conftest '.../tests/conftest.py'` 失败而无法运行。
uv pip install -r requirements/test/cuda.txt   # NVIDIA GPU
# 或：
# uv pip install -r requirements/test/rocm.txt # AMD ROCm
# uv pip install -r requirements/test/xpu.txt  # Intel XPU

# 3) 通用测试依赖（一般已被 dev.txt / test/*.txt 拉入，这里兜底）
uv pip install pytest pytest-asyncio tblib

# 4) 以 editable 模式安装 vLLM 本身（如果还没装），保证 import 的是当前仓库代码
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

> **常见报错**：`ImportError while loading conftest '.../tests/conftest.py'` / `tests/conftest.py:7: in <module> from tblib import pickling_support` 表示当前环境没装 `tblib`（以及多半也没装其它测试依赖）。`tests/conftest.py` 是 pytest 的全局 fixture 文件，跑 `tests/` 下任意用例都会先加载它，所以缺依赖时**所有** attention 测试都会立刻失败、连 collection 都进不去。修复方法就是补跑上面第 2 / 3 步。

可选依赖按需安装（缺失时对应文件会通过 `pytest.skip(..., allow_module_level=True)` 整体跳过）：

| 测试 | 需要的额外依赖 |
| ---- | -------------- |
| `test_flashinfer*.py`、`test_use_trtllm_attention.py`、`test_trtllm_kvfp8_dequant.py`、`test_flashinfer_trtllm_attention.py` | `flashinfer-python`（且通常需要 SM100 / Hopper+ GPU） |
| `test_aiter_flash_attn.py`、`test_rocm_attention_selector.py`、`tests/v1/attention/test_rocm_attention_backends_selection.py` | ROCm + `aiter` |
| `test_flashmla*.py` | FlashMLA（DeepSeek） + Hopper |
| `test_cutlass_mla_decode.py`、`tests/v1/attention/test_mla_backends.py`（CUTLASS_MLA / FLASHINFER_MLA 分支） | SM100（Blackwell） |
| `test_deepgemm_attention.py` | `deep_gemm` + Hopper |
| `test_cpu_attn.py`、`test_mla_decode_cpu.py` | CPU build（部分用例需要 AMX / VEC ISA） |
| `test_xpu_mla_sparse.py` | XPU build |
| `tests/kernels/test_flex_attention.py`、`tests/v1/attention/test_attention_backends.py` 中的 FlexAttention 分支 | PyTorch ≥ 2.5（`torch.nn.attention.flex_attention`） |

### 10.2 常用运行方式

下面所有命令都假设在仓库根目录 `/.../vllm/` 下执行。

```bash
# 1) 运行整个 kernel 级 attention 测试目录
pytest -s -v tests/kernels/attention/

# 2) 运行整个 v1 引擎 attention 测试目录
pytest -s -v tests/v1/attention/

# 3) 同时跑两组（最常用的一次性回归命令）
pytest -s -v tests/kernels/attention/ tests/v1/attention/

# 4) 只跑某个文件
pytest -s -v tests/kernels/attention/test_flash_attn.py
pytest -s -v tests/v1/attention/test_attention_backends.py

# 5) 只跑文件里的某个用例
pytest -s -v tests/kernels/attention/test_attention.py::test_paged_attention

# 6) 用 -k 过滤名字（适合 parametrize 出来的大量子用例）
pytest -s -v tests/kernels/attention/test_flash_attn.py -k "head_size128 and sliding_window-256 and fa_version2"
pytest -s -v tests/v1/attention/test_mla_backends.py -k "FLASHMLA and small_decode"

# 7) 列出所有子用例但不执行（确认参数化展开）
pytest --collect-only -q tests/kernels/attention/test_flash_attn.py

# 8) 失败时打印更多信息 / 失败即停 / 显示最慢的 N 个
pytest -s -v --tb=long -x --durations=20 tests/kernels/attention/

# 9) 并行加速（需要 pytest-xdist）
uv pip install pytest-xdist
pytest -n 8 tests/kernels/attention/
```

### 10.3 运行 backend 选择 / 注册类测试

这些测试主要是 mock + 单元逻辑，对硬件依赖较弱：

```bash
pytest -s -v tests/kernels/attention/test_attention_selector.py
pytest -s -v tests/kernels/attention/test_use_trtllm_attention.py
pytest -s -v tests/test_attention_backend_registry.py
pytest -s -v tests/v1/attention/test_attention_backends_selection.py
pytest -s -v tests/v1/attention/test_mla_prefill_selector.py
```

### 10.4 运行 compile / fusion pass 中的 attention 相关测试

```bash
pytest -s -v \
    tests/compile/passes/test_fusion_attn.py \
    tests/compile/passes/test_mla_attn_quant_fusion.py \
    tests/compile/passes/test_qk_norm_rope_fusion.py \
    tests/compile/passes/test_rope_kvcache_fusion.py \
    tests/compile/passes/test_fuse_mla_dual_rms_norm.py
```

### 10.5 运行端到端 / spec-decode 中的 attention 测试

这类测试会真正起 vLLM engine 加载小模型，**需要 GPU 且会下载 HF 模型**，运行时间相对较长：

```bash
# cascade attention 端到端
pytest -s -v tests/v1/e2e/general/test_cascade_attention.py

# sliding window 正确性
pytest -s -v tests/v1/e2e/general/test_correctness_sliding_window.py

# tree attention（投机解码）
pytest -s -v tests/v1/spec_decode/test_tree_attention.py
```

可以通过 `HF_HUB_OFFLINE=1` + 预下载模型避免网络拉取，或通过 `VLLM_TEST_MODEL=...` 等环境变量切换为本地小模型（视具体测试而定）。

### 10.6 常用环境变量与 backend 切换

很多 attention 测试会读取环境变量来选择后端 / 验证 fallback，可在命令前显式覆盖：

```bash
# 强制选择某个 backend（部分测试会自己设置，不需要手动指定）
VLLM_ATTENTION_BACKEND=FLASH_ATTN  pytest -s -v tests/v1/attention/test_attention_backends.py
VLLM_ATTENTION_BACKEND=FLASHINFER  pytest -s -v tests/v1/attention/test_attention_backends.py
VLLM_ATTENTION_BACKEND=TRITON_ATTN pytest -s -v tests/v1/attention/test_attention_backends.py
VLLM_ATTENTION_BACKEND=FLEX_ATTENTION pytest -s -v tests/kernels/test_flex_attention.py

# 限制使用的 GPU
CUDA_VISIBLE_DEVICES=0 pytest -s -v tests/kernels/attention/test_attention.py

# 跳过会下载模型的测试（按需结合 -k / -m 使用）
HF_HUB_OFFLINE=1 pytest -s -v tests/kernels/attention/
```

### 10.7 一些快速 smoke 用例

只想最快地确认 attention 路径没坏，可以挑这些“小而全”的子集：

```bash
# 最小集：PagedAttention + FA varlen + v1 standard backend correctness 抽样
pytest -s -v \
    tests/kernels/attention/test_attention.py::test_paged_attention \
    tests/kernels/attention/test_flash_attn.py::test_varlen_with_paged_kv \
    tests/v1/attention/test_attention_backends.py::test_causal_backend_correctness \
    -k "head_size128 and block_size16"

# MLA smoke
pytest -s -v \
    tests/kernels/attention/test_flashmla.py::test_flash_mla \
    tests/v1/attention/test_mla_backends.py::test_backend_correctness \
    -k "small_decode"

# Selector / 注册（CPU 即可跑）
pytest -s -v \
    tests/kernels/attention/test_attention_selector.py \
    tests/test_attention_backend_registry.py
```

> 提示：由于绝大多数文件都用 `pytest.mark.parametrize` 形成笛卡尔积，**全量执行非常耗时**。日常开发推荐先用 `-k` 锁定一个小参数子集快速回归，再在提交前跑完整目录。

---

## 11. 附录 A：文件 ↔ 测试函数完整对照

下面把每个 attention 测试文件的所有 `test_*` 函数（含 class 内的方法）逐一列出，并给出每条用例**实际验证的功能**，方便定位「改了 X，到底要看哪个文件里的哪个用例」。

### A.1 `tests/kernels/attention/`（kernel 级）

#### A.1.1 `test_attention.py` — PagedAttention v1 / v2
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_paged_attention` | PagedAttention v1/v2 解码 kernel 输出 vs `ref_masked_attention`，覆盖 GQA、ALiBi、`block_size∈{16,32}`、`head_size∈{32,80,128,256}`、`kv_cache_dtype∈{auto, fp8}`、多 CUDA 设备 |
| `test_num_heads_not_divisible_by_num_kv_heads` | `Attention` / `MMEncoderAttention` 在 `num_heads` 不被 `num_kv_heads` 整除时必须抛错 |

#### A.1.2 `test_flash_attn.py` — FlashAttention varlen
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_varlen_with_paged_kv` | FA2/FA3 varlen + paged KV，对比 `ref_paged_attn`；含 `use_out`、sliding window、`q_dtype∈{None, fp8_e4m3}`、`NUM_BLOCKS=2048` 触发 32-bit 索引溢出路径 |

#### A.1.3 `test_cascade_flash_attn.py` — Cascade FA
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_merge_kernel` | `merge_attn_states` Triton kernel（`num_tokens∈{1, 39, 16912}`） |
| `test_cascade` | cascade attention（共享前缀拆两段）与单段 FA 在相同输入下输出一致，覆盖 FA2/FA3、`soft_cap` |

#### A.1.4 `test_aiter_flash_attn.py` — ROCm AITER FA
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_varlen_with_paged_kv` | ROCm AITER FlashAttention varlen + paged KV vs reference，含 fp8 `q_dtype` |

#### A.1.5 `test_flashinfer.py` — FlashInfer decode/prefill
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_fast_decode_plan_importable` | FlashInfer fast plan API 能正常 import |
| `test_fast_plan_decode_warmup_uses_full_plan` | warmup 阶段必须落到 full plan 而不是 fast plan |
| `test_fast_plan_decode_matches_full_plan` | fast plan 与 full plan 输出数值一致 |
| `test_flashinfer_decode_with_paged_kv` | FlashInfer decode + paged KV vs `ref_paged_attn` |
| `test_flashinfer_prefill_with_paged_kv` | FlashInfer prefill + paged KV vs reference |
| `test_flashinfer_prefill_with_paged_fp8_kv` | FlashInfer prefill 在 FP8 paged KV cache 下的正确性 |
| `test_flashinfer_decode_with_paged_fp8_kv` | FlashInfer decode 在 FP8 paged KV cache 下的正确性 |

#### A.1.6 `test_flashinfer_mla_decode.py`
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_flashinfer_mla_decode` | FlashInfer MLA decode kernel（`bs∈{1,2,4,16}`, `block_size∈{32,64}`） vs `ref_mla` |

#### A.1.7 `test_flashinfer_trtllm_attention.py` — TRT-LLM gen
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_flashinfer_trtllm_decode_with_baseline` | TRT-LLM gen decode（含 NVFP4/FP8 量化、`kv_layout`、`window_left`、`soft_cap`、`has_sinks`）对比 baseline FlashInfer |
| `test_flashinfer_trtllm_prefill_with_baseline` | TRT-LLM gen prefill 同上 |

#### A.1.8 `test_use_trtllm_attention.py` — `use_trtllm_attention` 决策矩阵（纯单测）
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_supports_batch_invariant_disables` | batch-invariant 模式下禁用 TRT-LLM |
| `test_supports_sm100_with_artifactory` / `test_supports_sm100_without_artifactory` / `test_supports_non_sm100_platform` | 平台 + artifactory 组合下 `supports` 返回值 |
| `test_can_use_force_disabled` / `test_can_use_compatible_heads` / `test_can_use_incompatible_heads` / `test_can_use_platform_unsupported` | `can_use` 在 force / heads / 平台维度的判定 |
| `test_use_force_off` / `test_use_force_on` | `VLLM_USE_TRTLLM_ATTENTION=0/1` 强制开关 |
| `test_use_dcp_fallback` | DCP（context-parallel）下回退非-TRT-LLM |
| `test_use_platform_unsupported` / `test_use_platform_unsupported_force_on_still_false` | 不支持平台即使 force on 也不启用 |
| `test_use_incompatible_heads` / `test_use_incompatible_heads_force_on_still_false` | head 维度不兼容时即使 force on 也不启用 |
| `test_use_spec_decode_enables` | spec-decode 路径强制启用 TRT-LLM |
| `test_use_fp8_query_forces_trtllm` | FP8 query → 强制 TRT-LLM |
| `test_use_sinks_force_trtllm` | 带 attention sinks → 强制 TRT-LLM |
| `test_use_auto_prefill_kv_auto` / `test_use_auto_prefill_kv_fp8` | auto 模式下 prefill 路径根据 KV dtype 决策 |
| `test_use_auto_decode_small_batch` / `test_use_auto_decode_large_batch` | auto 模式下 decode 路径按批大小决策 |

#### A.1.9 `test_trtllm_kvfp8_dequant.py` — TRT-LLM KV FP8 dequant
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_trtllm_kvfp8_dequant` | 主路径：dequant 输出对比 `ref_dequant` |
| `test_block_tables_with_zero_pages` | block_table 含 padding (0) 页时不读越界 |
| `test_all_zero_block_tables` | 全 0 block_table 边界 |
| `test_different_k_v_scales` | K/V 使用不同 scale |
| `test_single_page_per_seq` | 每 seq 仅 1 个 page 的极端形状 |
| `test_large_page_indices` | 超大 page index 不溢出 |
| `test_large_block_size` | 超大 `block_size` |
| `test_cross_layer_many_layers` | 跨多层 KV cache 的 dequant |

#### A.1.10 `test_flashmla.py` / `test_cutlass_mla_decode.py` / `test_flashmla_sparse.py` / `test_xpu_mla_sparse.py` / `test_mla_decode_cpu.py`
| 文件::函数 | 验证功能 |
| ---------- | -------- |
| `test_flashmla.py::test_flash_mla` | FlashMLA decode kernel（`s_q∈{1,2}`, `mean_sk∈{4096,8192,16384}`, `h_q∈{16,32,64,128}`, `varlen`） |
| `test_cutlass_mla_decode.py::test_cutlass_mla_decode` | SM100 CUTLASS MLA decode（`d=576, dv=512`） |
| `test_flashmla_sparse.py::test_sparse_flashmla_metadata_smoke` | Sparse FlashMLA metadata 构建 smoke |
| `test_flashmla_sparse.py::test_sparse_flashmla_decode_smoke` | Sparse FlashMLA decode smoke |
| `test_flashmla_sparse.py::test_sparse_flashmla_prefill_smoke` | Sparse FlashMLA prefill smoke |
| `test_xpu_mla_sparse.py::test_bf16_triton_sparse_mla` | XPU 上 Triton 稀疏 MLA prefill vs `reference_mla_sparse_prefill` |
| `test_mla_decode_cpu.py::test_mla_decode_cpu` | CPU MLA decode vs `ref_mla` |

#### A.1.11 Triton kernels
| 文件::函数 | 验证功能 |
| ---------- | -------- |
| `test_triton_decode_attention.py::test_decode_attention` | Triton decode（B/L/H_Q/H_KV/D_QK/D_V/CACHE/PAGE 多组合） |
| `test_triton_decode_attention.py::test_decode_attention_fp8` | 上者的 FP8 KV cache 版本 |
| `test_triton_prefill_attention.py::test_context_attention` | Triton context (prefill) attention vs `ref_masked_attention` |
| `test_triton_prefill_attention.py::test_context_attention_sliding_window` | 加上 sliding window mask |
| `test_triton_unified_attention.py::test_triton_unified_attn` | 统一 prefill+decode kernel vs `ref_paged_attn` |
| `test_triton_unified_attention.py::test_triton_unified_attn_fp16_input_fp8_output` | fp16 输入 + fp8 输出路径 |
| `test_prefix_prefill.py::test_contexted_kv_attention` | chunked prefill / context attention，`OPS=[chunked_prefill_paged_decode, context_attention_fwd]`，sliding_window、`kv_cache_dtype∈{auto, fp8, fp8_e5m2}` |
| `test_prefix_prefill.py::test_contexted_kv_attention_alibi` | 同上 + ALiBi |
| `test_prefix_prefill.py::test_contexted_kv_attention_f32` | f32 数值 baseline |
| `test_prefix_prefill.py::test_contexted_kv_attention_alibi_f32` | f32 + ALiBi |
| `test_prefix_prefill.py::test_qwen3_nonstandard_block_size` | Qwen3 非标准 block size 兼容性 |
| `test_merge_attn_states.py::test_merge_attn_states` | CUDA 与 Triton `merge_attn_states` 一致性，含 `use_fp8` 与 prefill+context |
| `test_pack_unpack_triton.py::test_pack_seq_basic_fp8` | FP8 序列 pack 基础 |
| `test_pack_unpack_triton.py::test_pack_seq_custom_padding_fp8` | 自定义 padding 值 |
| `test_pack_unpack_triton.py::test_pack_seq_default_negative_inf_padding_fp8` | 默认 -inf padding |
| `test_pack_unpack_triton.py::test_pack_seq_edge_cases_fp8` | 极端形状（空 / 单元素 / 单序列） |
| `test_pack_unpack_triton.py::test_pack_seq_different_block_sizes_fp8` | 不同 Triton block size |
| `test_pack_unpack_triton.py::test_pack_seq_shape_consistency` | 输出形状一致性 |
| `test_pack_unpack_triton.py::test_pack_unpack_roundtrip_fp8` | pack→unpack 还原 |
| `test_pack_unpack_triton.py::test_unpack_seq_triton_edge_cases_fp8` | unpack 边界 |

#### A.1.12 `test_lightning_attn.py`
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_linear_decode_forward_triton` | Lightning Attention 线性 decode Triton kernel vs `reference_linear_decode` |
| `test_linear_decode_forward_triton_with_padding` | 上者带 padding |
| `test_lightning_attention_reference` | 参考实现自身的 sanity（block size 等） |

#### A.1.13 `test_cache.py` — KV cache 操作
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_reshape_and_cache` | 标准 paged KV `reshape_and_cache` |
| `test_reshape_and_cache_flash` | FlashAttention layout 版本，覆盖 `kv_cache_dtype∈{auto, fp8, fp8_e5m2, nvfp4}` 与多 scale 类型 |
| `test_swap_blocks` | CPU↔GPU block swap |
| `test_fp8_e4m3_conversion` | FP8 e4m3 cache 数值转换 |
| `test_concat_and_cache_mla` | MLA `concat_and_cache` |
| `test_concat_and_cache_ds_mla` | DeepSeek MLA `concat_and_cache` |
| `test_swap_blocks_mla` | MLA cache 的 swap |
| `test_gather_and_maybe_dequant_cache_mla` | MLA `gather + 可选 dequant` 路径 |
| `test_cp_gather_cache_mla` | Context-parallel `cp_gather` for MLA |
| `test_concat_and_cache_mla_cpu` | MLA cache 写入的 CPU 路径 |

#### A.1.14 Backend selectors
| 文件::函数 | 验证功能 |
| ---------- | -------- |
| `test_attention_selector.py::test_backend_selection` | 全平台 × backend 名称 × `use_mla` × `block_size` 决策矩阵 |
| `test_attention_selector.py::test_fp32_fallback` | FP32 必须回退默认 backend |
| `test_attention_selector.py::test_flash_attn` | FA 默认/版本相关分支 |
| `test_attention_selector.py::test_invalid_backend` | 非法 backend 名称报错 |
| `test_attention_selector.py::test_auto_backend_string` | `"AUTO"` 字符串解析 |
| `test_attention_selector.py::test_auto_backend_selection_behavior` | auto 模式下平台决策 |
| `test_attention_selector.py::test_per_head_quant_scales_backend_selection` | per-head quant scales 限制下的 backend 选择 |
| `test_attention_selector.py::test_non_causal_backend_selection` | 非 causal 显式选择 |
| `test_attention_selector.py::test_non_causal_autoselect_backend` | 非 causal 自动选择 |
| `test_rocm_attention_selector.py::test_selector` | ROCm 平台 backend 选择 |

#### A.1.15 `test_mha_attn.py` — 原生 MultiHeadAttention 层
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_mha_attn_platform` | `MultiHeadAttention` 按平台选 backend |
| `test_mha_attn_forward` | 标准 forward vs `ref_attention` |
| `test_mha_attn_varlen_forward` | varlen forward 正确性 |
| `test_mha_attn_varlen_forward_flashinfer` | varlen 走 FlashInfer 路径 |

#### A.1.16 `test_deepgemm_attention.py`
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_deepgemm_fp8_mqa_logits` | DeepGEMM FP8 MQA logits（`clean_logits∈{True, False}`） |
| `test_deepgemm_fp8_fp4_paged_mqa_logits` | DeepGEMM FP8/FP4 paged MQA logits |

#### A.1.17 `test_cpu_attn.py` — CPU varlen + paged KV
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_varlen_with_paged_kv_normal_vec` | CPU vec ISA 通用路径 |
| `test_varlen_with_paged_kv_normal_amx` | Intel AMX 路径 |
| `test_varlen_with_paged_kv_normal_vec16` | 16-wide vec 路径 |
| `test_varlen_with_paged_kv_normal_neon` | ARM NEON 路径 |
| `test_varlen_with_paged_kv_softcap` | soft cap 分支 |
| `test_varlen_with_paged_kv_alibi` | ALiBi 偏置 |
| `test_varlen_with_paged_kv_sink` | attention sink 分支 |

### A.2 `tests/v1/attention/`（v1 引擎后端）

#### A.2.1 `test_attention_backends.py` — 标准 backend 正确性
覆盖 `BACKENDS_TO_TEST = {FLASH_ATTN, FLASHINFER, FLEX_ATTENTION, TRITON_ATTN, TREE_ATTN, "FLEX_ATTENTION_SLOW"}`（按可用性裁剪）。
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_causal_backend_correctness` | causal 标准 attention vs SDPA reference，多 `BatchSpec` |
| `test_sliding_window_backend_correctness` | sliding window 正确性 |
| `test_sliding_window_encoder_backend_correctness` | encoder（非 causal）+ sliding window |
| `test_non_causal_backend_correctness` | 非 causal（encoder）路径 |

#### A.2.2 `test_mla_backends.py` — MLA backend 正确性
覆盖 `{CUTLASS_MLA, FLASHMLA, FLASH_ATTN_MLA, FLASHINFER_MLA, TRITON_MLA}`。
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_backend_correctness` | 各 MLA backend 在 prefill / decode / mixed batch 上 vs reference；文件顶部注明 `FLASH_ATTN_MLA` 在 `mixed_small` 下偶发 NaN |

#### A.2.3 `test_sparse_mla_backends.py` — Sparse MLA
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_sparse_backend_decode_correctness` | sparse MLA decode vs reference（含 FP8 DS-MLA 量化路径） |
| `test_triton_convert_req_index_to_global_index_decode_only` | decode-only 下 Triton index 转换 |
| `test_triton_convert_req_index_to_global_index_with_prefill_workspace` | 带 prefill workspace 的 index 转换 |
| `test_split_prefill_chunks` | prefill chunk 切分（`seq_lens` × `max_buf`） |
| `test_split_indexer_prefill_chunks` | indexer-aware chunk 切分 |
| `test_split_indexer_prefill_chunks_single_request_overflow` | 单请求超过 buffer 的边界 |
| `test_triton_convert_returns_valid_counts` | 转换后 counts 总和正确 |

#### A.2.4 Backend 选择 (v1)
| 文件::函数 | 验证功能 |
| ---------- | -------- |
| `test_attention_backends_selection.py::test_mamba_layers_get_attn_backend` | Mamba 层走 attention backend 注册 |
| `test_attention_backends_selection.py::test_mamba_layers_have_unified_interface` | Mamba 层接口与 attention 层一致 |
| `test_rocm_attention_backends_selection.py::test_standard_attention_backend_selection` | ROCm 上标准 attention backend 选择 |
| `test_rocm_attention_backends_selection.py::test_mla_backend_selection` | ROCm 上 MLA backend 选择 |
| `test_rocm_attention_backends_selection.py::test_aiter_fa_requires_mi3xx` | AITER FA 必须在 MI3xx |
| `test_rocm_attention_backends_selection.py::test_sparse_not_supported` | ROCm 上 sparse MLA 不支持时报错 |
| `test_mla_prefill_selector.py::TestGetMLAPrefillBackend.*` | 显式选择、非法 backend、import 失败抛错（`test_no_device_capability_returns_flash_attn` / `test_explicit_flash_attn_selection` / `test_explicit_backend_invalid_raises_error` / `test_explicit_backend_import_error_raises`） |
| `test_mla_prefill_selector.py::TestAutoSelectMLAPrefillBackend.*` | Hopper / Blackwell / 全失败下的自动选择（`test_auto_selection_on_hopper` / `test_blackwell_falls_back_to_trtllm` / `test_all_fail_raises_error`） |
| `test_mla_prefill_selector.py::TestBackendValidation.test_r1_dimension_requirement` | DeepSeek R1 维度要求验证 |
| `test_mla_prefill_selector.py::TestMLAPrefillBackendParsing.*` | 字符串 → enum 解析（`test_valid_string_parses_to_enum` / `test_invalid_string_raises_error`） |
| `test_mla_prefill_selector.py::TestDeprecatedFlagMigration.*` | `VLLM_USE_TRTLLM_RAGGED` / `VLLM_DISABLE_FLASHINFER_PREFILL` 等老 flag 的迁移与优先级（5 个 `test_*` 方法） |

#### A.2.5 Metadata / Batch / Splitting
| 文件::函数 | 验证功能 |
| ---------- | -------- |
| `test_attention_splitting.py::test_basic_slice_middle` / `_from_beginning` / `_to_end` / `test_single_element_slice` / `test_full_tensor_slice` / `test_slice_bounds_edge_cases` | `query_start_loc` 切片工具函数边界 |
| `test_attention_splitting.py::test_make_metadata_with_slice_decode_batch` | decode-only batch 元数据切片 |
| `test_attention_splitting.py::test_make_metadata_with_slice_mixed_batch` | mixed batch 切片 |
| `test_attention_splitting.py::test_split_attn_metadata_decode_batch` | `split_attn_metadata` 在大 decode batch 下的切分 |
| `test_attention_splitting.py::test_split_decodes_and_prefills_nonuniform_*` | non-uniform 模式：all_ones / all_short_decodes / all_prefills / mixed_batch |
| `test_attention_splitting.py::test_split_decodes_and_prefills_uniform_*` | uniform 模式：all_ones / all_short_decodes / all_prefills / mixed_uniform / mixed_non_uniform / padded_all_same |
| `test_attention_splitting.py::test_prefill_split_across_ubatches` | prefill 跨 micro-batch 切分 |
| `test_batch_reordering.py::test_reorder_batch_to_split_decodes_and_prefills` | 按 `ReorderTestCase` 重排 batch 使 decode/prefill 分组 |
| `test_chunked_local_attention.py::test_local_attention_virtual_batches` | chunked local attention 的虚拟 batch 切分 |
| `test_kv_head_stride_canonicalization.py::TestCanonicalizeSingletonDimStrides.*` | 10 个方法：FlashInfer / FlashAttention layout 下对 KV singleton 维度 stride 的 canonical 化（含 TMA 对齐、data pointer 保持、多 singleton 维、非连续外维等） |
| `test_gdn_metadata_builder.py::test_gdn_build_classification` | GDN metadata 构建分类 |
| `test_gdn_metadata_builder.py::test_has_initial_state_after_reclassification` | reclassify 后 `has_initial_state` 标志 |
| `test_indexer_deepseek_v4_slot_mapping.py::test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size` | DeepSeek-V4 indexer 用 storage block size 而非 attention block size |
| `test_mamba_update_block_table.py::test_update_block_table_copies_block_idx_to_persistent_buffers` | Mamba block table 更新写入 persistent buffer |

#### A.2.6 `test_trtllm_attention_integration.py` — TRT-LLM gen 集成
| 测试函数 | 验证功能 |
| -------- | -------- |
| `test_trtllm_gen_full_attention_integration` | TRT-LLM gen 全 attention 端到端集成（多 `BatchSpec`） |
| `test_trtllm_gen_nvfp4_kv_integration` | NVFP4 KV cache 下 TRT-LLM gen 集成 |

### A.3 其他相关位置（速查）
| 文件::函数 | 验证功能 |
| ---------- | -------- |
| `tests/test_attention_backend_registry.py::test_custom_is_not_alias_of_any_backend` | `CUSTOM` 不与现有 backend 别名冲突 |
| `tests/test_attention_backend_registry.py::test_register_custom_backend_with_class_path` | 通过 class path 注册自定义 attention backend |
| `tests/test_attention_backend_registry.py::test_mamba_custom_is_not_alias_of_any_backend` | Mamba 的 CUSTOM 同上 |
| `tests/test_attention_backend_registry.py::test_register_custom_mamba_backend_with_class_path` | 注册自定义 mamba backend |
| `tests/kernels/test_flex_attention.py::test_flex_attention_full_cudagraphs` | FlexAttention 在 full CUDA graph 下端到端 |
| `tests/kernels/test_flex_attention.py::test_flex_attention_vs_default_backend` | 输出与默认 backend 对齐 |
| `tests/kernels/test_flex_attention.py::test_encoder_flex_attention_vs_default_backend` | encoder 路径对齐 |
| `tests/kernels/test_flex_attention.py::test_block_mask_direct_vs_slow_path` | block mask 直接路径 vs slow path |
| `tests/kernels/test_flex_attention.py::test_physical_to_logical_mapping_handles_reused_blocks` | 复用块的物理→逻辑映射 |
| `tests/kernels/test_flex_attention.py::test_block_sparsity_hint_prunes_blocks` | sparsity hint 正确裁剪 block |
| `tests/v1/e2e/general/test_cascade_attention.py::test_cascade_attention` | v1 引擎 cascade attention 端到端 |
| `tests/v1/e2e/general/test_correctness_sliding_window.py` | sliding window 端到端正确性 |
| `tests/v1/spec_decode/test_tree_attention.py::test_tree_attn_correctness` | 投机解码 tree attention 正确性 |
| `tests/compile/passes/test_fusion_attn.py::test_attention_quant_pattern` | Attention + quant 融合 pass |
| `tests/compile/passes/test_mla_attn_quant_fusion.py::test_mla_attention_quant_pattern` | MLA + quant 融合 |
| `tests/compile/passes/test_qk_norm_rope_fusion.py` | QK-Norm + RoPE 融合 |
| `tests/compile/passes/test_rope_kvcache_fusion.py` | RoPE + KV cache 写入融合 |
| `tests/compile/passes/test_fuse_mla_dual_rms_norm.py` | MLA dual RMSNorm 融合 |
| `tests/kernels/core/test_vit_fp8_attn.py` / `test_vit_fp8_scaling.py` | ViT FP8 attention/scaling kernel |
| `tests/models/multimodal/generation/test_vit_backend_functionality.py` | ViT attention backend 功能性 |

> 注：上表只列出 `test_*` 函数；这些函数大多通过 `pytest.mark.parametrize` 形成数十~数百个子用例，子用例参数空间见正文第 2~3 节。

---

## 12. 备注

- 大量 kernel 测试通过 `pytest.mark.parametrize` 形成笛卡尔积，实际用例数远大于函数数量。修改通用 attention 路径前建议先在小参数子集上快速回归。
- `tests/kernels/attention/conftest.py` 控制 device/平台过滤；`tests/v1/attention/utils.py` 中的 `BatchSpec`、`BackendConfig`、`create_common_attn_metadata` 是 v1 后端测试的核心工具。
- `tests/v1/attention/test_mla_backends.py` 顶部已注明 `FLASH_ATTN_MLA` 在 `mixed_small` 用例下偶发 NaN 的已知问题，调试时需关注用例执行顺序。
- 当 FlashInfer / AITER / DeepGEMM / CUTLASS MLA / FlashMLA 等可选依赖缺失时，相关测试会通过 `pytest.skip(..., allow_module_level=True)` 跳过整个文件，CI 中是否覆盖取决于运行环境。
