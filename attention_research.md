# vLLM Attention 模块深度研究报告

## 目录

1. [概述](#1-概述)
   - 1.1 [各 Attention 组件之间的关系](#各-attention-组件之间的关系)
2. [架构总览](#2-架构总览)
3. [核心 Attention 层](#3-核心-attention-层)
   - 3.1 [Eager Mode 与 Compile Mode 控制逻辑](#eager-mode-与-compile-mode-控制逻辑)
4. [V1 Backend 抽象层](#4-v1-backend-抽象层)
5. [Backend 选择机制](#5-backend-选择机制)
6. [主要 Backend 实现](#6-主要-backend-实现)
   - 6.1 [FlashAttention](#61-flashattention-backend)
   - 6.2 [FlashInfer](#62-flashinfer-backend)
   - 6.3 [Triton](#63-triton-backend)
7. [MLA (Multi-Head Latent Attention)](#7-mla-multi-head-latent-attention)
8. [Paged Attention 与 KV Cache](#8-paged-attention-与-kv-cache)
9. [特殊 Attention 变体](#9-特殊-attention-变体)
10. [完整调用流程](#10-完整调用流程)
11. [性能优化技术](#11-性能优化技术)
12. [关键发现与总结](#12-关键发现与总结)

---

## 1. 概述

vLLM 的 attention 模块是整个推理引擎的核心组件，负责实现高效的注意力计算和 KV Cache 管理。它采用了多层抽象的设计，支持多种硬件后端和注意力变体，是 vLLM 实现高吞吐量 LLM 推理的关键。

### 核心设计理念

- **Paged Attention**：将 KV Cache 组织为固定大小的页（block），通过 block table 映射实现虚拟内存式的内存管理，消除内存碎片
- **Backend 可插拔**：通过抽象接口支持 FlashAttention、FlashInfer、Triton 等多种计算后端
- **双路径架构**：项目同时维护 legacy（`model_executor/layers/attention/`）和 V1（`v1/attention/`）两套实现
- **Custom Ops**：通过 `torch.ops.vllm` 自定义算子实现与 `torch.compile` 的兼容

### 各 Attention 组件之间的关系

vLLM 中的 FlashAttention、FlashInfer、Triton、MLA、Paged Attention 以及各种特殊 Attention 变体并非平级并列的概念，而是分属于**三个不同维度**，互相正交组合：

```
维度 1: 注意力类型（WHAT — 计算什么样的注意力）
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  标准 Decoder Attention (Attention)                                 │
│  ├── 全量注意力 (FullAttentionSpec)                                 │
│  ├── 滑动窗口注意力 (SlidingWindowSpec)                             │
│  ├── Chunked Local Attention — 分块局部注意力                       │
│  └── Static Sink Attention — 保留初始 token 的注意力                │
│                                                                     │
│  MLA Attention (MLAAttention) — 低秩潜在空间 KV 压缩               │
│  Cross Attention (CrossAttention) — 编码器-解码器交叉注意力         │
│  Encoder-Only Attention (EncoderOnlyAttention) — 双向注意力         │
│  Mamba/SSM Attention — 线性时间状态空间模型                         │
│  Linear Attention — O(n) 线性注意力                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

维度 2: 计算后端（HOW — 用什么硬件/核来计算）
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  标准 Attention 后端               MLA 专用后端                              │
│  (标准 Attention 使用)             (MLA Attention 使用)                      │
│  ┌────────────────────────────┐   ┌────────────────────────────────┐        │
│  │ FlashAttention v2/v3 [C/R/X]│   │ FlashMLA          [CUDA]      │        │
│  │ FlashInfer          [CUDA] │   │ FlashInfer MLA    [CUDA]      │        │
│  │ Triton Attn       [ROCm/X] │   │ CUTLASS MLA       [CUDA SM100]│        │
│  │ ROCm Attn          [ROCm] │ ◄─►│ FlashAttn MLA     [CUDA/ROCm] │        │
│  │ ROCm AITER FA      [ROCm] │互替│ Triton MLA       [ROCm/X/CPU] │        │
│  │ ROCm AITER Unified [ROCm] │   │ ROCm AITER MLA    [ROCm]      │        │
│  │ Flex Attention      [CUDA] │   │ ROCm AITER Triton MLA [ROCm]  │        │
│  │ CPU Attn            [CPU]  │   │                                │        │
│  └────────────────────────────┘   └────────────────────────────────┘        │
│  [C]=CUDA [R]=ROCm [X]=XPU                                                  │
│       ↑ 根据注意力类型二选一，两组后端互相替代 ↑                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

维度 3: 内存管理（WHERE — KV Cache 如何存储和访问）
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Paged Attention — 所有 Backend 共享的底层内存管理机制              │
│  ├── Block Table: 虚拟块 → 物理块映射                               │
│  ├── Slot Mapping: Token → Cache Slot 精确定位                      │
│  └── KVCacheSpec: 每种注意力类型的内存需求规格                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 组合关系示例

这三个维度正交组合，形成具体的执行路径：

| 注意力类型 | 计算后端 | 内存管理 | 实际场景 |
|-----------|---------|---------|---------|
| Attention（标准） | FlashAttention v3 | Paged (FullAttentionSpec) | LLaMA/GPT 推理 |
| Attention（滑动窗口） | FlashAttention v2 | Paged (SlidingWindowSpec) | Mistral 推理 |
| Attention（标准） | Triton | Paged (FullAttentionSpec) | 编码器模型/小 batch |
| MLA Attention | Triton MLA | Paged (MLAAttentionSpec) | DeepSeek-V3 解码 |
| MLA Attention | FlashMLA | Paged (MLAAttentionSpec) | DeepSeek-V3 高性能 |
| Cross Attention | FlashAttention | Paged (CrossAttentionSpec) | T5/BART 推理 |
| Encoder-Only | FlashAttention | 无 Cache | BERT 推理 |
| Chunked Local | FlashAttention | Paged (ChunkedLocalAttentionSpec) | 长序列局部注意力 |
| Static Sink | FlashAttention | Paged (SinkFullAttentionSpec) | StreamingLLM |
| Mamba/SSM | Mamba Backend | MambaSpec（固定状态） | Mamba 模型 |

#### 关键关系总结

1. **Paged Attention 是底层基础设施**，不是一种注意力类型。它提供所有 Backend 共享的 KV Cache 分页内存管理（block table + slot mapping），无论上层使用 FlashAttention、FlashInfer 还是 Triton 后端，底层都通过 Paged Attention 的 block 管理来读写 KV Cache。

2. **FlashAttention / FlashInfer / Triton 是同层可替换的后端**。它们实现相同的 `AttentionBackend` 接口，通过 `selector.py` 根据硬件和模型配置自动选择。它们的功能等价，但性能和兼容性不同。选择关系如下：

   ```
   get_attn_backend()  (CUDA 平台，标准 Attention)
     ├─ NVIDIA Blackwell (CC 10+) → FlashInfer (优先) → FlashAttention → Triton
     ├─ NVIDIA Hopper (CC 9.x)   → FlashAttention (优先) → FlashInfer → Triton
     ├─ NVIDIA Ampere (CC 8.x)   → FlashAttention (优先) → FlashInfer → Triton
     ├─ NVIDIA 更早架构 (CC < 8)  → Triton (唯一选择)
     ├─ AMD GPU                   → ROCm Backend
     └─ CPU                       → CPU Backend
   ```

   **三者的具体关系——共同点与差异**：

   **共同点**：三者均实现 `AttentionBackend` 接口，对外暴露一致的 `forward()` 签名，支持 Paged KV Cache（block table + slot mapping）、Cascade Attention（共享前缀优化）和 Sliding Window，且都提供 FP8 KV Cache 量化支持。

   **差异对比**：

   | 维度 | FlashAttention | FlashInfer | Triton |
   |------|---------------|------------|--------|
   | **底层实现** | Dao-AILab Flash-Attention 库（CUDA C++ 核） | FlashInfer 库（CUDA C++ 核 + Blackwell 走 TRTLLM 核） | vLLM 自研 Triton 核（Python-DSL，设备无关） |
   | **最低硬件** | CC ≥ 8.0（Ampere+） | CC ≥ 7.5 ~ 12.1（Turing ~ Blackwell） | 无限制（任意 GPU） |
   | **数据类型** | float16, bfloat16 | float16, bfloat16 | float16, bfloat16, **float32**（唯一支持） |
   | **FP8 格式** | fp8（v3+） | fp8, fp8_e4m3, fp8_e5m2（最全） | fp8, fp8_e4m3, fp8_e5m2 |
   | **Block Size** | 16 的倍数 | 仅 16/32/64（更严格） | 16 的倍数 |
   | **Head Size** | 通用 | 仅 64/128/256（最严格） | ≥ 32（最宽松） |
   | **Encoder Attn** | 全部（DECODER/ENCODER/ENCODER_ONLY/ENCODER_DECODER） | 仅 DECODER | 全部 |
   | **MM Prefix** | ✗ | ✗ | ✓（唯一支持多模态前缀） |
   | **Per-Head 量化** | ✓（v3+） | ✗ | ✗ |
   | **Sink Token** | ✓（需版本支持） | ✓（需 TRTLLM, SM100+） | ✓ |
   | **ALiBi-sqrt** | ✗ | ✗ | ✓（唯一支持） |
   | **Blackwell 优化** | 支持 | 最优（TRTLLM 路径 + HND 布局） | 通用 |

   **选择逻辑要点**：
   - **性能优先**：在 Blackwell 上 FlashInfer 通过 TRTLLM 路径获得最佳性能，因此优先级最高；在 Hopper/Ampere 上 FlashAttention 是最成熟、性能最优的选择
   - **兼容性兜底**：Triton 是 vLLM 自研的通用后端，不依赖外部库，支持最广泛的硬件和特性（float32、ALiBi-sqrt、MM Prefix 等），当 FlashAttention 和 FlashInfer 都不满足条件时作为最终回退
   - **自动降级**：`selector.py` 按优先级逐一调用各后端的 `validate_configuration()` 方法检查 10+ 项兼容性条件（head_size、dtype、block_size、attn_type 等），第一个通过全部校验的后端即被选用

3. **MLA 后端与标准后端互相替代**。当模型使用 MLA 注意力类型时，系统从 MLA 专用后端（FlashMLA、Triton MLA 等）中选择；当使用标准 Attention 时，则从标准后端（FlashAttention、FlashInfer、Triton）中选择。两组后端各自内部可替换，但跨组不混用——由注意力类型决定走哪条路径。MLA 拥有独立的 `MLAAttention` 层和前向路径（`forward_mha` + `forward_mqa`），但同样使用 Paged Attention 管理 KV Cache（只是 cache 的形状不同——576 维 vs 标准 MHA 的数千维）。

4. **特殊 Attention 变体是标准 Attention 的扩展**。`ChunkedLocalAttention`、`StaticSinkAttention` 等都继承并扩展了基础 `Attention` 类的行为，主要差异在于：
   - **Block Table 重组**（Chunked Local 创建虚拟 batch）
   - **Cache 策略变化**（Sink 保留初始 token，Sliding Window 丢弃旧 token）
   - **元数据构建差异**（各自有独立的 `MetadataBuilder`）
   
   它们仍然使用相同的 FlashAttention/FlashInfer/Triton 后端执行实际计算。

5. **Mamba/SSM 和 Linear Attention 是非 Transformer 注意力**。它们不使用 Q/K/V 点积注意力机制，而是使用状态空间模型或线性递推，但在 vLLM 中仍然实现了 `AttentionBackend` 接口以统一管理。

#### 层次关系图

```
                    ┌──────────────────────┐
                    │   Model Layer        │
                    │  (LlamaAttention等)  │
                    └─────────┬────────────┘
                              │ 创建
                              ▼
              ┌───────────────────────────────────┐
              │         Attention 类型层           │
              │  ┌──────────┐  ┌────────────────┐ │
              │  │Attention │  │ MLAAttention   │ │
              │  │(标准)    │  │ (低秩压缩)     │ │
              │  └────┬─────┘  └───────┬────────┘ │
              │       │                │           │
              │  ┌────┴───────────┐    │           │
              │  │ 特殊变体:      │    │           │
              │  │ CrossAttn /   │    │           │
              │  │ EncoderOnly / │    │           │
              │  │ ChunkedLocal /│    │           │
              │  │ StaticSink    │    │           │
              │  └───────────────┘    │           │
              └───────┬───────────────┼───────────┘
                      │ 委托给        │ 委托给
                      ▼               ▼
   ┌─────────────────────────┐ ┌─────────────────────────┐
   │ 标准后端 (互相替代)     │ │ MLA 专用后端 (互相替代) │
   │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │
   │ │FlashAttn   [C/R/X]  │ │ │ │FlashMLA      [CUDA] │ │
   │ │FlashInfer  [CUDA]   │ │ │ │FlInferMLA    [CUDA] │ │
   │ │Triton      [R/X]    │ │ │ │CUTLASS MLA   [CUDA] │ │
   │ │ROCm Attn   [ROCm]   │ │ │ │FlashAttnMLA  [C/R]  │ │
   │ │ROCm AITER  [ROCm]   │ │ │ │TritonMLA     [R/X]  │ │
   │ │FlexAttn    [CUDA]   │ │ │ │ROCm AITER MLA[ROCm] │ │
   │ │CPU Attn    [CPU]    │ │ │ │                      │ │
   │ └─────────────────────┘ │ │ └─────────────────────┘ │
   └────────────┬────────────┘ └────────────┬────────────┘
                │ 两组后端互相替代(◄─►)      │
                └────────────┬───────────────┘
                             │ 读写
                             ▼
              ┌───────────────────────────────────┐
              │     Paged Attention 内存层         │
              │  ┌──────────────────────────────┐ │
              │  │ Block Table + Slot Mapping   │ │
              │  │ KV Cache Pool (物理块)       │ │
              │  │ KVCacheSpec (规格计算)        │ │
              │  └──────────────────────────────┘ │
              └───────────────────────────────────┘
[C]=CUDA  [R]=ROCm  [X]=XPU
```

---

## 2. 架构总览

### 目录结构

```
vllm/
├── model_executor/layers/attention/         # 模型层 Attention 接口
│   ├── attention.py                         # 核心 Attention 类
│   ├── attention_layer_base.py              # 抽象基类
│   ├── cross_attention.py                   # 交叉注意力
│   ├── mla_attention.py                     # MLA 注意力
│   ├── encoder_only_attention.py            # 编码器注意力
│   ├── chunked_local_attention.py           # 分块局部注意力
│   ├── static_sink_attention.py             # Sink Token 注意力
│   └── mm_encoder_attention.py              # 多模态编码器注意力
│
├── v1/attention/                            # V1 引擎 Attention 后端
│   ├── backend.py                           # 抽象 Backend 接口
│   ├── selector.py                          # Backend 选择逻辑
│   ├── backends/                            # 具体 Backend 实现
│   │   ├── flash_attn.py                    # FlashAttention v2/v3
│   │   ├── flashinfer.py                    # FlashInfer
│   │   ├── triton_attn.py                   # Triton
│   │   ├── cpu_attn.py                      # CPU 回退
│   │   ├── rocm_attn.py                     # ROCm (AMD)
│   │   ├── mamba_attn.py / mamba2_attn.py   # Mamba SSM
│   │   ├── linear_attn.py                   # 线性注意力
│   │   ├── tree_attn.py                     # 树形注意力（推测解码）
│   │   └── mla/                             # MLA 专用后端
│   │       ├── flashmla.py
│   │       ├── flashinfer_mla.py
│   │       ├── triton_mla.py
│   │       └── cutlass_mla.py
│   └── ops/                                 # 底层算子
│       ├── paged_attn.py                    # Paged Attention 接口
│       ├── merge_attn_states.py             # 注意力状态合并
│       └── triton_prefill_attention.py      # Triton Prefill 核
│
├── v1/kv_cache_interface.py                 # KV Cache 规格定义
│
└── csrc/attention/                          # CUDA/C++ 内核
    ├── attention_kernels.cuh                # 核心 Paged Attention CUDA 内核模板
    ├── attention_generic.cuh                # 通用注意力工具函数
    ├── attention_utils.cuh                  # 注意力计算工具（warp reduce 等）
    ├── attention_dtypes.h                   # 数据类型分发
    ├── dtype_float16.cuh                    # FP16 向量化操作
    ├── dtype_bfloat16.cuh                   # BF16 向量化操作
    ├── dtype_float32.cuh                    # FP32 向量化操作
    ├── dtype_fp8.cuh                        # FP8 向量化操作
    ├── paged_attention_v1.cu                # Paged Attention V1（短序列，单 pass）
    ├── paged_attention_v2.cu                # Paged Attention V2（长序列，分区 reduce）
    ├── merge_attn_states.cu                 # 多 Backend 注意力状态合并（LogSumExp）
    ├── vertical_slash_index.cu              # 稀疏注意力索引计算
    └── mla/                                 # CUTLASS MLA 内核（SM100/Blackwell）
        ├── sm100_cutlass_mla_kernel.cu      # MLA 主入口
        └── cutlass_sm100_mla/               # CUTLASS MLA 模板库
            ├── device/sm100_mla.hpp         # Device 级 MLA 配置
            └── kernel/                      # Kernel 级实现
                ├── sm100_fmha_mla_tma_warpspecialized.hpp  # TMA warp 特化核
                ├── sm100_fmha_mla_reduction.hpp            # Reduction 核
                └── sm100_mla_tile_scheduler.hpp            # Tile 调度器
```

### Kernel 实现文件结构（全平台）

```
vllm/
├── csrc/attention/                          # CUDA 注意力内核（见上方详细列表）
│
├── csrc/rocm/                               # ROCm (AMD) 专用内核
│   └── attention.cu                         # ROCm Paged Attention 实现
│
├── csrc/cpu/                                # CPU 注意力内核
│   ├── cpu_attn.cpp                         # CPU 注意力主入口
│   ├── cpu_attn_impl.hpp                    # 实现模板
│   ├── cpu_attn_vec.hpp                     # SSE/AVX 向量化实现
│   ├── cpu_attn_vec16.hpp                   # AVX-512 向量化实现
│   ├── cpu_attn_amx.hpp                     # AMX（Intel Advanced Matrix Extensions）实现
│   ├── cpu_attn_neon.hpp                    # ARM NEON 实现
│   ├── cpu_attn_neon_bfmmla.hpp             # ARM NEON BF16 矩阵乘实现
│   ├── cpu_attn_vxe.hpp                     # s390x VXE 实现
│   ├── mla_decode.cpp                       # CPU MLA 解码实现
│   └── generate_cpu_attn_dispatch.py        # 自动生成分发代码
│
├── v1/attention/ops/                        # Python 层注意力算子
│   ├── paged_attn.py                        # Paged Attention Python 接口
│   ├── merge_attn_states.py                 # 注意力状态合并
│   ├── triton_merge_attn_states.py          # Triton 实现的状态合并
│   ├── prefix_prefill.py                    # 前缀 Prefill 核（ROCm 使用）
│   ├── triton_prefill_attention.py          # Triton Prefill 注意力核
│   ├── triton_decode_attention.py           # Triton Decode 注意力核
│   ├── triton_unified_attention.py          # Triton 统一注意力核（prefill+decode）
│   ├── triton_reshape_and_cache_flash.py    # Triton KV Cache 写入核
│   ├── chunked_prefill_paged_decode.py      # 分块 Prefill + Paged Decode 融合核
│   ├── flashmla.py                          # FlashMLA 算子封装
│   ├── rocm_aiter_mla_sparse.py             # ROCm AITER 稀疏 MLA 算子
│   ├── vit_attn_wrappers.py                 # ViT 注意力包装器
│   └── common.py                            # 公共工具函数
│
├── v1/attention/backends/                   # Python 后端实现
│   ├── flash_attn.py                        # FlashAttention v2/v3 [CUDA/ROCm/XPU]
│   ├── flash_attn_diffkv.py                 # FlashAttention 差分 KV [CUDA]
│   ├── flashinfer.py                        # FlashInfer [CUDA]
│   ├── triton_attn.py                       # Triton 注意力 [ROCm/XPU]
│   ├── flex_attention.py                    # PyTorch Flex Attention [CUDA]
│   ├── cpu_attn.py                          # CPU 注意力 [CPU]
│   ├── rocm_attn.py                         # ROCm 注意力 [ROCm]
│   ├── rocm_aiter_fa.py                     # ROCm AITER FlashAttention [ROCm gfx9]
│   ├── rocm_aiter_unified_attn.py           # ROCm AITER 统一注意力 [ROCm gfx9]
│   ├── tree_attn.py                         # 树形注意力（推测解码）
│   ├── gdn_attn.py                          # GDN 注意力
│   ├── linear_attn.py                       # 线性注意力
│   ├── short_conv_attn.py                   # 短卷积注意力
│   ├── mamba_attn.py                        # Mamba SSM
│   ├── mamba1_attn.py                       # Mamba-1
│   ├── mamba2_attn.py                       # Mamba-2
│   └── mla/                                 # MLA 专用后端
│       ├── flashmla.py                      # FlashMLA [CUDA]
│       ├── flashmla_sparse.py               # FlashMLA Sparse [CUDA]
│       ├── flashattn_mla.py                 # FlashAttn MLA [CUDA/ROCm]
│       ├── flashinfer_mla.py                # FlashInfer MLA [CUDA]
│       ├── flashinfer_mla_sparse.py         # FlashInfer MLA Sparse [CUDA]
│       ├── cutlass_mla.py                   # CUTLASS MLA [CUDA SM100]
│       ├── triton_mla.py                    # Triton MLA [ROCm/XPU/CPU]
│       ├── aiter_triton_mla.py              # AITER Triton MLA [ROCm]
│       ├── rocm_aiter_mla.py                # ROCm AITER MLA [ROCm gfx9]
│       └── rocm_aiter_mla_sparse.py         # ROCm AITER MLA Sparse [ROCm gfx9]
│
└── model_executor/layers/                   # 注意力层实现
    └── attention/
        ├── attention.py                     # 标准 Decoder Attention
        ├── mla_attention.py                 # MLA Attention
        ├── cross_attention.py               # 交叉注意力
        ├── encoder_only_attention.py        # 编码器注意力 (BERT)
        ├── chunked_local_attention.py       # 分块局部注意力
        ├── static_sink_attention.py         # Sink Token 注意力
        ├── mm_encoder_attention.py          # 多模态编码器注意力
        └── lightning_attn.py                # Lightning Attention
```

### 类层次关系

```
AttentionLayerBase (ABC)                    ← 抽象接口
├── Attention (nn.Module)                   ← 标准 Decoder Attention
├── MLAAttention (nn.Module)                ← Multi-Head Latent Attention
├── CrossAttention (nn.Module)              ← 编码器-解码器交叉注意力
├── EncoderOnlyAttention (nn.Module)        ← 编码器注意力 (BERT)
├── ChunkedLocalAttention (nn.Module)       ← 分块局部注意力
├── StaticSinkAttention (nn.Module)         ← Sink Token 注意力
└── MMEncoderAttention (nn.Module)          ← 多模态编码器注意力

AttentionBackend (ABC)                      ← Backend 抽象接口
├── FlashAttentionBackend                   ← FlashAttention v2/v3
├── FlashInferBackend                       ← FlashInfer
├── TritonAttentionBackend                  ← Triton 自定义核
├── CPUAttentionBackend                     ← CPU 实现
├── ROCmAttentionBackend                    ← AMD ROCm
└── ...                                     ← 更多 Backend

AttentionImpl (ABC)                         ← 实现层抽象
├── FlashAttentionImpl                      ← FlashAttention 实现
├── FlashInferImpl                          ← FlashInfer 实现
├── TritonAttentionImpl                     ← Triton 实现
├── MLAAttentionImpl                        ← MLA 实现
└── ...
```

---

## 3. 核心 Attention 层

### 3.1 AttentionLayerBase（抽象基类）

**文件**：`vllm/model_executor/layers/attention/attention_layer_base.py`

定义了所有注意力层必须实现的接口：

```python
class AttentionLayerBase(ABC):
    impl: "AttentionImpl"  # Backend 实现实例

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """获取该层使用的 Backend 类"""

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """获取 KV Cache 规格，用于内存分配"""
```

### 3.2 Attention 类（核心实现）

**文件**：`vllm/model_executor/layers/attention/attention.py`

这是 vLLM 中最核心的注意力层，被所有标准 Transformer 模型使用。

#### 初始化流程

```python
class Attention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,              # Query 头数
        head_size: int,              # 每个头的维度
        scale: float,                # Softmax 缩放因子（通常为 1/√d）
        num_kv_heads: int | None,    # KV 头数（支持 GQA/MQA）
        alibi_slopes: list | None,   # ALiBi 位置编码斜率
        cache_config: CacheConfig,   # KV Cache 配置
        quant_config: QuantizationConfig,  # 量化配置
        logits_soft_cap: float | None,     # Logit 上限（Gemini 风格）
        per_layer_sliding_window: int | None,  # 滑动窗口大小
        ...
    ):
```

初始化过程按以下步骤执行：

1. **确定滑动窗口大小** — 优先使用层级配置，否则使用全局配置
2. **配置 KV Cache 数据类型** — 支持 auto、fp8、bfloat16 等
3. **选择 Attention Backend** — 调用 `get_attn_backend()` 根据硬件和配置选择最优后端
4. **创建 Backend 实现** — 通过 `backend.get_impl_cls()` 获取实现类并实例化
5. **初始化 KV Cache** — 为每个 Pipeline 并行阶段创建占位 KV Cache 张量
6. **设置量化参数** — 初始化 FP8 Q/K/V 缩放因子（详见下方说明）
7. **确定调用模式** — 根据平台设置 `use_direct_call`（详见"Eager Mode 与 Compile Mode"小节）

#### FP8 Q/K/V 缩放因子详解

FP8 (8-bit 浮点) 格式仅有 8 bit，动态范围有限（E4M3 格式最大 ±240）。缩放因子的作用是将高精度张量映射到 FP8 可表示的范围内，保留相对精度。

**数学定义**：

```
scale = max(abs(tensor)) / range_constant
      ↑ 张量中所有元素取绝对值后的最大值
```

其中 `range_constant` 为预设常数（默认 Q=200, K=200, V=100），用于在 FP8 范围内留出余量。

**三个缩放因子**：

| 缩放因子 | 用途 | 计算方式 |
|---------|------|---------|
| `_q_scale` | Query 量化缩放 | `max(abs(query)) / 200` |
| `_k_scale` | Key 写入 KV Cache 时量化 | `max(abs(key)) / 200` |
| `_v_scale` | Value 写入 KV Cache 时量化 | `max(abs(value)) / 100` |

**计算时机与流程**：

```python
# 1. 初始化时注册为缓冲区（默认值 1.0）
self.register_buffer("_q_scale", torch.tensor(1.0))
self.register_buffer("_k_scale", torch.tensor(1.0))
self.register_buffer("_v_scale", torch.tensor(1.0))

# 2. 首次 forward 时动态计算（仅一次）
def calc_kv_scales(self, query, key, value):
    self._q_scale.copy_(torch.abs(query).max() / self.q_range)  # 200
    self._k_scale.copy_(torch.abs(key).max() / self.k_range)    # 200
    self._v_scale.copy_(torch.abs(value).max() / self.v_range)   # 100
    self.calculate_kv_scales = False  # 后续 forward 不再计算

# 3. 量化（写入 Cache）— E4M3 格式范围: [-240, 240]
key_fp8 = (key / k_scale).clamp(-240, 240).to(fp8_e4m3)

# 4. 反量化（读取时在核内融合）
key_restored = key_fp8 * k_scale
```

**两种模式**：
- **动态缩放**（`calculate_kv_scales=True`）：首次 forward 时根据实际数据计算，之后冻结复用
- **静态缩放**（`calculate_kv_scales=False`）：使用模型 checkpoint 中保存的缩放因子或默认值 1.0

**为什么需要缩放因子**：直接将 FP16/BF16 值截断为 FP8 会导致大量溢出或精度丢失。缩放因子确保数据分布适配 FP8 的有限范围，使量化误差最小化。当前 vLLM 采用 **per-tensor** 缩放（整个张量共享一个 scale），而非 per-channel。

#### Forward 方法

```python
def forward(
    self,
    query: torch.Tensor,    # [num_tokens, num_heads * head_size]
    key: torch.Tensor,      # [num_tokens, num_kv_heads * head_size]
    value: torch.Tensor,    # [num_tokens, num_kv_heads * head_size]
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
```

**执行流程**（详细）：

```
forward(query, key, value, output_shape)
  │
  ├─ 1. 计算 KV 缩放因子（FP8 量化，仅首次调用）
  │     torch.ops.vllm.maybe_calc_kv_scales(query, key, value, layer_name)
  │     │
  │     └─ 首次调用时:
  │        _q_scale = max(|query|) / q_range   # per-tensor scale
  │        _k_scale = max(|key|) / k_range
  │        _v_scale = max(|value|) / v_range
  │        calculate_kv_scales = False          # 之后不再计算
  │
  ├─ 2. 量化 Query（如果 Backend 支持 FP8 query 输入）
  │     if self.impl.supports_quant_query_input:
  │         query, _ = self.query_quant(query, self._q_scale)
  │         # query: FP16 → FP8 (query / q_scale → clamp → fp8_e4m3)
  │
  ├─ 3. 重塑张量维度（2D → 3D）
  │     query:  [num_tokens, num_heads * head_size]
  │          →  [num_tokens, num_heads, head_size]
  │     key:    [num_tokens, num_kv_heads * head_size]
  │          →  [num_tokens, num_kv_heads, head_size]
  │     value:  [num_tokens, num_kv_heads * head_size]
  │          →  [num_tokens, num_kv_heads, head_size]
  │
  ├─ 4. 更新 KV Cache（如果 Backend 不在 forward 中自行更新）
  │     if not self.impl.forward_includes_kv_cache_update:
  │         torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)
  │         │
  │         └─ 内部流程:
  │            ├─ get_attention_context(layer_name)
  │            │   → 从 ForwardContext 获取 kv_cache, slot_mapping
  │            ├─ impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
  │            │   ├─ key_cache, value_cache = kv_cache.unbind(0)
  │            │   ├─ reshape_and_cache_flash(key, value, key_cache, value_cache,
  │            │   │                          slot_mapping, kv_cache_dtype,
  │            │   │                          k_scale, v_scale)
  │            │   │   └─ CUDA 核: 按 slot_mapping 将 K/V 写入对应物理块位置
  │            │   │       如果是 FP8 模式: 写入前将 K/V 量化为 FP8
  │            │   └─ 返回
  │
  ├─ 5. 执行注意力计算
  │     torch.ops.vllm.unified_attention_with_output(
  │         query, key, value, output, layer_name)
  │     │
  │     └─ 内部流程:
  │        ├─ get_attention_context(layer_name)
  │        │   → 从 ForwardContext 获取 attn_metadata, kv_cache
  │        ├─ impl.forward(layer, query, key, value, kv_cache, attn_metadata,
  │        │               output=output, ...)
  │        │   ├─ 以 FlashAttention 为例:
  │        │   │   ├─ key_cache, value_cache = kv_cache.unbind(0)
  │        │   │   ├─ 检查是否使用 Cascade Attention（共享前缀优化）
  │        │   │   ├─ flash_attn_varlen_func(
  │        │   │   │     q=query[:num_actual_tokens],
  │        │   │   │     k=key_cache,           # 从 Paged KV Cache 读取
  │        │   │   │     v=value_cache,
  │        │   │   │     cu_seqlens_q=query_start_loc,
  │        │   │   │     seqused_k=seq_lens,
  │        │   │   │     block_table=block_table,  # 虚拟→物理块映射
  │        │   │   │     softmax_scale=1/√d,
  │        │   │   │     causal=True,
  │        │   │   │     window_size=sliding_window,
  │        │   │   │     k_descale=_k_scale,     # FP8 反量化因子
  │        │   │   │     v_descale=_v_scale,
  │        │   │   │   )
  │        │   │   └─ 返回 output: [num_tokens, num_heads, head_size]
  │        │   │
  │        │   ├─ 以 FlashInfer 为例:
  │        │   │   ├─ wrapper.run(q, kv_cache)  # 使用预规划的 wrapper
  │        │   │   └─ 返回 output
  │        │   │
  │        │   └─ 以 Triton 为例:
  │        │       ├─ unified_attention(q, k_cache, v_cache, ...)  # 自适应 2D/3D 核
  │        │       └─ 返回 output
  │        │
  │        └─ output 已写入预分配的输出缓冲区（原地操作）
  │
  └─ 6. 返回结果
        return output.reshape(output_shape)
        # output: [num_tokens, num_heads * head_size]
        # → 后续经过 O Projection → 残差连接 → 下一层
```

#### Eager Mode 与 Compile Mode 控制逻辑

vLLM 的注意力系统支持两种执行模式：**Eager Mode**（直接函数调用）和 **Compile Mode**（通过不透明 Custom Op 调用）。模式选择影响 `torch.compile` 对注意力计算的可见性。

##### 编译模式配置

**文件**：`vllm/config/compilation.py`

```python
class CompilationMode(enum.IntEnum):
    NONE = 0               # 完全 Eager 模式，不使用 torch.compile
    STOCK_TORCH_COMPILE = 1  # 标准 torch.compile 编译流水线
    DYNAMO_TRACE_ONCE = 2    # 单次 Dynamo trace，避免重编译
    VLLM_COMPILE = 3         # vLLM 自定义 Inductor 后端（分段编译 + 缓存 + Custom Pass）
```

`CompilationMode` 由启动配置决定。当 `enforce_eager=True` 或 `mode=0` 时，模型以纯 Eager 方式运行，不触发任何 `torch.compile` 编译。

##### 平台级模式控制：`use_direct_call`

**文件**：`vllm/model_executor/layers/attention/attention.py`

```python
# Attention.__init__() 中
self.use_direct_call = not current_platform.opaque_attention_op()
```

| 平台 | `opaque_attention_op()` | `use_direct_call` | 说明 |
|------|------------------------|-------------------|------|
| CUDA | `True` | `False` | 注意力作为不透明 Custom Op，torch.compile 不会 trace 进入 |
| ROCm | `True` | `False` | 同 CUDA |
| XPU | `True` | `False` | 同 CUDA |
| CPU | `False`（默认） | `True` | 直接函数调用，torch.compile 可以 trace 整个注意力逻辑 |

##### Forward 中的双路径分发

`Attention.forward()` 根据 `use_direct_call` 选择不同的调用路径：

```python
# 路径 A: 直接调用（Eager 模式 / CPU 平台）
if self.use_direct_call:
    # 直接调用 Python 函数，torch.compile 可以 trace 进入内部逻辑
    kv_cache_dummy_dep = unified_kv_cache_update(key, value, self.layer_name)
    unified_attention_with_output(query, key, value, output, self.layer_name,
                                  kv_cache_dummy_dep=kv_cache_dummy_dep)

# 路径 B: 不透明 Custom Op（Compile 模式 / CUDA/ROCm/XPU 平台）
else:
    # 通过 torch.ops.vllm.* 调用，torch.compile 将其视为黑盒
    kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
        key, value, self.layer_name)
    torch.ops.vllm.unified_attention_with_output(
        query, key, value, output, self.layer_name,
        kv_cache_dummy_dep=kv_cache_dummy_dep)
```

**为什么需要两条路径**：

- **不透明 Custom Op（路径 B）**：`torch.compile` 将注意力视为单个黑盒算子，不尝试融合或优化其内部逻辑。这避免了编译器对复杂注意力核（FlashAttention CUDA 核等）进行不当优化，同时允许编译器优化注意力之外的线性层、激活函数等
- **直接调用（路径 A）**：在 CPU 等不需要 Custom Op 的平台，torch.compile 可以完整 trace 注意力逻辑，进行端到端优化

##### Custom Op 注册机制

**文件**：`vllm/utils/torch_utils.py` — `direct_register_custom_op()`

```python
def direct_register_custom_op(
    op_name: str,          # 算子名，如 "unified_attention_with_output"
    op_func: Callable,     # 实际执行函数
    mutates_args: list,    # 声明哪些参数会被原地修改
    fake_impl: Callable,   # torch.compile 用的 fake 实现（只返回正确形状的空张量）
    dispatch_key: str,     # 分发到哪个后端（"CUDA"/"CPU" 等）
    tags: tuple,           # 可标记 cudagraph_unsafe
):
    schema_str = infer_schema(op_func, mutates_args=mutates_args)
    vllm_lib.define(op_name + schema_str, tags=tags)     # 1. 定义算子 schema
    vllm_lib.impl(op_name, op_func, dispatch_key=...)    # 2. 注册真实实现
    vllm_lib._register_fake(op_name, fake_impl)          # 3. 注册 fake 实现
```

注册后，`torch.ops.vllm.unified_attention_with_output(...)` 在真实执行时调用 `op_func`，在 `torch.compile` trace 时调用 `fake_impl`（返回正确形状的空张量，仅用于形状推导）。

##### 注意力算子与分段编译

**文件**：`vllm/config/compilation.py`

vLLM 的 `VLLM_COMPILE` 模式（mode=3）使用**分段编译（Piecewise Compilation）**：将模型 FX 图在注意力算子处切分为多段，每段独立编译和 CUDA Graph 捕获。

```python
# 被视为 "分段切割点" 的注意力算子
_attention_ops: ClassVar[list[str]] = [
    "vllm::unified_attention",
    "vllm::unified_attention_with_output",
    "vllm::unified_mla_attention",
    "vllm::unified_mla_attention_with_output",
    "vllm::mamba_mixer2",
    "vllm::mamba_mixer",
    # ... 以及其他 SSM/线性注意力算子
]
```

分段编译流程：

```
模型 FX 图
┌──────────────────────────────────────────────────────────┐
│  Linear → RoPE → [切割] → Attention → [切割] → Linear   │
│  │                         ↑ 不透明 Custom Op              │
│  ▼                         │                               │
│  段1: 可编译 + CUDAGraph   段2: Eager 执行   段3: 可编译    │
└──────────────────────────────────────────────────────────┘
```

- **段1/段3**（线性层、激活等）：由 Inductor 编译优化 + CUDAGraph 捕获
- **段2**（注意力算子）：标记为 `cudagraph_unsafe`，以 Eager 方式执行（因为 batch 大小动态变化，CUDAGraph 需要固定形状）

`splitting_ops` 配置控制切分行为：
- `None`（默认）：使用 `_attention_ops` 作为切分点
- `[]`（空列表）：不切分，适用于全量 CUDAGraph 模式（如 FlashAttention v3 的 `ALWAYS` CG 支持级别）

##### ForwardContext：连接编译图与运行时状态

**文件**：`vllm/forward_context.py`

Custom Op 内部通过 `ForwardContext` 访问当前批次的运行时状态，这是编译模式能正常工作的关键：

```python
@dataclass
class ForwardContext:
    no_compile_layers: dict[str, Any]  # layer_name → Attention 层实例
    attn_metadata: dict[str, AttentionMetadata]  # 注意力元数据
    slot_mapping: dict[str, torch.Tensor]  # KV Cache 写入映射
    # ...

# Custom Op 内部获取上下文
def unified_attention_with_output(query, key, value, output, layer_name, ...):
    forward_context = get_forward_context()
    attn_layer = forward_context.no_compile_layers[layer_name]  # 获取层实例
    attn_metadata = forward_context.attn_metadata                # 获取元数据
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    attn_layer.impl.forward(attn_layer, query, key, value, kv_cache, attn_metadata, output=output)
```

`no_compile_layers` 的设计意义：层实例引用不嵌入编译图中，而是在运行时通过 `layer_name` 字符串查找。这使得：
- 编译图可以在不同 batch 间复用
- 层的权重更新不需要重编译
- 支持 MoE 等动态路由模型的快速冷启动

---

## 4. V1 Backend 抽象层

### 4.1 AttentionBackend 接口

**文件**：`vllm/v1/attention/backend.py`

```python
class AttentionBackend(ABC):
    # 类变量
    accept_output_buffer: bool = False           # 是否接受预分配的输出缓冲区
    supported_dtypes: list[torch.dtype]          # 支持的数据类型
    supported_kv_cache_dtypes: list[CacheDType]  # 支持的 KV Cache 类型
    forward_includes_kv_cache_update: bool = True # forward 是否包含 KV Cache 更新

    # 必须实现的方法
    @abstractmethod
    def get_name() -> str: ...
    @abstractmethod
    def get_impl_cls() -> type[AttentionImpl]: ...
    @abstractmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]: ...
    @abstractmethod
    def get_kv_cache_shape(...) -> tuple: ...

    # 能力检查方法
    def supports_head_size(head_size) -> bool: ...
    def supports_dtype(dtype) -> bool: ...
    def supports_kv_cache_dtype(kv_cache_dtype) -> bool: ...
    def supports_block_size(block_size) -> bool: ...
    def is_mla() -> bool: ...
    def supports_sink() -> bool: ...
    def supports_attn_type(attn_type) -> bool: ...
```

### 4.2 CommonAttentionMetadata

所有 Backend 共享的批次元数据：

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor      # 每个请求的 Query 起始位置 (batch_size+1,)
    seq_lens: torch.Tensor             # 每个请求已计算的 token 数 (batch_size,)
    num_reqs: int                      # 批次中的请求数
    num_actual_tokens: int             # 实际 token 数（不含 padding）
    max_query_len: int                 # 最长 Query 长度
    max_seq_len: int                   # 最长序列长度
    block_table_tensor: torch.Tensor   # KV Cache Block 映射表
    slot_mapping: torch.Tensor         # Token → Cache Slot 映射
    causal: bool = True                # 是否使用因果掩码
```

### 4.3 AttentionMetadataBuilder

负责将 `CommonAttentionMetadata` 转换为 Backend 特定的元数据：

```python
class AttentionMetadataBuilder(ABC, Generic[M]):
    _cudagraph_support: ClassVar[AttentionCGSupport]  # CUDA Graph 支持级别

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:  # 返回 Backend 特定的元数据
        ...
```

### 4.4 AttentionImpl 层次

```python
class AttentionImplBase(ABC):
    """所有实现的基类，处理上下文并行（CP）初始化"""
    dcp_world_size: int   # Decode Context Parallelism
    pcp_world_size: int   # Prefill Context Parallelism

class AttentionImpl(AttentionImplBase):
    """标准注意力实现"""
    @abstractmethod
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None) -> torch.Tensor: ...

class MLAAttentionImpl(AttentionImplBase):
    """MLA 注意力实现，分为 MHA（prefill）和 MQA（decode）两条路径"""
    @abstractmethod
    def forward_mha(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata, ...): ...
    @abstractmethod
    def forward_mqa(self, q, kv_cache, attn_metadata, layer): ...
```

---

## 5. Backend 选择机制

**文件**：`vllm/v1/attention/selector.py`

### 选择流程

```python
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    block_size: int,
    use_mla: bool = False,
    ...
) -> type[AttentionBackend]:
    """根据硬件和模型配置选择最优 Backend"""

    # 1. 验证 kv_cache_dtype
    # 2. 获取 vllm_config
    # 3. 构建 AttentionSelectorConfig
    # 4. 调用平台特定的选择逻辑
    return current_platform.get_attn_backend_cls(backend, config)
```

### 选择标准

| 标准 | 说明 |
|------|------|
| `head_size` | 头维度（不同 Backend 支持的维度不同） |
| `dtype` | 模型数据类型（fp16/bf16） |
| `kv_cache_dtype` | KV Cache 数据类型（auto/fp8/bf16） |
| `block_size` | 页大小（必须是 Backend 支持的倍数） |
| `use_mla` | 是否使用 MLA（选择 MLA 专用 Backend） |
| `compute_capability` | GPU 计算能力（如 SM80+） |
| `attn_type` | 注意力类型（decoder/encoder/cross） |

### Backend 优先级

各平台有独立的后端选择优先级链：

#### NVIDIA CUDA

```
get_attn_backend_cls()
  ├─ MLA 模式:
  │    ├─ SM100 (Blackwell) → CUTLASS MLA → FlashMLA → FlashInfer MLA → Triton MLA
  │    └─ Other             → FlashMLA → FlashInfer MLA → Triton MLA
  └─ 标准模式:
       ├─ SM100 (Blackwell) → FlashInfer → FlashAttention → Triton → FlexAttention
       └─ Other             → FlashAttention → FlashInfer → Triton → FlexAttention
```

#### AMD ROCm

```
get_attn_backend_cls()
  ├─ MLA 模式:
  │    ├─ AITER 可用 → ROCm AITER MLA (Sparse) → ROCm AITER Triton MLA
  │    └─ 默认       → FlashAttn MLA → Triton MLA
  └─ 标准模式:
       ├─ AITER Unified 启用 → ROCm AITER Unified Attn
       ├─ AITER FA 启用      → ROCm AITER FA (仅 gfx9: MI300/MI325)
       ├─ ROCm Attn 启用     → ROCm Attention
       └─ 默认               → Triton Attention
```

#### Intel XPU

```
get_attn_backend_cls()
  ├─ MLA 模式 → Triton MLA (唯一支持)
  └─ 标准模式:
       ├─ fp16/bf16 → FlashAttention (IPEX 实现)
       └─ fp32      → Triton Attention (FlashAttn 不支持 fp32)
```

#### CPU

```
get_attn_backend_cls() → CPU Attention (唯一选项)
```

---

## 6. 主要 Backend 实现

### 6.1 FlashAttention Backend

**文件**：`vllm/v1/attention/backends/flash_attn.py`

FlashAttention 是 vLLM 最重要的 Backend，针对 NVIDIA GPU 做了深度优化。

#### Metadata 结构

```python
@dataclass
class FlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor    # cu_seqlens_q（累积序列长度）
    max_seq_len: int
    seq_lens: torch.Tensor           # 每个请求的 KV 长度
    block_table: torch.Tensor        # Paged attention Block 映射
    slot_mapping: torch.Tensor       # Token → Cache Slot 映射
    causal: bool

    # Cascade Attention（共享前缀优化）
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor
    prefix_kv_lens: torch.Tensor
    suffix_kv_lens: torch.Tensor

    # FA3 AOT 调度
    scheduler_metadata: torch.Tensor  # 预计算的 tile 调度
```

#### Forward 执行流程

```
FlashAttentionImpl.forward()
  │
  ├─ 1. 输入校验与 profiling 处理
  │
  ├─ 2. 按注意力类型路由
  │     ├─ ENCODER/ENCODER_ONLY → _forward_encoder_attention()
  │     └─ DECODER → 继续下一步
  │
  ├─ 3. 提取 KV Cache
  │     key_cache, value_cache = kv_cache.unbind(0)
  │
  ├─ 4. FP8 类型转换（如果需要）
  │     key_cache = key_cache.view(fp8_dtype)
  │
  ├─ 5. 主计算路由
  │     ├─ Cascade 模式 → cascade_attention()
  │     │   # 两次 FA 调用：prefix（非因果）+ suffix（因果）
  │     │
  │     ├─ DCP 模式 → _forward_with_dcp()
  │     │   # All-gather Q → 分布式计算 → Reduce-scatter 输出
  │     │
  │     └─ 标准模式 → flash_attn_varlen_func()
  │           flash_attn_varlen_func(
  │             q=query[:num_actual_tokens],
  │             k=key_cache,              # 从 paged cache 读取
  │             v=value_cache,
  │             cu_seqlens_q=query_start_loc,
  │             seqused_k=seq_lens,       # 每序列 KV 长度
  │             block_table=block_table,  # Paged 映射
  │             causal=True,
  │             softmax_scale=self.scale,
  │             scheduler_metadata=...,   # FA3 AOT 调度
  │           )
  │
  └─ 6. 返回输出
```

#### Cascade Attention 优化

当多个请求共享同一前缀时（如系统提示），cascade attention 将计算分为两步：

1. **Prefix 注意力**：所有请求对共享前缀做非因果注意力（可以复用计算）
2. **Suffix 注意力**：每个请求对自己的后缀做因果注意力
3. **合并**：使用 LogSumExp 技巧合并两部分结果

触发条件：`common_prefix_len > 256` 且有足够的并行度。

#### DCP（Decode Context Parallelism）

在多 GPU 场景下分布式处理长上下文的解码：

```python
def _forward_with_dcp(self, ...):
    # 1. All-gather：收集所有 rank 的 query
    query_across_dcp = get_dcp_group().all_gather(query, dim=1)

    # 2. 计算 context attention（在各 rank 分布的 KV 上）
    context_attn_out, context_lse = flash_attn_varlen_func(...)

    # 3. Reduce-scatter：使用 LSE 加权合并输出
    context_out_cor, context_lse_cor = cp_lse_ag_out_rs(...)

    # 4. 计算 query-only attention（当前 token 的 self-attention）
    query_attn_out, query_lse = flash_attn_varlen_func(...)

    # 5. 合并两部分
    merge_attn_states(output, context_out_cor, context_lse_cor,
                      query_attn_out, query_lse)
```

### 6.2 FlashInfer Backend

**文件**：`vllm/v1/attention/backends/flashinfer.py`

FlashInfer 使用独特的 **Wrapper-based** 设计，将计算分为 `plan()`（规划）和 `run()`（执行）两步。

#### 关键特点

| 特性 | 说明 |
|------|------|
| **双路径** | Prefill 和 Decode 使用独立的 Wrapper 实例 |
| **TRTLLM 集成** | 支持 NVIDIA TRT-LLM 后端作为替代执行路径 |
| **Cascade 注意力** | 使用 `MultiLevelCascadeAttentionWrapper` |
| **Paged KV 索引** | 使用 `paged_kv_indptr/indices/last_page_len` 而非 block_table |

#### Metadata 结构

```python
@dataclass
class FlashInferMetadata:
    num_actual_tokens: int
    slot_mapping: torch.Tensor
    q_data_type: torch.dtype

    # 双路径设计
    prefill: FIPrefill | TRTLLMPrefill | None   # Prefill 路径
    decode: FIDecode | TRTLLMDecode | None       # Decode 路径

    # Cascade
    use_cascade: bool
    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None
```

#### 与 FlashAttention 的差异

- FlashInfer 需要预先 `plan()` 计算内存和调度
- 支持更灵活的页式 KV 索引格式
- 可切换 FlashInfer native 和 TRT-LLM 执行路径
- Decode 阶段有专门的 GQA 优化 Wrapper

### 6.3 Triton Backend

**文件**：`vllm/v1/attention/backends/triton_attn.py`

Triton Backend 使用纯 Triton 编写的注意力核，具有最好的可移植性。

#### 关键特点

| 特性 | 说明 |
|------|------|
| **统一核** | 单个 `unified_attention()` 处理所有情况 |
| **自适应核选择** | 根据 batch 大小在 2D 和 3D 核之间切换 |
| **编码器支持** | 完整的双向注意力支持 |
| **灵活头维度** | 支持 head_size ≥ 32（比 FlashAttention 更灵活） |
| **Softmax 分块** | 使用预分配的 segmentation 缓冲区做 tiled softmax |

#### Metadata 结构

```python
@dataclass
class TritonAttentionMetadata:
    # 标准字段
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # Triton 特有：核调优
    seq_threshold_3D: int                # 2D/3D 核切换阈值
    num_par_softmax_segments: int        # 分块 softmax 段数
    softmax_segm_output: torch.Tensor    # 预分配的中间缓冲区
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor
```

### Backend 对比

| 维度 | FlashAttention | FlashInfer | Triton | ROCm Attn | ROCm AITER | CPU Attn |
|------|---------------|-----------|--------|-----------|-----------|----------|
| **平台** | CUDA, ROCm, XPU | CUDA | ROCm, XPU | ROCm | ROCm (gfx9) | CPU |
| **核策略** | 单核 varlen 接口 | 预规划 Wrapper | 统一自适应核 | Prefill/Decode 分离 | AITER 融合核 | PyTorch Ops |
| **Prefill/Decode** | 相同核（FA3）/ 分开（FA2） | 独立 Wrapper | 同一核 | PagedAttn + Triton | 统一核 | 统一 |
| **编码器支持** | 有 | 仅 Decoder | 完整支持 | 仅 Decoder | 仅 Decoder | 有 |
| **头维度** | 固定集合 | 64/128/256 | ≥32 | 同 FA | 同 FA | 任意 |
| **FP8 路径** | 原生 descale | TRT-LLM 解量化核 | 按序列 descale | 无 | 原生 | 无 |
| **CUDA Graph** | FA3 全支持 | 每 batch 大小独立 | 预分配缓冲区 | 支持 | 支持 | 不适用 |
| **最佳场景** | 大 batch NVIDIA GPU | 分布式/TRT-LLM | 编码器/小 batch | AMD MI 系列 | MI300/325 高性能 | CPU 推理 |

### 各平台 Kernel 实现概览

| 平台 | 主力 Kernel | 备选 Kernel | MLA Kernel |
|------|-----------|-----------|------------|
| **CUDA (Ampere, SM80+)** | FlashAttention v2 (C++/CUDA) | FlashInfer, Triton (Python) | FlashMLA, FlashInfer MLA, Triton MLA |
| **CUDA (Hopper, SM90)** | FlashAttention v3 (Hopper TMA) | FlashInfer, Triton | 同上 |
| **CUDA (Blackwell, SM100)** | FlashInfer (优先) | FlashAttention v3, Triton | CUTLASS MLA (优先), FlashMLA, FlashInfer MLA |
| **ROCm (gfx9: MI300/325)** | ROCm AITER FA / Unified (aiter_ops) | Triton Attention, FlashAttention | ROCm AITER MLA, FlashAttn MLA, Triton MLA |
| **ROCm (gfx1x: RDNA3/4)** | Triton Attention | FlashAttention (Triton 后端) | Triton MLA |
| **XPU (Intel)** | FlashAttention (IPEX 实现) | Triton Attention (fp32 回退) | Triton MLA (唯一) |
| **CPU** | CPU Attention (PyTorch ops) | 无 | 不支持 |

---

## 7. MLA (Multi-Head Latent Attention)

### 7.1 原理

MLA 是 DeepSeek-V2/V3 提出的注意力机制，通过将 KV 压缩到一个**低秩潜在空间**，大幅减少 KV Cache 内存占用。

#### 与标准 MHA/GQA 对比

| 特性 | MHA | GQA | MLA |
|------|-----|-----|-----|
| KV 头数 | N | N/G | **1 个潜在维度** |
| KV 表示 | 完整 (head_size × N) | 分组 | **压缩 (kv_lora_rank)** |
| 内存节省 | 基线 | ~G 倍 | **~64-100 倍** |
| 计算策略 | 统一 | 统一 | **双路径**（MHA prefill + MQA decode） |

#### 核心思想

```
原始 hidden_states
    │
    ├─ → W_DKV → kv_c (512 维)        # 压缩的 KV 潜在表示
    │              │
    │              └─ LayerNorm → kv_c_normed → [存入 KV Cache]
    │
    ├─ → W_KR → k_pe (64 维)          # 分离的位置编码
    │              │
    │              └─ RoPE → [存入 KV Cache]
    │
    └─ → W_DQ/W_Q → q                 # Query（标准或压缩）

KV Cache 存储: [kv_c_normed (512) | k_pe (64)] = 576 维
标准 MHA 存储: num_heads × head_size × 2 ≈ 8192+ 维
```

### 7.2 MLAAttention 实现

**文件**：`vllm/model_executor/layers/attention/mla_attention.py`

```python
class MLAAttention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,    # 非 RoPE QK 维度（128）
        qk_rope_head_dim: int,    # RoPE QK 维度（64）
        v_head_dim: int,          # Value 头维度（128）
        kv_lora_rank: int,        # KV 潜在秩（512）
        kv_b_proj: ColumnParallelLinear,  # KV 解压权重
        ...
    ):
```

### 7.3 双路径计算

MLA 根据 Query 长度选择不同的计算路径：

#### Prefill 路径（MHA 风格，计算友好）

当 `Sq ≈ Skv` 时（prefill 阶段），在全维度空间计算注意力：

```python
# 1. 解压 KV 到全维度
k_nope = kv_c @ W_UK    # (Skv, N, qk_nope_head_dim)
v = kv_c @ W_UV          # (Skv, N, v_head_dim)

# 2. 拼接位置编码
k = concat([k_nope, k_pe.expand(N)], dim=-1)  # (Skv, N, qk_head_dim)

# 3. 标准 MHA 注意力
output = scaled_dot_product_attention(q, k, v)  # O(N·Sq·Skv)
```

#### Decode 路径（MQA 风格，数据移动友好）

当 `Sq = 1, Skv >> 1` 时（decode 阶段），在潜在空间计算：

```python
# 1. 将 Query 投影到潜在空间
ql_nope = q_nope @ W_UK.T   # (1, N, kv_lora_rank) — 在低维空间

# 2. MQA 风格注意力（KV 只有 1 个头）
q_latent = concat([ql_nope, q_pe], dim=-1)     # (1, N, kv_lora_rank + rope_dim)
kv_combined = concat([kv_c, k_pe], dim=-1)     # (Skv, 1, kv_lora_rank + rope_dim)
attn_out = attention(q_latent, kv_combined, kv_c)  # (1, N, kv_lora_rank)

# 3. 将输出从潜在空间投影回来
output = attn_out @ W_UV    # (1, N, v_head_dim)
```

**为什么 Decode 使用 MQA**：
- 避免对每个 decode token 做昂贵的全 KV 展开
- 在潜在空间计算减少数据移动
- KV 只有 1 个头（vs N 个），内存带宽需求降低 N 倍

### 7.4 KV Cache 结构

```python
# MLA 的 KV Cache 形状
shape = (num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
# 例如 DeepSeek-V3: (num_blocks, block_size, 576)

# 对比标准 MHA
shape = (2, num_blocks, block_size, num_kv_heads, head_size)
# 例如 LLaMA-70B: (2, num_blocks, block_size, 8, 128)
```

### 7.5 MLA Backend 矩阵

| Backend | 文件 | 支持平台 | 特点 |
|---------|------|---------|------|
| FlashMLA | `mla/flashmla.py` | CUDA | 专用 MLA CUDA 核，decode 优化 |
| FlashMLA Sparse | `mla/flashmla_sparse.py` | CUDA | FlashMLA + 稀疏注意力（block size=1） |
| FlashInfer MLA | `mla/flashinfer_mla.py` | CUDA | FlashInfer 的 MLA 路径，plan/run 模式 |
| FlashInfer MLA Sparse | `mla/flashinfer_mla_sparse.py` | CUDA | FlashInfer MLA + 稀疏注意力 |
| CUTLASS MLA | `mla/cutlass_mla.py` | CUDA (SM100) | NVIDIA Blackwell 专用，CUTLASS FMHA |
| FlashAttn MLA | `mla/flashattn_mla.py` | CUDA, ROCm | 基于 FlashAttention 的 MLA 实现 |
| Triton MLA | `mla/triton_mla.py` | ROCm, XPU, CPU | 纯 Triton 编写，跨平台可移植 |
| ROCm AITER MLA | `mla/rocm_aiter_mla.py` | ROCm (gfx9) | AMD AITER ops 专用 MLA 核 |
| ROCm AITER MLA Sparse | `mla/rocm_aiter_mla_sparse.py` | ROCm (gfx9) | AITER MLA + 稀疏注意力 |
| ROCm AITER Triton MLA | `mla/aiter_triton_mla.py` | ROCm | AITER + Triton 混合 MLA |

---

## 8. Paged Attention 与 KV Cache

### 8.1 Paged Attention 原理

Paged Attention 是 vLLM 的核心创新，将 KV Cache 的内存管理类比为操作系统的虚拟内存分页：

```
物理内存（KV Cache Pool）
┌────────┬────────┬────────┬────────┬────────┐
│ Block 0│ Block 1│ Block 2│ Block 3│ Block 4│
│ 16 tok │ 16 tok │ 16 tok │ 16 tok │ 16 tok │
└────────┴────────┴────────┴────────┴────────┘
    ↑         ↑         ↑         ↑         ↑
    │         │         │         │         │
Block Table（虚拟 → 物理映射）
┌─────────────────────────────────┐
│ Seq 0: [0, 3]       → 32 tokens│
│ Seq 1: [1, 4]       → 32 tokens│
│ Seq 2: [2]           → 16 tokens│
└─────────────────────────────────┘
```

**优势**：
- **消除内存碎片**：所有 block 大小固定，可以任意分配和回收
- **按需分配**：请求只在需要时获取新 block，不需要预分配最大长度
- **内存共享**：共享前缀的请求可以共享 block（copy-on-write）

### 8.2 KV Cache 内存布局

#### 标准 Attention

```python
# 物理 Cache 形状
kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
#          ↑                ↑
#          K和V两层         每个 block 存储 block_size 个 token
#
# 分离为：
key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
value_cache: [num_blocks, block_size, num_kv_heads, head_size]
```

#### CUDA 核中的 Key Cache 布局（用于向量化加载）

```cpp
// 转置布局，优化内存访问模式
key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
//                                     ↑                        ↑
//                               head 维度分块          x = 16/sizeof(dtype)
//
// 这样每个线程组可以连续加载一个 head_size/x 的片段
```

### 8.3 Slot Mapping

`slot_mapping` 将每个新 token 映射到 KV Cache 中的精确位置：

```python
# 示例：block_size = 4
# Seq 0 有 5 个 token，分配了 block [0, 3]
# Seq 1 有 3 个 token，分配了 block [1]

slot_mapping = [
    0, 1, 2, 3,    # Seq 0 前 4 个 token → block 0 的 slot 0-3
    12,             # Seq 0 第 5 个 token → block 3 的 slot 0 (3*4=12)
    4, 5, 6,        # Seq 1 的 3 个 token → block 1 的 slot 0-2 (1*4=4)
]
```

### 8.4 KV Cache 写入

```python
def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping,
                            kv_cache_dtype, k_scale, v_scale):
    """
    将新 token 的 K/V 写入 paged cache

    key:          [num_tokens, num_kv_heads, head_size]
    value:        [num_tokens, num_kv_heads, head_size]
    key_cache:    [num_blocks, block_size, num_kv_heads, head_size]
    value_cache:  [num_blocks, block_size, num_kv_heads, head_size]
    slot_mapping: [num_tokens] — 每个 token 的目标 slot
    """
    # 对于 FP8：先量化再写入
    # key_quantized = key * k_scale
    # 写入 key_cache[slot_mapping[i]] = key_quantized[i]
```

### 8.5 CUDA 核实现

**文件**：`csrc/attention/attention_kernels.cuh`

Paged Attention CUDA 核的执行分为三个阶段：

```
阶段 1: QK 点积计算
┌─────────────────────────────────────────┐
│ for each block in block_table:          │
│   physical_block = block_table[block_idx]│
│   for each token in block:              │
│     load K from k_cache[physical_block] │
│     qk = Q · K * scale                 │
│     qk += alibi_slope * position        │
│     logits[token] = qk                  │
│   track max_logit for numerical stability│
└─────────────────────────────────────────┘
           │
           ▼
阶段 2: Softmax
┌─────────────────────────────────────────┐
│ global_max = reduce_max(qk_max)         │
│ for each token:                         │
│   logits[i] = exp(logits[i] - global_max)│
│ exp_sum = reduce_sum(logits)            │
│ for each token:                         │
│   logits[i] /= exp_sum                  │
└─────────────────────────────────────────┘
           │
           ▼
阶段 3: AV（Attention × Value）
┌─────────────────────────────────────────┐
│ for each block in block_table:          │
│   load V from v_cache[physical_block]   │
│   for each token in block:              │
│     output += logits[token] * V[token]  │
│ warp_reduce(output)                     │
└─────────────────────────────────────────┘
```

### 8.6 KV Cache 规格系统

**文件**：`vllm/v1/kv_cache_interface.py`

```python
class KVCacheSpec(ABC):
    """KV Cache 规格基类"""
    block_size: int

class FullAttentionSpec(AttentionSpec):
    """完整注意力的 Cache 规格"""
    # 内存 = ⌈max_model_len / block_size⌉ × page_size
    # page_size = 2 × block_size × num_kv_heads × head_size × dtype_size

class SlidingWindowSpec(AttentionSpec):
    """滑动窗口 Cache 规格"""
    sliding_window: int
    # 只缓存 sliding_window 个 token，大幅减少内存

class MLAAttentionSpec(FullAttentionSpec):
    """MLA 的 Cache 规格"""
    # page_size = block_size × (kv_lora_rank + qk_rope_head_dim) × dtype_size
    # 只有标准 MHA 的 1/10 左右

class MambaSpec(KVCacheSpec):
    """Mamba 的状态 Cache 规格"""
    # 固定大小的状态向量，不随序列长度增长
```

### 8.7 Attention 状态合并

**文件**：`vllm/v1/attention/ops/merge_attn_states.py`

当注意力分为多部分计算时（cascade、DCP、推测解码），需要合并部分结果：

```python
def merge_attn_states(output, prefix_output, prefix_lse,
                      suffix_output, suffix_lse):
    """
    使用 LogSumExp 技巧合并两部分注意力输出

    公式：
    max_lse = max(prefix_lse, suffix_lse)
    output = (exp(prefix_lse - max_lse) * prefix_output +
              exp(suffix_lse - max_lse) * suffix_output) /
             (exp(prefix_lse - max_lse) + exp(suffix_lse - max_lse))
    """
```

---

## 9. 特殊 Attention 变体

### 9.1 Cross Attention（交叉注意力）

**用途**：编码器-解码器模型（T5、BART 等）

**特点**：
- 非因果注意力，解码器可以看到所有编码器位置
- 使用独立的编码器 KV Cache
- 自定义 slot mapping 处理编码器序列

### 9.2 Encoder-Only Attention（编码器注意力）

**用途**：BERT、RoBERTa 等编码器模型

**特点**：
- 双向注意力（非因果）
- **不需要 KV Cache**（`get_kv_cache_spec()` 返回 None）
- 最简单的注意力变体

### 9.3 Chunked Local Attention（分块局部注意力）

**用途**：长序列场景下的局部注意力窗口

**特点**：
- 注意力仅在固定大小的 chunk 内计算
- 创建虚拟 batch 来重组 block table
- 不支持 CUDA Graph

### 9.4 Static Sink Attention（Sink Token 注意力）

**用途**：StreamingLLM 风格的推理，保留初始 token

**特点**：
- 前 `sink_len` 个 token 永久保留在 Cache 中
- Block table 头部是 sink block，后面是滑动窗口
- 使用 Triton 核执行 sink KV 的填充

### 9.5 Mamba/SSM Attention

**用途**：Mamba 模型（线性时间的 Transformer 替代方案）

**特点**：
- 不使用 Q/K/V 投影，而是线性递推处理
- 状态大小固定，不随序列增长
- Mamba2 使用 chunk-aligned 计算

### 9.6 Linear Attention（线性注意力）

**用途**：线性时间注意力机制

**特点**：
- O(n) 复杂度（vs 标准 O(n²)）
- 使用状态索引代替 slot mapping
- 元数据最简化

---

## 10. 完整调用流程

### 10.1 从请求到注意力计算的完整链路

```
客户端请求
    │
    ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Scheduler（调度器）
    vllm/v1/core/sched/scheduler.py
    │
    ├─ 选择要执行的请求
    ├─ 分配 KV Cache Block（通过 BlockTable）
    └─ 输出 SchedulerOutput
         ├─ num_scheduled_tokens
         ├─ block_table
         └─ slot_mapping
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ModelRunner（模型运行器）
    vllm/v1/worker/gpu_model_runner.py
    │
    ├─ execute_model(scheduler_output)
    │
    ├─ 1. 构建 CommonAttentionMetadata
    │     CommonAttentionMetadata(
    │       query_start_loc = [...],
    │       seq_lens = [...],
    │       block_table_tensor = [...],
    │       slot_mapping = [...],
    │     )
    │
    ├─ 2. 调用 Backend Builder 构建特定元数据
    │     attn_metadata = builder.build(
    │       common_prefix_len,
    │       common_attn_metadata
    │     )
    │     # → FlashAttentionMetadata / FlashInferMetadata / ...
    │
    ├─ 3. 设置 ForwardContext
    │     set_forward_context(
    │       attn_metadata,     # 注意力元数据
    │       slot_mapping,      # KV Cache 写入映射
    │     )
    │
    └─ 4. 执行模型前向传播
          model_forward(input_ids, positions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    模型前向传播
    例如 vllm/model_executor/models/llama.py
    │
    LlamaForCausalLM.forward()
      └─ for layer in layers:
           LlamaDecoderLayer.forward()
             └─ LlamaAttention.forward()
                  │
                  ├─ 1. QKV 投影
                  │     qkv = self.qkv_proj(hidden_states)
                  │     q, k, v = split(qkv)
                  │
                  ├─ 2. 应用旋转位置编码 (RoPE)
                  │     q, k = self.rotary_emb(positions, q, k)
                  │
                  └─ 3. 调用 Attention 层
                        attn_output = self.attn(q, k, v)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Attention Layer
    vllm/model_executor/layers/attention/attention.py
    │
    Attention.forward(q, k, v)
      │
      ├─ 1. FP8 量化处理（可选）
      │
      ├─ 2. 重塑张量维度
      │     q: [tokens, heads*dim] → [tokens, heads, dim]
      │
      ├─ 3. 更新 KV Cache
      │     unified_kv_cache_update(k, v, layer_name)
      │       ↓
      │     get_attention_context(layer_name)
      │       ├─ ForwardContext → attn_metadata
      │       ├─ kv_cache = layer.kv_cache[engine]
      │       └─ slot_mapping
      │     impl.do_kv_cache_update(layer, k, v, kv_cache, slot_mapping)
      │       ↓
      │     reshape_and_cache_flash(k, v, key_cache, value_cache,
      │                             slot_mapping, dtype, k_scale, v_scale)
      │
      └─ 4. 执行注意力计算
            unified_attention_with_output(q, k, v, output, layer_name)
              ↓
            get_attention_context(layer_name)
            impl.forward(layer, q, k, v, kv_cache, attn_metadata, output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Backend 执行
    例如 FlashAttentionImpl.forward()
    │
    ├─ 提取 Key/Value Cache
    │   key_cache, value_cache = kv_cache.unbind(0)
    │
    ├─ 调用 CUDA 核
    │   flash_attn_varlen_func(
    │     q = query[:num_actual_tokens],
    │     k = key_cache,           # Paged KV Cache
    │     v = value_cache,
    │     block_table = block_table,  # 虚拟 → 物理块映射
    │     cu_seqlens_q = query_start_loc,
    │     seqused_k = seq_lens,
    │     causal = True,
    │     softmax_scale = 1/√d,
    │   )
    │
    └─ 返回 output: [num_tokens, num_heads, head_size]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
    Output → O Projection → 残差连接 → 下一层
```

### 10.2 ForwardContext：连接调度器和注意力层的桥梁

```python
# ModelRunner 设置 Context
with set_forward_context(attn_metadata, vllm_config, slot_mapping=...):
    model.forward(input_ids, positions)  # 模型前向

# Attention 层读取 Context
def unified_attention(query, key, value, layer_name):
    ctx = get_forward_context()              # 获取当前 Context
    attn_metadata = ctx.attn_metadata        # 注意力元数据
    layer = ctx.no_compile_layers[layer_name] # 该层的 Attention 实例
    kv_cache = layer.kv_cache[ctx.virtual_engine]  # KV Cache
    slot_mapping = ctx.slot_mapping.get(layer_name) # Slot 映射
    return layer.impl.forward(layer, query, key, value, kv_cache, attn_metadata)
```

### 10.3 KV Cache 生命周期

```
1. 系统启动
   │
   ├─ 计算 GPU 可用内存
   ├─ 根据 KVCacheSpec 计算每层 page 大小
   ├─ 分配物理 Block 池: [2, total_blocks, block_size, num_kv_heads, head_size]
   └─ 绑定到每层 Attention.kv_cache

2. 请求到达
   │
   ├─ Scheduler 分配 Block Table: {seq_id: [block_0, block_1, ...]}
   └─ 计算 Slot Mapping: [token_0 → slot_X, token_1 → slot_Y, ...]

3. Prefill 阶段
   │
   ├─ 所有 prefill token 的 K/V 写入分配的 slots
   └─ 注意力在所有 prefill token 上计算（因果掩码）

4. Decode 阶段（逐 token）
   │
   ├─ 新 token 的 K/V 写入下一个 slot
   ├─ 如果当前 block 已满，分配新 block
   └─ 注意力在所有历史 token（通过 block table 访问）上计算

5. 请求完成
   │
   └─ 释放所有分配的 blocks 回池中
```

---

## 11. 性能优化技术

### 11.1 KV Cache 量化（FP8）

```python
# 首次 forward 时计算缩放因子
def calc_kv_scales(self, query, key, value):
    self._q_scale = max(|query|) / fp8_max
    self._k_scale = max(|key|) / fp8_max
    self._v_scale = max(|value|) / fp8_max
    self.calculate_kv_scales = False  # 只计算一次

# 写入 Cache 时量化
key_fp8 = key * k_scale  # FP16 → FP8
value_fp8 = value * v_scale

# 读取时反量化（在核内融合）
key_restored = key_fp8 / k_scale  # FP8 → FP16
```

### 11.2 CUDA Graph 支持

不同 Backend 的 CUDA Graph 支持级别：

```python
class AttentionCGSupport(Enum):
    ALWAYS = 3                        # 支持混合 prefill-decode
    UNIFORM_BATCH = 2                 # 仅均匀 batch
    UNIFORM_SINGLE_TOKEN_DECODE = 1   # 仅单 token decode
    NEVER = 0                         # 不支持
```

- **FA3**：`ALWAYS` — 最完整的 CUDA Graph 支持
- **FA2**：`UNIFORM_BATCH` — 需要 batch 内 query 长度相同
- **Triton**：通过预分配缓冲区支持
- **ChunkedLocalAttention**：不支持 CUDA Graph

### 11.3 Cascade Attention

针对共享前缀的场景优化：

```
普通模式：每个请求独立计算注意力
Request 0: [System Prompt | User A's message] → 完整注意力
Request 1: [System Prompt | User B's message] → 完整注意力
                ↑ 重复计算

Cascade 模式：分离共享前缀
Step 1: Prefix Attention（非因果，所有请求共享）
  [System Prompt] → 计算一次

Step 2: Suffix Attention（因果，每个请求独立）
  Request 0: [User A's message]
  Request 1: [User B's message]

Step 3: Merge（使用 LogSumExp）
  output = merge(prefix_output, suffix_output)
```

### 11.4 AOT Scheduling（FA3）

FlashAttention v3 支持提前计算 tile 调度，减少核启动开销：

```python
scheduler_metadata = get_scheduler_metadata(
    batch_size=num_reqs,
    max_seqlen_q=max_query_len,
    max_seqlen_k=max_seq_len,
    num_heads_q=num_heads,
    num_heads_kv=num_kv_heads,
    ...
)
# 预计算每个 tile 的分配方案，避免核内动态调度
```

### 11.5 Context Parallelism

vLLM 支持两种上下文并行方式：

- **DCP (Decode Context Parallelism)**：将 KV Cache 分布到多个 GPU，解码时 all-gather query 并 reduce-scatter 输出
- **PCP (Prefill Context Parallelism)**：将 prefill 的 KV 分布到多个 GPU 上并行计算

### 11.6 融合操作

- **Fused RoPE + KV Cache Update**：将旋转位置编码和 KV Cache 写入融合为单个核
- **Fused Output Quantization**：将注意力输出和后续量化融合
- **Query Quantization**：在 `torch.compile` 中将 query 量化融合到前序操作

---

## 12. 关键发现与总结

### 12.1 架构设计亮点

1. **多层抽象设计**：通过 `AttentionLayerBase` → `AttentionBackend` → `AttentionImpl` 三层抽象，实现了模型代码与硬件优化的完全解耦

2. **Custom Ops 机制**：通过 `torch.ops.vllm` 注册自定义算子，使注意力计算与 `torch.compile` 完全兼容，同时保留了直接调用的快速路径

3. **ForwardContext 设计**：使用线程局部的 `ForwardContext` 作为调度器到注意力层的数据通道，避免了大量参数传递

4. **KV Cache 规格系统**：通过 `KVCacheSpec` 层次结构，统一管理不同注意力类型的内存需求计算

### 12.2 Paged Attention 的工程实现

1. **Block Table + Slot Mapping 双层映射**：Block Table 负责序列级的块映射，Slot Mapping 负责 token 级的精确定位

2. **CUDA 核优化**：Key Cache 使用转置布局（head 维度分块）优化内存访问模式，Value Cache 使用行优先布局

3. **V1 和 V2 实现**：V1 适合短序列（单 pass），V2 适合长序列（分区 reduce）

### 12.3 MLA 的突破性创新

1. **KV Cache 压缩率**：从 8192+ 维压缩到 576 维（~14× 压缩），使 DeepSeek-V3 的长上下文推理成为可能

2. **双路径策略**：Prefill 使用 MHA（计算友好），Decode 使用 MQA（内存友好），根据场景自动切换

3. **权重预处理**：`W_UK` 和 `W_UV` 在模型加载后预处理为转置形式，避免运行时转置开销

### 12.4 Backend 生态

vLLM 支持的 Backend 生态极其丰富：

| 类别 | Backend | 主要用途 |
|------|---------|---------|
| 标准 | FlashAttention v2/v3 | NVIDIA GPU 主力 |
| 标准 | FlashInfer | TRT-LLM 集成 |
| 标准 | Triton | 可移植性/编码器模型 |
| MLA | FlashMLA/Triton MLA | DeepSeek 模型 |
| SSM | Mamba/Mamba2 | 线性时间模型 |
| 线性 | Linear Attention | O(n) 注意力 |
| 特殊 | Tree Attention | 推测解码 |
| 硬件 | ROCm/CPU | AMD/CPU 支持 |

### 12.5 代码质量观察

1. **文档完善**：核心文件（如 `mla_attention.py`）包含详细的数学公式和算法说明
2. **类型标注**：全面使用 Python 类型标注和泛型
3. **测试覆盖**：每个 Backend 都有对应的单元测试
4. **配置灵活**：通过 `VllmConfig` 集中管理所有配置，支持运行时动态选择
5. **性能意识**：大量使用 `@cache` 装饰器、预分配缓冲区、融合操作等优化手段

### 12.6 潜在改进方向

1. **Legacy 代码清理**：`model_executor/layers/attention/` 和 `v1/attention/` 存在一定的代码重复
2. **Backend 统一**：不同 Backend 的元数据结构差异较大，增加了维护复杂度
3. **文档分散**：架构文档分散在代码注释中，缺少集中的设计文档
