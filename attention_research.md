# vLLM Attention 模块深度研究报告

## 目录

1. [概述](#1-概述)
   - 1.1 [各 Attention 组件之间的关系](#各-attention-组件之间的关系)
2. [架构总览](#2-架构总览)
3. [核心 Attention 层](#3-核心-attention-层)
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
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  FlashAttention v2/v3 — NVIDIA GPU 主力后端，IO-aware 融合核       │
│  FlashInfer — Wrapper-based plan/run 模式，支持 TRT-LLM            │
│  Triton — 纯 Triton 编写，可移植性最好                              │
│  CPU — CPU 回退实现                                                 │
│  ROCm — AMD GPU 专用                                               │
│                                                                     │
│  MLA 专用后端:                                                      │
│  ├── FlashMLA — 专用 CUDA 核                                       │
│  ├── FlashInfer MLA / Triton MLA / CUTLASS MLA                     │
│  └── ROCm MLA                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

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
   get_attn_backend()
     ├─ NVIDIA Hopper+ → FlashAttention v3 (优先)
     ├─ NVIDIA Ampere+  → FlashAttention v2 → FlashInfer → Triton (按优先级回退)
     ├─ AMD GPU         → ROCm Backend
     └─ CPU             → CPU Backend
   ```

3. **MLA 是独立的注意力类型**，拥有自己的 `MLAAttention` 层和专用后端（FlashMLA、Triton MLA 等）。它不通过标准 `Attention` 类，而是有独立的前向路径（`forward_mha` + `forward_mqa`），但同样使用 Paged Attention 管理 KV Cache（只是 cache 的形状不同——576 维 vs 标准 MHA 的数千维）。

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
              │       │    ┌───────────┘           │
              │  ┌────┴────┴──────────────────┐    │
              │  │ 特殊变体:                  │    │
              │  │ CrossAttn / EncoderOnly /  │    │
              │  │ ChunkedLocal / StaticSink  │    │
              │  └────────────────────────────┘    │
              └───────────────┬───────────────────┘
                              │ 委托给
                              ▼
              ┌───────────────────────────────────┐
              │         Backend 计算层             │
              │  ┌──────────┬──────────┬────────┐ │
              │  │FlashAttn │FlashInfer│ Triton │ │
              │  │ (v2/v3)  │         │        │ │
              │  └────┬─────┴────┬─────┴───┬────┘ │
              │       │  MLA后端 │         │      │
              │       │FlashMLA  │         │      │
              │       │TritonMLA │         │      │
              └───────┴──────────┴─────────┴──────┘
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
└── csrc/attention/                          # CUDA 内核
    ├── attention_kernels.cuh                # 核心 CUDA 内核
    ├── paged_attention_v1.cu                # Paged Attention V1
    ├── paged_attention_v2.cu                # Paged Attention V2
    └── merge_attn_states.cu                 # 状态合并 CUDA 内核
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
6. **设置量化参数** — 初始化 FP8 Q/K/V 缩放因子

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

**执行流程**：

```
forward()
  │
  ├─ 1. 计算 KV 缩放因子（FP8 量化，仅首次调用）
  │     torch.ops.vllm.maybe_calc_kv_scales(query, key, value)
  │
  ├─ 2. 量化 Query（如果启用 FP8）
  │     query, _ = self.query_quant(query, self._q_scale)
  │
  ├─ 3. 重塑张量维度
  │     query: [num_tokens, num_heads*head_size] → [num_tokens, num_heads, head_size]
  │     key:   [num_tokens, num_kv_heads*head_size] → [num_tokens, num_kv_heads, head_size]
  │     value: [num_tokens, num_kv_heads*head_size] → [num_tokens, num_kv_heads, head_size]
  │
  ├─ 4. 更新 KV Cache（如果 Backend 不包含 KV Cache 更新）
  │     unified_kv_cache_update(key, value, layer_name)
  │
  └─ 5. 执行注意力计算
        unified_attention_with_output(query, key, value, output, layer_name)
```

#### Custom Ops 机制

vLLM 将注意力计算注册为 PyTorch Custom Ops，使其与 `torch.compile` 兼容：

```python
# 注册为 Custom Op
@torch.library.custom_op("vllm::unified_attention_with_output", mutates_args=["output"])
def unified_attention_with_output(query, key, value, output, layer_name, ...):
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=output)

@torch.library.custom_op("vllm::unified_kv_cache_update", mutates_args=[])
def unified_kv_cache_update(key, value, layer_name):
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    attn_layer.impl.do_kv_cache_update(attn_layer, key, value, kv_cache, slot_mapping)
```

这里的关键是 `get_attention_context(layer_name)` 函数，它从 `ForwardContext` 中提取当前批次的注意力元数据、KV Cache 和 slot 映射。

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

平台特定（NVIDIA 为例）：
1. 用户指定 → 直接使用
2. MLA → FlashMLA / FlashInfer MLA / Triton MLA
3. FlashAttention v3（Hopper 架构）→ FlashAttention v2 → FlashInfer → Triton

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

| 维度 | FlashAttention | FlashInfer | Triton |
|------|---------------|-----------|--------|
| **核策略** | 单核 varlen 接口 | 预规划 Wrapper | 统一自适应核 |
| **Prefill/Decode** | 相同核（FA3）/ 分开（FA2） | 独立 Wrapper | 同一核 |
| **编码器支持** | 有 | 仅 Decoder | 完整支持 |
| **头维度** | 固定集合 | 64/128/256 | ≥32 |
| **FP8 路径** | 原生 descale | TRT-LLM 解量化核 | 按序列 descale |
| **CUDA Graph** | FA3 全支持 | 每 batch 大小独立 | 预分配缓冲区 |
| **最佳场景** | 大 batch NVIDIA GPU | 分布式/TRT-LLM | 编码器/小 batch |

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

| Backend | 文件 | 特点 |
|---------|------|------|
| FlashMLA | `mla/flashmla.py` | 专用 MLA CUDA 核 |
| FlashInfer MLA | `mla/flashinfer_mla.py` | FlashInfer 的 MLA 路径 |
| Triton MLA | `mla/triton_mla.py` | Triton 编写的 MLA 核 |
| CUTLASS MLA | `mla/cutlass_mla.py` | NVIDIA SM100 专用 |
| ROCm MLA | `mla/rocm_aiter_mla.py` | AMD GPU 支持 |

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
