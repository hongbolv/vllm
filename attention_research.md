# vLLM Attention Kernel 深度研究报告

## 目录

1. [项目概述与Attention架构总览](#1-项目概述与attention架构总览)
2. [核心抽象层设计](#2-核心抽象层设计)
3. [Backend选择机制](#3-backend选择机制)
4. [标准Attention后端详解](#4-标准attention后端详解)
5. [MLA（Multi-Head Latent Attention）后端详解](#5-mlamulti-head-latent-attention后端详解)
6. [特殊Attention后端详解](#6-特殊attention后端详解)
7. [CUDA/C++ 底层内核分析](#7-cudac-底层内核分析)
8. [Attention操作层（Ops）](#8-attention操作层ops)
9. [完整调用流程](#9-完整调用流程)
10. [配置系统与环境变量](#10-配置系统与环境变量)
11. [各Backend对比矩阵](#11-各backend对比矩阵)
12. [关键设计模式与发现](#12-关键设计模式与发现)
13. [总结](#13-总结)

---

## 1. 项目概述与Attention架构总览

### 1.1 目录结构

vLLM的Attention系统分布在以下核心目录中：

```
vllm/
├── v1/attention/                          # v1 新统一Attention框架
│   ├── backend.py                         # 核心抽象类定义
│   ├── selector.py                        # Backend选择逻辑
│   ├── backends/                          # 所有Backend实现
│   │   ├── registry.py                    # Backend枚举与注册
│   │   ├── utils.py                       # 共享工具函数
│   │   ├── fa_utils.py                    # Flash Attention工具
│   │   ├── flash_attn.py                  # FlashAttention v2/3/4
│   │   ├── flash_attn_diffkv.py           # 不同KV维度的FlashAttention
│   │   ├── flashinfer.py                  # FlashInfer后端
│   │   ├── flex_attention.py              # PyTorch Flex Attention
│   │   ├── triton_attn.py                 # Triton自定义内核
│   │   ├── rocm_attn.py                   # ROCm专用attention
│   │   ├── rocm_aiter_fa.py               # ROCm AITER Flash Attention
│   │   ├── rocm_aiter_unified_attn.py     # ROCm统一attention
│   │   ├── cpu_attn.py                    # CPU回退实现
│   │   ├── tree_attn.py                   # 树形attention（推测解码）
│   │   ├── mamba_attn.py                  # Mamba基础attention
│   │   ├── mamba1_attn.py                 # Mamba SSM v1
│   │   ├── mamba2_attn.py                 # Mamba SSM v2
│   │   ├── short_conv_attn.py             # 短卷积attention
│   │   ├── linear_attn.py                 # 线性attention
│   │   ├── gdn_attn.py                    # GDN attention
│   │   └── mla/                           # MLA后端集合
│   │       ├── flashinfer_mla.py
│   │       ├── flashinfer_mla_sparse.py
│   │       ├── flashmla.py
│   │       ├── flashmla_sparse.py
│   │       ├── flashattn_mla.py
│   │       ├── triton_mla.py
│   │       ├── cutlass_mla.py
│   │       ├── aiter_triton_mla.py
│   │       ├── rocm_aiter_mla.py
│   │       ├── rocm_aiter_mla_sparse.py
│   │       ├── indexer.py
│   │       └── sparse_utils.py
│   └── ops/                               # 优化的attention操作
│       ├── common.py
│       ├── paged_attn.py
│       ├── merge_attn_states.py
│       ├── triton_merge_attn_states.py
│       ├── prefix_prefill.py
│       ├── triton_prefill_attention.py
│       ├── triton_decode_attention.py
│       ├── triton_unified_attention.py
│       ├── chunked_prefill_paged_decode.py
│       ├── triton_reshape_and_cache_flash.py
│       ├── flashmla.py
│       └── rocm_aiter_mla_sparse.py
├── model_executor/layers/attention/        # 模型执行器Attention层
│   ├── attention.py                        # 主Attention层集成
│   ├── mla_attention.py                    # MLA Attention层
│   └── ...
├── config/
│   └── attention.py                        # Attention配置
└── platforms/                              # 平台特定选择逻辑
    ├── cuda.py
    ├── rocm.py
    ├── cpu.py
    └── xpu.py

csrc/attention/                             # CUDA/C++底层内核
├── paged_attention_v1.cu                   # V1 Paged Attention内核
├── paged_attention_v2.cu                   # V2 Paged Attention内核
├── attention_kernels.cuh                   # 核心内核定义
├── attention_generic.cuh                   # 泛型内核
├── attention_utils.cuh                     # 工具函数
├── merge_attn_states.cu                    # 合并attention状态
├── dtype_float16.cuh                       # FP16数据类型
├── dtype_bfloat16.cuh                      # BF16数据类型
├── dtype_float32.cuh                       # FP32数据类型
├── dtype_fp8.cuh                           # FP8数据类型
└── mla/                                    # MLA专用CUDA内核
    └── cutlass_sm100_mla/                  # SM100 CUTLASS MLA
```

### 1.2 架构层次

vLLM的Attention系统采用**四层架构**设计：

```
┌─────────────────────────────────────────────────────────┐
│ 第一层：模型层（Model Layer）                              │
│   Attention Layer → 调用Backend的forward方法               │
├─────────────────────────────────────────────────────────┤
│ 第二层：后端抽象层（Backend Abstraction Layer）             │
│   AttentionBackend → 声明能力                             │
│   AttentionMetadataBuilder → 构建元数据                    │
│   AttentionImpl → 执行计算                                │
├─────────────────────────────────────────────────────────┤
│ 第三层：操作层（Operations Layer）                          │
│   paged_attn, merge_states, triton_unified_attention等    │
├─────────────────────────────────────────────────────────┤
│ 第四层：内核层（Kernel Layer）                              │
│   CUDA C++ kernels, Triton kernels, FlashAttention库       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 核心抽象层设计

### 2.1 AttentionType 枚举

定义了四种attention类型：

```python
class AttentionType(str, Enum):
    DECODER              # 解码器自注意力（Q/K/V之间）
    ENCODER              # 编码器自注意力
    ENCODER_ONLY         # 仅编码器attention
    ENCODER_DECODER      # 编码器-解码器交叉attention
```

### 2.2 AttentionBackend 抽象基类

**职责**：声明后端能力、提供工厂方法

```python
class AttentionBackend(ABC):
    # 类变量 - 声明后端特性
    accept_output_buffer: bool = False          # 是否接受预分配输出缓冲
    supported_dtypes: list[torch.dtype]         # 支持的数据类型
    supported_kv_cache_dtypes: list[CacheDType] # 支持的KV缓存数据类型
    forward_includes_kv_cache_update: bool = True  # forward是否包含KV缓存更新

    # 工厂方法
    def get_name() -> str                       # 后端标识符
    def get_impl_cls() -> type[AttentionImpl]   # 获取实现类
    def get_builder_cls() -> type[MetadataBuilder]  # 获取元数据构建器类
    def get_kv_cache_shape(...)                 # KV缓存形状

    # 能力检查方法（classmethod）
    def supports_head_size(head_size) -> bool   # 支持的注意力头大小
    def supports_dtype(dtype) -> bool           # 支持的数据类型
    def supports_kv_cache_dtype(kv_cache_dtype) -> bool  # 支持的KV缓存类型
    def supports_block_size(block_size) -> bool  # 支持的块大小
    def supports_attn_type(attn_type) -> bool   # 支持的attention类型
    def supports_sink() -> bool                 # 是否支持attention sink
    def supports_mm_prefix() -> bool            # 是否支持多模态前缀
    def is_mla() -> bool                        # 是否为MLA后端
    def is_sparse() -> bool                     # 是否为稀疏后端
    def supports_per_head_quant_scales() -> bool  # 是否支持逐头量化
    def supports_compute_capability(cap) -> bool  # 是否支持特定算力

    # 综合验证
    def validate_configuration(...) -> list[str]  # 返回不兼容原因列表
```

### 2.3 AttentionImpl 抽象实现类

**职责**：执行实际的attention计算

```python
class AttentionImpl(ABC, Generic[M]):
    num_heads: int
    head_size: int
    scale: float
    num_kv_heads: int
    can_return_lse_for_decode: bool    # 是否能返回LSE用于DCP

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,           # [num_tokens, num_heads, head_size]
        key: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,           # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,        # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: M,              # 后端特定元数据
        output: Optional[torch.Tensor],
        output_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:                 # [num_tokens, num_heads * head_size]
```

### 2.4 MLAAttentionImpl 和 SparseMLAAttentionImpl

MLA（Multi-Head Latent Attention）的专用接口：

```python
class MLAAttentionImpl(ABC, Generic[M]):
    """使用低维潜在空间投影的attention实现"""

class SparseMLAAttentionImpl(MLAAttentionImpl):
    """MLA的稀疏变体，支持选择性注意力"""
```

### 2.5 CommonAttentionMetadata

**职责**：每批次共享的、跨层的通用元数据

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor       # 查询token边界（累积）
    query_start_loc_cpu: torch.Tensor   # CPU版本
    seq_lens: torch.Tensor              # 每序列总长度
    num_reqs: int                       # 请求数量
    num_actual_tokens: int              # 实际token总数（无填充）
    max_query_len: int                  # 批次最大查询长度
    max_seq_len: int                    # 批次最大序列长度
    block_table_tensor: torch.Tensor    # KV缓存块索引
    slot_mapping: torch.Tensor          # 槽位映射
    causal: bool                        # 因果掩码标志

    # 可选字段
    logits_indices_padded: Optional[torch.Tensor]  # 快速prefill用
    encoder_seq_lens: Optional[torch.Tensor]       # 交叉attention用
    dcp_local_seq_lens: Optional[torch.Tensor]     # 分布式上下文并行用
```

### 2.6 AttentionMetadataBuilder

**职责**：将通用元数据转换为后端特定格式

```python
class AttentionMetadataBuilder(ABC, Generic[M]):
    _cudagraph_support: AttentionCGSupport  # CUDA图支持级别
    reorder_batch_threshold: int            # 批次重排阈值

    def build(
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> M:
        """主要转换方法：CommonAttentionMetadata → 后端特定元数据"""

    def build_for_cudagraph_capture(...) -> M:
        """CUDA图记录专用构建"""

    def build_for_drafting(...) -> M:
        """推测解码快速构建"""
```

---

## 3. Backend选择机制

### 3.1 选择入口

选择流程从 `get_attn_backend()` 函数开始：

```python
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    block_size: int | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
```

### 3.2 选择流程

```
用户请求/模型配置
    ↓
get_attn_backend()
    ↓
创建 AttentionSelectorConfig（头大小、数据类型、块大小、MLA标志等）
    ↓
检查 AttentionConfig.backend 是否有用户指定的后端
    ↓
调用 current_platform.get_attn_backend_cls()
    ↓
平台特定选择逻辑（CUDA/ROCm/CPU/XPU）
    ↓
验证后端兼容性（validate_configuration）
    ↓
返回选中的 AttentionBackend 类
```

### 3.3 Backend注册表

所有后端在 `AttentionBackendEnum` 中注册：

```python
class AttentionBackendEnum(str, Enum):
    # 标准Attention
    FLASH_ATTN                    # FlashAttention v2/3/4
    FLASH_ATTN_DIFFKV             # 不同KV维度的FlashAttention
    TRITON_ATTN                   # Triton自定义内核
    FLASHINFER                    # FlashInfer后端
    FLEX_ATTENTION                # PyTorch Flex Attention
    CPU_ATTN                      # CPU回退

    # ROCm专用
    ROCM_ATTN                     # ROCm参考实现
    ROCM_AITER_FA                 # ROCm AITER Flash Attention
    ROCM_AITER_UNIFIED_ATTN       # ROCm统一attention

    # MLA变体
    FLASHINFER_MLA                # FlashInfer MLA
    FLASHINFER_MLA_SPARSE         # FlashInfer MLA稀疏
    TRITON_MLA                    # Triton MLA
    CUTLASS_MLA                   # CUTLASS MLA（SM100）
    FLASHMLA                      # FlashMLA
    FLASHMLA_SPARSE               # FlashMLA稀疏
    FLASH_ATTN_MLA                # FlashAttention MLA
    ROCM_AITER_MLA                # ROCm AITER MLA
    ROCM_AITER_TRITON_MLA         # ROCm AITER Triton MLA
    ROCM_AITER_MLA_SPARSE         # ROCm AITER MLA稀疏

    # 特殊类型
    TREE_ATTN                     # 树形attention
    NO_ATTENTION                  # 无attention
    TORCH_SDPA                    # ViT SDPA
    CUSTOM                        # 自定义后端
```

Mamba SSM后端在 `MambaAttentionBackendEnum` 中单独注册：

```python
class MambaAttentionBackendEnum(str, Enum):
    MAMBA1                        # Mamba SSM v1
    MAMBA2                        # Mamba SSM v2
    SHORT_CONV                    # 短卷积
    LINEAR                        # 线性attention
    GDN_ATTN                      # GDN attention
    CUSTOM                        # 自定义
```

### 3.4 平台特定选择逻辑

#### CUDA平台选择（优先级排序）

**非MLA模式：**

| 优先级 | Blackwell (CC 10.x) | 其他GPU (CC 8.0+) |
|--------|----------------------|---------------------|
| 1 | FLASHINFER | FLASH_ATTN |
| 2 | FLASH_ATTN | FLASHINFER |
| 3 | TRITON_ATTN | TRITON_ATTN |
| 4 | FLEX_ATTENTION | FLEX_ATTENTION |

**MLA模式：**

| 优先级 | Blackwell (CC 10.x) | 其他GPU (CC 8.0+) |
|--------|----------------------|---------------------|
| 1 | FLASHINFER_MLA | FLASH_ATTN_MLA |
| 2 | CUTLASS_MLA | FLASHMLA |
| 3 | FLASH_ATTN_MLA | FLASHINFER_MLA |
| 4 | FLASHMLA | TRITON_MLA |
| 5 | TRITON_MLA | FLASHMLA_SPARSE |

#### ROCm平台选择

ROCm使用环境变量驱动的选择逻辑：

```
1. 稀疏检查 → 如果use_sparse: ROCM_AITER_MLA_SPARSE

2. MLA处理 → 如果use_mla:
   ├─ 默认: ROCM_AITER_MLA 或 TRITON_MLA
   └─ 根据VLLM_ROCM_USE_AITER_MLA环境变量

3. 环境变量驱动（无指定backend时）：
   ├─ VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION → ROCM_AITER_UNIFIED_ATTN
   ├─ VLLM_ROCM_USE_AITER_MHA + gfx9 → ROCM_AITER_FA
   ├─ use_prefill_decode_attention → ROCM_ATTN
   ├─ VLLM_ROCM_USE_AITER + gfx9 → ROCM_AITER_FA
   └─ 默认: TRITON_ATTN
```

#### CPU平台选择

CPU平台始终返回 `CPU_ATTN`，不支持MLA和稀疏attention。

#### XPU平台选择

XPU平台默认使用 `FLASH_ATTN`，仅支持NHD布局，MLA使用 `TRITON_MLA`。

---

## 4. 标准Attention后端详解

### 4.1 FlashAttention后端（`flash_attn.py`）

**核心特性：**
- 支持FlashAttention v2、v3、v4
- 支持数据类型：float16、bfloat16
- 计算能力要求：≥ 8.0
- 块大小：MultipleOf(16)
- KV缓存形状：`[2, num_blocks, block_size, num_kv_heads, head_size]`

**元数据（FlashAttentionMetadata）：**
```python
@dataclass(frozen=True)
class FlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # 级联attention支持
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # DCP（解码上下文并行）支持
    max_dcp_context_kv_len: Optional[int]
    dcp_context_kv_lens: Optional[torch.Tensor]

    # FA3 AOT调度
    scheduler_metadata: Optional[Any]
```

**前向传播流程：**
1. 验证元数据存在（profile运行时为None）
2. **编码器attention路径**：直接使用Q/K/V张量，不使用KV缓存
3. **解码器/交叉attention路径**：
   - 解绑KV缓存为key_cache和value_cache
   - 处理FP8量化视图转换
   - **非级联流程**：标准attention，使用block_table和scheduler_metadata
   - **级联流程**：分别对前缀和后缀进行attention，使用双调度器
   - 调用 `flash_attn_varlen_func()` 执行计算

**关键优势：**
- FA3的AOT（Ahead-of-Time）调度优化CUDA图
- 级联attention支持公共前缀高效处理
- FP8量化支持（逐序列缩放因子）

### 4.2 FlashInfer后端（`flashinfer.py`）

**核心特性：**
- 集成FlashInfer/TRT-LLM优化内核
- 支持FP4和FP8量化
- 计算能力要求：≥ 8.0
- 块大小：MultipleOf(16)
- KV缓存形状：`[2, num_blocks, num_kv_heads, block_size, head_size]`（HND布局）

**关键差异：**
- 使用wrapper-based API而非直接内核调用
- TRT-LLM集成提供优化的TensorRT后端支持
- 分别为decode和prefill使用独立wrapper
- 支持prefill时FP8 KV缓存的按需解量化

**元数据构建流程：**
1. 将batch分为decode和prefill token
2. 创建 `BatchDecodeWithPagedKVCacheWrapper` 和 `BatchPrefillWithPagedKVCacheWrapper`
3. 通过 `flashinfer.decode.fast_decode_plan()` 规划attention
4. 处理多级级联attention（`MultiLevelCascadeAttentionWrapper`）
5. 管理TRT-LLM attention（如启用，处理FP8 KV缓存解量化）

**前向传播流程：**
1. 更新wrapper的序列长度和块表
2. 通过 `fast_decode_plan()` 规划attention内核
3. **Decode路径**：调用 `paged_kv_cache_decode_wrapper.forward()`
4. **Prefill路径**：调用 `paged_kv_cache_prefill_wrapper.forward()`
5. 必要时合并结果
6. 级联attention单独处理

### 4.3 Triton Attention后端（`triton_attn.py`）

**核心特性：**
- 自定义Triton编译内核
- 支持数据类型：float16、bfloat16、**float32**
- KV缓存数据类型：auto、bfloat16、fp8、fp8_e4m3、fp8_e5m2
- 块大小：MultipleOf(16)
- KV缓存形状：`[num_blocks, 2, block_size, num_kv_heads, head_size]`（物理布局不同）
- 支持**所有**attention类型

**特殊功能：**
- 统一内核处理prefill和decode
- 3D与2D内核自动切换（`seq_threshold_3D`）
- Softmax分段处理长序列
- 支持ALiBi slopes（sqrt变体）和滑动窗口
- 支持融合FP8输出量化

**前向传播流程：**
1. 处理编码器attention（直接Q/K/V）
2. 解绑KV缓存（注意不同的步长顺序）
3. FP8数据类型视图转换
4. 调用 `unified_attention()`：
   - 统一内核同时处理prefill和decode
   - 使用分段softmax
   - 支持ALiBi和滑动窗口
   - 返回attention输出和可选LSE

### 4.4 FlexAttention后端（`flex_attention.py`）

**核心特性：**
- 纯PyTorch实现（无CUDA内核）
- 基于 `torch.nn.attention.flex_attention`（需PyTorch ≥ 2.7）
- 支持数据类型：float16、bfloat16、float32
- 支持attention类型：DECODER、ENCODER_ONLY
- 支持多模态前缀（mm_prefix）

**关键机制：**
- **物理到逻辑地址转换**：在mask_mod中实现分页KV缓存的地址映射
- **BlockMask**：使用PyTorch的BlockMask对象进行块级掩码
- **score_mod**：可组合的分数修改函数

**前向传播流程：**
1. 验证元数据
2. 检查block_mask是否需要重建（滑动窗口或mm_prefix变化）
3. **编码器（非因果）路径**：直接4D张量attention
4. **解码器（因果）路径**：
   - 构建分页KV缓存的block_mask
   - Q/K/V重塑为4D
   - 调用 `flex_attention_compiled()` 
   - 输出重塑回2D

**限制**：无量化支持，滑动窗口和ALiBi支持有限

### 4.5 CPU Attention后端（`cpu_attn.py`）

**核心特性：**
- 支持数据类型：float16、bfloat16、float32
- 头大小：32、64、80、96、112、128、160、192、224、256
- 支持所有attention类型
- KV缓存形状：`[2, num_blocks, num_kv_heads, block_size, head_size]`
- 无CUDA图支持

**CPU指令集选择（ISA）：**
```python
isa = {
    "amx"   # x86 AMX指令
    "neon"  # ARM NEON
    "vxe"   # IBM VXE
    "vec"   # 通用向量
    "vec16" # 16位向量
}
```

**前向传播流程：**
1. **编码器attention**：使用 `_run_sdpa_forward()` 和PyTorch SDPA
2. **解码器attention**：
   - `ops.cpu_attn_reshape_and_cache()` 写入KV缓存
   - 如SDPA prefill启用：对prefill token运行SDPA
   - 对decode token运行 `ops.cpu_attention_with_kv_cache()`
3. **SDPA前向**：
   - 转换为4D张量
   - 逐序列掩码（ALiBi或滑动窗口）
   - `torch.nn.functional.scaled_dot_product_attention()`

### 4.6 ROCm Attention后端（`rocm_attn.py`）

**核心特性：**
- AMD GPU优化
- 支持块大小：16、32、544
- 支持头大小：32、64、80、96、128、160、192、224、256
- 支持FP8 KV缓存量化

**前向传播流程：**
- 使用 `chunked_prefill_paged_decode()` 进行attention计算
- 非2的幂次块大小回退到Triton内核
- 融合RoPE + KV缓存更新：`rocm_aiter_ops.triton_rope_and_cache()`

### 4.7 ROCm AITER Flash Attention后端（`rocm_aiter_fa.py`）

**核心特性：**
- 高级张量迭代（AITER）优化
- 支持推测解码（draft token）
- 三阶段工作负载划分：prefill、decode、extend
- 自定义Triton内核：`cp_mha_gather_cache_kernel`

**独特功能：**
- **上下文扩展（extend阶段）**：处理上下文增长时的滑动窗口
- **Shuffle布局**：`reshape_and_cache_shuffle_kernel` 使用排列布局优化内存访问
- **工作空间动态分配**：按chunk处理的动态内存

### 4.8 FlashAttention DiffKV后端（`flash_attn_diffkv.py`）

**用途**：支持K和V具有**不同头维度**的模型

**关键差异：**
- KV缓存形状：`[num_blocks, block_size, num_kv_heads, head_size + head_size_v]`
  - 前 `head_size` 维为key
  - 后 `head_size_v` 维为value
- 输出形状：`[num_tokens, num_heads * head_size_v]`
- 使用 `triton_reshape_and_cache_flash_diffkv()` 进行缓存更新

---

## 5. MLA（Multi-Head Latent Attention）后端详解

MLA是DeepSeek-V2/V3模型引入的一种高效attention机制，通过低维潜在空间投影减少KV缓存开销。

### 5.1 MLA工作原理

```
标准Multi-Head Attention:
  Q = W_q · x       K = W_k · x       V = W_v · x
  KV缓存大小 = num_kv_heads × head_size × 2

MLA（Multi-Head Latent Attention）:
  1. 将输入投影到低维潜在空间：c = W_down · x
  2. 将潜在表示展开到K/V：K = W_uk · c,  V = W_uv · c
  3. KV缓存只存储压缩后的c（加上位置编码部分k_pe）
  KV缓存大小 = latent_dim + pe_dim << num_kv_heads × head_size × 2

典型维度：
  - q_nope: 512维（潜在注意力部分）
  - q_pe:    64维（旋转位置编码部分）
  - kv_c:   512维（压缩的KV潜在表示）
  - k_pe:    64维（key的位置编码）
```

### 5.2 MLA后端变体

#### Triton MLA（`triton_mla.py`）
- 使用Triton自定义内核
- 解码时调用 `decode_attention_fwd()`
- Q分割为 `q_nope`（潜在部分）和 `q_pe`（位置编码部分）
- 可变KV分割数（批次不变时1个，否则4个）
- **限制**：不支持ALiBi、滑动窗口、logits soft cap、FP8 KV缓存

#### CUTLASS MLA（`cutlass_mla.py`）
- 使用CUTLASS库优化
- **仅支持SM 10.0+**（Blackwell架构）
- 固定128字节块大小
- 支持FP8（fp8_e4m3）
- 工作空间管理：`SM100Workspace` 类
- 填充隐藏维度到128头以满足内核效率要求

#### FlashMLA（`flashmla.py`）
- 基于FlashAttention的MLA实现
- 支持计算能力9.0（Hopper）和10.0（Blackwell）
- 分别构建decode和prefill元数据
- 批次重排阈值：128（大于标准值以提高MLA效率）

#### FlashInfer MLA（`flashinfer_mla.py`）
- FlashInfer库的MLA实现
- 支持TRT-LLM集成

#### FlashAttention MLA（`flashattn_mla.py`）
- 使用标准FlashAttention库实现MLA
- 更通用的兼容性

#### 稀疏MLA变体
- **FlashInfer MLA Sparse**：稀疏注意力模式的FlashInfer MLA
- **FlashMLA Sparse**：稀疏注意力模式的FlashMLA
- **ROCm AITER MLA Sparse**：ROCm平台的稀疏MLA

#### ROCm MLA变体
- **ROCM_AITER_MLA**：ROCm AITER优化的MLA
- **ROCM_AITER_TRITON_MLA**：ROCm AITER + Triton组合的MLA

### 5.3 MLA Indexer（`indexer.py`）

为DeepSeek-V3.2提供的索引逻辑：
- **Prefill分块**：将大型prefill请求分割为 `max_prefill_buffer_size` 大小的块
- **Decode展开**：将多token decode展开为单token条目
- **KV span计算**：`kv_spans_from_batches()` 计算每个token在连接缓存中的KV起止位置

---

## 6. 特殊Attention后端详解

### 6.1 Tree Attention（`tree_attn.py`）

**用途**：推测解码中的树形attention结构

**工作原理：**
```
根节点（token 0）
├─ Token 1 → 可关注根节点 + 自身
├─ Token 2 → 可关注根节点 + 自身 + 父节点
└─ Token 3 → 可关注根节点 + 自身 + 父节点 + 祖先

Tree Attention Bias矩阵构建：
1. 初始化为-inf（禁止所有attention）
2. 对角线设为0（允许自注意力）
3. [:, 0] = 0（所有token关注根节点）
4. 每个token的祖先位置设为0（关注父节点链）
```

**前向传播：**
- 使用 `reshape_and_cache_flash` 缓存KV
- 对prefill和decode分别运行 `unified_attention`
- Decode阶段使用 `tree_attn_bias` 强制树形结构

### 6.2 Mamba Attention（`mamba_attn.py`、`mamba1_attn.py`、`mamba2_attn.py`）

**用途**：Mamba状态空间模型的"attention"（实际为递归状态更新）

**与标准Attention的根本区别：**
- **无Q×K^T计算** — 使用递归状态更新
- **选择性SSM**：可根据输入选择性地更新状态
- **线性复杂度**：O(L) 而非 O(L²)
- 因果conv1d用于局部上下文

**状态管理：**
```python
@dataclass
class BaseMambaAttentionMetadata:
    has_initial_states_p: bool          # prefill序列是否有初始状态
    state_indices_tensor_p/d: Tensor    # 访问RNN状态的索引
    cu_chunk_seqlen_p: Tensor           # 每块的累积序列长度
    block_idx_last_scheduled_token: Tensor   # 前缀缓存最后计算的块
```

### 6.3 Linear Attention（`linear_attn.py`）

**用途**：线性时间复杂度的attention（如RWKV风格模型）

**特点：**
- 使用 `MambaSpec`（基于状态的KV规范，非分页缓存）
- 无块表或槽位映射
- CUDA图支持：`UNIFORM_SINGLE_TOKEN_DECODE`

### 6.4 Short Conv Attention（`short_conv_attn.py`）

**用途**：Mamba2模型中的短卷积内核

**特点：**
- 继承 `BaseMambaAttentionMetadata`
- 专注短距离token依赖的卷积操作

### 6.5 GDN Attention（`gdn_attn.py`）

**用途**：Gated Delta Net attention，支持推测解码

**特点：**
- 复杂的推测解码元数据处理
- 将batch分为：非推测decode、prefill、推测decode
- 支持因果卷积元数据
- 完整CUDA图支持

---

## 7. CUDA/C++ 底层内核分析

### 7.1 Paged Attention内核（`attention_kernels.cuh`）

这是vLLM最核心的CUDA内核，实现了分页attention计算。

#### 核心内核：`paged_attention_kernel`

```cuda
template <
    typename scalar_t,       // 输出/查询数据类型
    typename cache_t,        // 缓存数据类型
    int HEAD_SIZE,           // 注意力头大小
    int BLOCK_SIZE,          // 页块大小
    int NUM_THREADS,         // 每block线程数
    vllm::Fp8KVCacheDataType KV_DTYPE,  // FP8配置
    bool IS_BLOCK_SPARSE     // 是否块稀疏
>
__global__ void paged_attention_kernel(...)
```

**算法流程：**

```
Phase 1: Query加载（寄存器级别）
  └─ 每个线程组加载query向量到寄存器

Phase 2: Key块迭代
  ├─ 每个warp处理一个key块
  ├─ 线程组计算Q×K^T（向量化点积）
  ├─ 应用因果掩码（mask掉未来token）
  ├─ 应用ALiBi slopes（如有）
  └─ Logits存入共享内存

Phase 3: Softmax计算
  ├─ 并行max-reduction找到最大logit
  ├─ 指数归一化: exp(logit - max)
  ├─ 并行sum-reduction计算归一化因子
  └─ 存储归一化的attention权重

Phase 4: Value聚合
  ├─ 遍历value块
  ├─ 获取value向量（考虑缓存格式）
  ├─ 加权累积: sum(softmax × V)
  └─ 跨warp并行reduction得到最终输出

Phase 5: 输出写入
  └─ 每个head的最终attention输出写入全局内存
```

### 7.2 Paged Attention V1 vs V2

| 方面 | V1 | V2 |
|------|----|----|
| **分区** | 单次pass（PARTITION_SIZE=0） | 多分区（PARTITION_SIZE=512） |
| **Grid维度** | (num_heads, num_seqs, 1) | (num_heads, num_seqs, max_num_partitions) |
| **输出存储** | 直接写入最终输出 | 写入临时缓冲tmp_out |
| **中间数据** | 无 | 每个分区的exp_sums和max_logits |
| **Reduce阶段** | 不需要 | 需要paged_attention_v2_reduce_kernel |
| **适用场景** | 较短序列 | 较长序列（内存效率更高） |

**V2 Reduce算法：**
```
1. 加载所有分区的max_logits，找到全局max
2. 重新缩放exp_sums: exp_sum_i × exp(max_i - global_max)
3. 计算全局归一化: inv_sum = 1 / (∑ rescaled_exp_sums)
4. 加权聚合: out += (tmp_out_i × scaled_exp_sum_i × inv_sum)
```

### 7.3 Merge Attention States（`merge_attn_states.cu`）

**用途**：合并前缀和后缀的attention输出（用于Split-KV attention）

```cuda
算法：
For each token-head pair:
  1. 加载prefix_lse和suffix_lse（log-sum-exp值）
  2. max_lse = max(prefix_lse, suffix_lse)
  3. 如果max_lse = -inf: 输出prefix_output
  4. 否则:
     p_scale = exp(prefix_lse - max_lse) / (exp(prefix_lse - max_lse) + exp(suffix_lse - max_lse))
     output = p_scale × prefix_output + s_scale × suffix_output
     output_lse = log(exp(prefix_lse - max_lse) + exp(suffix_lse - max_lse)) + max_lse
```

### 7.4 Reshape and Cache（`cache_kernels.cu`）

**用途**：将token的K/V写入分页KV缓存

```cuda
reshape_and_cache_kernel:
For each token:
  1. 从slot_mapping获取slot_idx
  2. 计算物理位置:
     block_idx = slot_idx / block_size
     block_offset = slot_idx % block_size
  3. For each head:
     key[token, head, :] → key_cache[block, head, :, block_offset, :]
     value[token, head, :] → value_cache[block, head, :, block_offset]
  4. 可选: 写入时应用FP8量化
```

### 7.5 数据类型特化

| 文件 | 数据类型 | 大小 |
|------|----------|------|
| `dtype_float16.cuh` | FP16 | 12.1 KB |
| `dtype_bfloat16.cuh` | BF16 | 12.1 KB |
| `dtype_float32.cuh` | FP32 | 5.6 KB |
| `dtype_fp8.cuh` | FP8 | 607 B |

每种数据类型提供向量化加载/存储和格式转换的特化实现。

---

## 8. Attention操作层（Ops）

### 8.1 Paged Attention操作（`paged_attn.py`）

```python
class PagedAttention:
    @staticmethod
    def split_kv_cache(kv_cache, num_kv_heads, head_size):
        """将KV缓存从[2, num_blocks, ...]分割为独立的key/value视图
        key_cache: [num_blocks, num_kv_heads, head_size//x, -1, x]
        value_cache: [num_blocks, num_kv_heads, head_size, -1]
        x = 16 // element_size (用于16字节向量化)
        """

    @staticmethod
    def write_to_paged_cache(key, value, key_cache, value_cache,
                            slot_mapping, kv_cache_dtype, k_scale, v_scale):
        """调用reshape_and_cache CUDA内核写入分页KV缓存"""
```

### 8.2 Merge Attention States（`merge_attn_states.py`）

```python
def merge_attn_states(output, prefix_output, prefix_lse,
                      suffix_output, suffix_lse, output_lse=None):
    """合并前缀和后缀attention输出
    策略：
    - CUDA内核: dtype ∈ {float32, float16, bfloat16} 且 head_size % pack_size == 0
    - Triton回退: FP8 dtype或不支持的head_size
    """
```

### 8.3 Triton统一Attention（`triton_unified_attention.py`）

统一的prefill+decode Triton内核，支持：
- 分段softmax处理长序列
- 3D与2D内核自动切换
- ALiBi slopes和滑动窗口
- FP8输出量化

### 8.4 前缀预填充（`prefix_prefill.py`）

处理带有公共前缀的prefill attention，允许多个请求共享前缀的KV缓存。

### 8.5 通用操作（`common.py`）

```python
# 上下文并行attention输出校正
_correct_attn_cp_out_kernel()  # Triton内核

# 序列打包/解包
pack_seq_triton()    # 变长序列 → 固定 [B, Lmax, D]
unpack_seq_triton()  # 固定 [B, Lmax, D] → 变长序列
```

---

## 9. 完整调用流程

### 9.1 初始化阶段

```
Model.__init__()
    ↓
AttentionLayer.__init__()
    ├─ get_attn_backend()
    │   ├─ 创建 AttentionSelectorConfig（head_size, dtype, block_size等）
    │   ├─ 检查 AttentionConfig.backend（用户指定）
    │   ├─ current_platform.get_attn_backend_cls()
    │   │   ├─ CUDA: 按优先级列表逐个验证
    │   │   ├─ ROCm: 环境变量驱动选择
    │   │   ├─ CPU: 始终返回CPU_ATTN
    │   │   └─ XPU: 默认FLASH_ATTN
    │   ├─ backend.validate_configuration()
    │   └─ 返回 AttentionBackend类
    │
    ├─ backend.get_impl_cls() → FlashAttentionImpl / TritonAttentionImpl / ...
    ├─ backend.get_builder_cls() → FlashAttentionMetadataBuilder / ...
    ├─ AttentionImpl.__init__()
    │   └─ 存储超参数: num_heads, head_size, scale, kv_cache_dtype等
    └─ AttentionMetadataBuilder.__init__()
        └─ 预分配缓冲区（如Triton的softmax分段缓冲）
```

### 9.2 每批次前向传播阶段

```
A. 输入批次准备（worker/scheduler）
   ↓
   创建 CommonAttentionMetadata:
   ├─ query_start_loc: token边界（累积）
   ├─ seq_lens: 每序列长度
   ├─ block_table_tensor: 从block_manager获取块索引
   ├─ slot_mapping: 从block_manager获取槽位映射
   └─ causal, dcp_local_seq_lens等标志

B. 逐层元数据构建
   ↓
   For each attention layer:
   ├─ 获取该层的 AttentionMetadataBuilder
   ├─ builder.build(common_prefix_len, common_attn_metadata)
   │   ├─ 后端特定转换:
   │   │   ├─ FlashAttn: FA3 AOT调度、级联attention处理
   │   │   ├─ FlashInfer: Wrapper创建和规划
   │   │   ├─ Triton: Softmax分段缓冲分配
   │   │   ├─ CPU: ISA选择、SDPA掩码创建
   │   │   └─ FlexAttn: BlockMask构建、物理逻辑映射
   │   ├─ 分配或复用后端特定缓冲区
   │   └─ 返回后端特定元数据
   └─ 存储 layer_metadata[layer_name] = metadata

C. Attention前向传播
   ↓
   For each layer:
   ├─ AttentionLayer.forward()
   │   ├─ 线性投影: Q = W_q @ x, K = W_k @ x, V = W_v @ x
   │   ├─ 位置编码: RoPE / ALiBi
   │   ├─ 获取 layer_metadata[layer_name]
   │   │
   │   └─ impl.forward(
   │        query=Q,                    # [num_tokens, num_heads, head_size]
   │        key=K,                      # [num_tokens, num_kv_heads, head_size]
   │        value=V,                    # [num_tokens, num_kv_heads, head_size]
   │        kv_cache=cache,             # [2, num_blocks, ...]
   │        attn_metadata=metadata,     # 后端特定
   │        output=out_tensor           # 预分配（如accept_output_buffer=True）
   │      )
   │        ↓
   │        后端特定内核执行:
   │        ├─ FlashAttn: flash_attn_varlen_func()
   │        ├─ FlashInfer: wrapper.forward()
   │        ├─ Triton: unified_attention()
   │        ├─ CPU: torch.nn.functional.scaled_dot_product_attention()
   │        ├─ FlexAttn: flex_attention_compiled()
   │        └─ MLA: decode_attention_fwd() / sm100_cutlass_mla()
   │        ↓
   │        返回: attn_output [num_tokens, num_heads * head_size]
   │
   ├─ 线性投影: out = W_o @ attn_output
   └─ 返回层输出
```

### 9.3 KV缓存更新流程

```
大多数后端在forward中包含KV缓存更新:
  forward_includes_kv_cache_update = True

KV缓存更新路径:
  1. 通过slot_mapping确定每个token的物理存储位置
  2. 调用 reshape_and_cache() 或后端特定的缓存写入函数
  3. 支持可选的FP8量化（写入时量化）

分页KV缓存寻址:
  slot_idx → (block_idx, block_offset)
  block_idx = slot_idx / block_size
  block_offset = slot_idx % block_size
  
  key_cache[block_idx, head, :, block_offset, :] = key[token]
  value_cache[block_idx, head, :, block_offset] = value[token]
```

---

## 10. 配置系统与环境变量

### 10.1 AttentionConfig

```python
@config
class AttentionConfig:
    backend: AttentionBackendEnum | None = None
    # 自动选择（如为None）

    flash_attn_version: Literal[2, 3, 4] | None = None
    # 强制指定Flash Attention版本

    use_prefill_decode_attention: bool = False
    # 使用分离的prefill/decode内核

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    # CudaGraph的最大分割数

    use_cudnn_prefill: bool = False
    # 使用cuDNN进行prefill

    use_trtllm_ragged_deepseek_prefill: bool = True
    # TRT-LLM DeepSeek prefill

    use_trtllm_attention: bool | None = None
    # TRT-LLM attention后端

    disable_flashinfer_prefill: bool = False
    # 禁用FlashInfer prefill

    disable_flashinfer_q_quantization: bool = False
    # 跳过FP8 KV时的Q量化

    use_prefill_query_quantization: bool = False
    # 在prefill中量化Q
```

### 10.2 关键环境变量

| 环境变量 | 平台 | 用途 |
|----------|------|------|
| `VLLM_ROCM_USE_AITER` | ROCm | 启用AITER操作 |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | ROCm | 使用统一attention |
| `VLLM_ROCM_USE_AITER_MHA` | ROCm | 使用MHA变体 |
| `VLLM_ROCM_USE_AITER_MLA` | ROCm | 使用MLA |
| `FORCE_NUM_KV_SPLITS` | CUDA | 强制KV分割数（CUTLASS MLA） |

### 10.3 CLI配置方式

```bash
# 指定后端
--attention-config.backend flash_attn

# 指定Flash Attention版本
--attention-config.flash_attn_version 3

# 使用分离的prefill/decode内核
--attention-config.use_prefill_decode_attention true
```

---

## 11. 各Backend对比矩阵

### 11.1 标准Attention后端对比

| 特性 | FlashAttn | FlashInfer | Triton | CPU | FlexAttn | ROCm | AITER FA |
|------|-----------|-----------|--------|-----|----------|------|----------|
| **FP16/BF16** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **FP32** | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **FP8 KV缓存** | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| **滑动窗口** | ✓ | ✓ | ✓ | ✓ | 部分 | ✓ | ✓ |
| **ALiBi** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| **Attention Sink** | ✓ | 部分 | ✓ | ✗ | ✗ | ✓ | ✓ |
| **级联Attention** | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| **DCP** | ✓ | ✓ | ✓ | ✗ | 部分 | ✓ | ✓ |
| **CUDA图** | ✓(FA3) | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| **多模态前缀** | 部分 | 部分 | ✓ | ✗ | ✓ | 部分 | 部分 |
| **所有Attention类型** | ✓ | 部分 | ✓ | ✓ | 部分 | ✓ | ✓ |
| **计算能力要求** | ≥8.0 | ≥8.0 | 任意 | N/A | ≥8.0 | ROCm | ROCm |
| **推测解码** | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |

### 11.2 MLA后端对比

| 特性 | Triton MLA | CUTLASS MLA | FlashMLA | FlashInfer MLA | FlashAttn MLA |
|------|-----------|-------------|----------|----------------|---------------|
| **目标硬件** | 通用GPU | SM 10.0+ | CC 9.0/10.0 | CC 8.0+ | CC 8.0+ |
| **FP8 KV** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **块大小** | 灵活 | 固定128 | 灵活 | 灵活 | 灵活 |
| **ALiBi** | ✗ | ✗ | 部分 | 部分 | 部分 |
| **滑动窗口** | ✗ | ✗ | 部分 | 部分 | 部分 |
| **稀疏变体** | ✗ | ✗ | ✓ | ✓ | ✗ |

### 11.3 特殊后端适用场景

| 后端 | 适用模型/场景 |
|------|--------------|
| Tree Attention | 推测解码（树形draft） |
| Mamba1/2 | Mamba/Mamba2状态空间模型 |
| Linear Attention | RWKV等线性复杂度模型 |
| Short Conv | Mamba2短卷积层 |
| GDN Attention | Gated Delta Net模型 |
| FlashAttn DiffKV | K/V头维度不同的自定义模型 |

---

## 12. 关键设计模式与发现

### 12.1 插件化Backend架构

vLLM采用了高度模块化的**插件式Backend架构**：

```
AttentionBackend (接口)
    ├─ 能力声明 (supports_*)
    ├─ 工厂方法 (get_impl_cls, get_builder_cls)
    └─ 验证方法 (validate_configuration)

AttentionMetadataBuilder (转换器)
    └─ CommonAttentionMetadata → Backend特定Metadata

AttentionImpl (执行器)
    └─ forward() → 实际内核调用
```

**优点**：
- 新后端可以独立开发和集成
- 运行时动态选择最优后端
- 每个后端可以声明自己的限制条件

### 12.2 惰性导入与缓存

- 后端类仅在被选中时导入（惰性导入）
- 选择结果通过 `@cache` 装饰器缓存
- 避免加载不需要的依赖库

### 12.3 基于优先级的自动选择

- CUDA平台根据计算能力排列优先级列表
- 逐个尝试，选择第一个通过验证的后端
- Blackwell (SM100) 优先FlashInfer，其他GPU优先FlashAttention

### 12.4 KV缓存布局抽象

两种布局模式：
- **NHD** (num_kv_heads, head_size last): FlashAttention、Triton
- **HND** (head_size, num_kv_heads interleaved): FlashInfer

布局选择在后端选择后自动调整。

### 12.5 批次重排优化

```python
def split_decodes_and_prefills():
    """将batch重排，decode token放前面，prefill token放后面
    
    优势：
    - 更好的缓存局部性
    - 独立的内核路径
    - 支持推测解码的batch组合
    """
```

### 12.6 级联Attention（Cascade Attention）

当多个请求共享公共前缀时：
1. 对公共前缀的KV运行一次attention
2. 对每个请求的独有后缀分别运行attention
3. 使用LSE（log-sum-exp）合并两部分结果

### 12.7 数值稳定性技巧

- **Log-Sum-Exp技巧**：防止softmax溢出
- **Max-logit归一化**：在分区合并时使用
- **128位向量化加载/存储**：优化内存带宽

### 12.8 Override系统

```python
@register_backend(AttentionBackendEnum.FLASH_ATTN)
class CustomFlashAttention(FlashAttentionBackend):
    """允许覆盖默认后端实现"""
```

### 12.9 发现与洞察

1. **后端数量惊人**：包含31个后端实现（20个标准 + 5个Mamba + 6个MLA变体），反映了不同硬件和模型的优化需求
2. **MLA是关键方向**：MLA后端变体众多（12+），说明DeepSeek-V2/V3的MLA架构是当前优化重点
3. **ROCm支持成熟**：3个专用ROCm后端 + 3个ROCm MLA变体，表明AMD GPU支持已经相当完善
4. **Triton作为通用回退**：Triton后端支持最多的数据类型和attention类型，作为通用兜底方案
5. **推测解码深度集成**：Tree Attention和GDN后端专门为推测解码设计，批次管理涉及draft token分离
6. **非Transformer架构支持**：Mamba/SSM系列后端说明vLLM已超越纯Transformer服务框架
7. **内存效率是核心关注点**：分页attention、FP8量化、级联attention都围绕减少内存占用

---

## 13. 总结

vLLM的Attention系统是一个高度模块化、可扩展的框架，其设计体现了以下核心理念：

### 架构层次清晰

四层架构（模型层 → 后端抽象层 → 操作层 → 内核层）使得每一层可以独立演进。

### 硬件适配灵活

通过平台特定的选择逻辑和优先级排序，自动为每种GPU架构（NVIDIA Ampere/Hopper/Blackwell、AMD gfx9、Intel XPU、CPU）选择最优后端。

### 前沿技术集成

从Flash Attention v2到v4、FlashInfer、CUTLASS、Triton，集成了当前所有主流的attention优化技术。

### 新架构支持

不仅支持标准Multi-Head Attention，还全面支持MLA（Multi-Head Latent Attention）、Mamba/SSM、线性attention等新兴架构。

### 性能优化全面

分页attention、FP8/FP4量化、级联attention、推测解码、CUDA图、分布式上下文并行等优化技术全面覆盖。

### 统计数据

| 指标 | 数量 |
|------|------|
| 后端实现总数 | 31 (20标准 + 5 Mamba + 6 MLA) |
| 后端文件数 | 18 + 11 (MLA) = 29 |
| 操作文件数 | 12 |
| CUDA内核文件数 | 15+ |
| 支持的架构平台 | CUDA, ROCm, CPU, XPU |
| Attention类型 | 4 (Decoder, Encoder, Encoder-Only, Cross) |
| 支持的数据类型 | FP32, FP16, BF16, FP8, FP4 |
| MLA变体数量 | 12+ |

---

*报告生成日期：2026-03-12*
*基于vLLM项目源代码深度分析*
