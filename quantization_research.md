# vLLM 量化技术深度研究报告

## 目录

1. [概述](#1-概述)
2. [量化对象：对什么做量化](#2-量化对象对什么做量化)
3. [量化方法：怎样做量化](#3-量化方法怎样做量化)
4. [量化对象 × 量化方法交叉矩阵](#4-量化对象--量化方法交叉矩阵)
5. [架构设计与核心抽象](#5-架构设计与核心抽象)
6. [量化方法注册与发现机制](#6-量化方法注册与发现机制)
7. [完整调用流程](#7-完整调用流程)
8. [主要量化方法详解](#8-主要量化方法详解)
   - 8.1 [AWQ 量化](#81-awq-activation-aware-weight-quantization)
   - 8.2 [GPTQ 量化](#82-gptq-量化)
   - 8.3 [FP8 量化](#83-fp8-量化)
   - 8.4 [CompressedTensors 统一框架](#84-compressedtensors-统一框架)
   - 8.5 [BitsAndBytes 量化](#85-bitsandbytes-量化)
   - 8.6 [GGUF 量化](#86-gguf-量化)
   - 8.7 [ModelOpt 量化](#87-modelopt-nvidia-量化框架)
   - 8.8 [TorchAO 量化](#88-torchao-量化)
   - 8.9 [Quark 量化框架](#89-quark-量化框架)
   - 8.10 [其他量化方法](#810-其他量化方法)
9. [MoE 模型量化](#9-moe-模型量化)
10. [KV Cache 量化](#10-kv-cache-量化)
11. [工具层与内核支持](#11-工具层与内核支持)
12. [量化方法对比总结](#12-量化方法对比总结)
13. [关键发现与洞察](#13-关键发现与洞察)

---

## 1. 概述

vLLM 是一个高性能的大语言模型推理引擎，其量化子系统是其核心组件之一。该系统实现了 **30+ 种量化方法**，对 **6 大类量化对象** 进行压缩，覆盖了从 2-bit 到 8-bit 的各种精度，支持整数量化和浮点量化两大类技术路线。

理解 vLLM 的量化系统需要回答两个核心问题：
- **量化对象（对什么做量化）**：模型中哪些张量/参数被量化？
- **量化方法（怎样做量化）**：用什么技术和算法来量化这些对象？

下面先分别阐述这两个维度，再通过交叉矩阵展示它们的关系。

量化代码位于：

```
vllm/model_executor/layers/quantization/
├── __init__.py              # 注册表和工厂方法
├── base_config.py           # 抽象基类
├── schema.py                # Pydantic 验证模式
├── awq.py / awq_marlin.py / awq_triton.py    # AWQ 系列
├── gptq.py / gptq_marlin.py                  # GPTQ 系列
├── fp8.py / fbgemm_fp8.py / ptpc_fp8.py       # FP8 系列
├── bitsandbytes.py                             # BitsAndBytes
├── gguf.py                                     # GGUF 格式
├── modelopt.py / torchao.py / inc.py           # 框架集成
├── mxfp4.py / petit.py / fp_quant.py           # 特殊格式
├── experts_int8.py / moe_wna16.py / cpu_wna16.py  # MoE/CPU 特化
├── kv_cache.py                                 # KV Cache 量化
├── compressed_tensors/                         # CompressedTensors 框架
│   ├── compressed_tensors.py
│   ├── compressed_tensors_moe.py
│   └── schemes/                               # 11 种量化方案
├── quark/                                      # Quark 量化框架
│   ├── quark.py / quark_moe.py
│   └── schemes/
└── utils/                                      # 24 个工具文件
    ├── marlin_utils.py / fp8_utils.py
    ├── w8a8_utils.py / nvfp4_utils.py
    └── ...
```

**总计：73+ 个 Python 文件**，构成了一个高度模块化、可扩展的量化生态系统。

---

## 2. 量化对象：对什么做量化

在 LLM 推理中，模型内部存在多种不同类型的张量，vLLM 的量化系统对以下 **6 大类对象** 进行量化压缩：

### 2.1 线性层权重（Linear Layer Weights）

**最核心的量化对象。** 线性层是 Transformer 模型中参数量最大的组件，包括：

| 层类型 | 所在模块 | 权重形状 | 参数占比 |
|--------|---------|---------|---------|
| **Q/K/V 投影** | 注意力层 (Attention) | `(hidden, hidden)` × 3 | ~25% |
| **O 投影** | 注意力层 (Attention) | `(hidden, hidden)` | ~8% |
| **Gate/Up 投影** | MLP 层 | `(hidden, intermediate)` × 2 | ~33% |
| **Down 投影** | MLP 层 | `(intermediate, hidden)` | ~17% |
| **LM Head** | 输出层 | `(hidden, vocab_size)` | ~17% |

**在 vLLM 中的实现位置：** `vllm/model_executor/layers/linear.py` 中的 `LinearBase`、`ColumnParallelLinear`、`RowParallelLinear` 等。几乎所有量化方法（AWQ、GPTQ、FP8 等）都以线性层权重为主要量化目标。

**量化的核心操作：**
```python
# LinearBase.__init__() → 为每个线性层分配量化方法
self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

# 推理时 → 量化 GEMM 替代标准矩阵乘法
output = self.quant_method.apply(self, x, bias)
```

### 2.2 激活值（Activations）

**推理时动态量化的对象。** 激活值是线性层的输入张量，在推理过程中实时量化。

| 激活量化类型 | 粒度 | 时机 | 典型方法 |
|------------|------|------|---------|
| **Per-Tensor** | 整个激活张量共享一个 scale | 静态（预计算）| FP8 静态模式 |
| **Per-Token** | 每个 token 一个 scale | 动态（运行时）| FP8 动态模式, W8A8-INT8 |
| **Per-Channel** | 每个输出通道一个 scale | 静态 | PTPC-FP8 |
| **Per-Group** | 按组划分 | 动态 | InputQuant-FP8 |

**在 vLLM 中的实现：** 激活量化发生在 `quant_method.apply()` 的执行过程中。支持激活量化的方法包括：FP8、FBGEMM-FP8、BitsAndBytes（LLM.int8）、CompressedTensors（W8A8 系列）、Quark、ModelOpt 等。

**不是所有量化方法都做激活量化：** AWQ、GPTQ 等仅量化权重（Weight-Only），激活保持 FP16/BF16。

### 2.3 KV Cache（键值缓存）

**注意力机制中的中间状态。** KV Cache 存储了历史 token 的 Key 和 Value 向量，是长序列推理的内存瓶颈。

| 缩放参数 | 含义 | 格式 |
|---------|------|------|
| `k_scale` | Key 缓存的量化缩放因子 | per-tensor 标量 |
| `v_scale` | Value 缓存的量化缩放因子 | per-tensor 标量 |
| `q_scale` | Query 的量化缩放因子（可选）| per-tensor 标量 |
| `prob_scale` | 注意力概率的缩放因子（可选）| per-tensor 标量 |

**在 vLLM 中的实现：** `kv_cache.py` 中的 `BaseKVCacheMethod`。目前仅支持 FP8 格式（E4M3）的 KV Cache 压缩，粒度为 per-tensor。

**支持 KV Cache 量化的方法：** FP8、ModelOpt、CompressedTensors、Quark、PETIT。

### 2.4 MoE 专家权重（Expert Weights）

**MoE 模型中每个专家网络的权重。** 与标准线性层不同，专家权重是 3D 张量，需要逐专家管理量化参数。

| 参数 | 形状 | 说明 |
|------|------|------|
| `w13_qweight` | `(num_experts, 2×intermediate, hidden//pack)` | 门控 + 上投影融合权重 |
| `w2_qweight` | `(num_experts, hidden, intermediate//pack)` | 下投影权重 |
| `w13_scales` | `(num_experts, 2×intermediate, num_groups)` | 逐专家逐组的缩放因子 |
| `w2_scales` | `(num_experts, hidden, num_groups)` | 逐专家逐组的缩放因子 |

**在 vLLM 中的实现：**
- `moe_wna16.py`: GPU 通用 MoE 权重 N-bit 量化
- `experts_int8.py`: MoE INT8 专用量化
- `compressed_tensors/compressed_tensors_moe.py`: CompressedTensors MoE 方案
- `quark/quark_moe.py`: Quark MoE 量化
- 各量化方法中的 `get_quant_method(FusedMoE)` 分支

**支持 MoE 量化的方法：** AWQ、GPTQ、FP8、CompressedTensors、Quark、BitsAndBytes、INC、MXFP4、ModelOpt。

### 2.5 Embedding 层权重

**词嵌入表（Token Embedding）** 通常是模型中最大的单个参数矩阵（`vocab_size × hidden_size`）。

| Embedding 类型 | 说明 | 典型参数量 |
|---------------|------|-----------|
| 输入嵌入 (embed_tokens) | 将 token ID 映射为向量 | vocab_size × hidden_size |
| 输出嵌入 (lm_head) | 可与输入嵌入共享权重（tied） | vocab_size × hidden_size |

**在 vLLM 中的实现：** `QuantizeMethodBase` 提供了可选的 `embedding()` 方法。目前仅 GGUF 和部分方法实现了 Embedding 层量化。大多数方法（AWQ、GPTQ、FP8 等）**跳过** Embedding 层，保持其为 FP16/BF16。

### 2.6 注意力权重 / Softmax 概率（有限支持）

某些高级量化方案也涉及注意力计算中的 Softmax 概率量化：

- `prob_scale`: 注意力概率的 FP8 量化缩放因子（在 `kv_cache.py` 中定义）
- 目前支持有限，主要通过 KV Cache 量化方法间接支持

---

## 3. 量化方法：怎样做量化

vLLM 支持的 30+ 种量化方法可按以下维度分类：

### 3.1 按数据类型分类

| 类别 | 数据类型 | 位宽 | 方法 |
|------|---------|------|------|
| **整数量化** | INT4/INT8 | 2-8 bit | AWQ, GPTQ, BitsAndBytes, INC, GGUF |
| **浮点量化** | FP8 (E4M3) | 8 bit | FP8, FBGEMM-FP8, PTPC-FP8, ModelOpt |
| **浮点量化** | FP4 (NVFP4/MXFP4) | 4 bit | MXFP4, PETIT, FP-Quant, NVFP4 |
| **混合类型** | INT + FP 组合 | 4-8 bit | CompressedTensors, Quark, TorchAO |

### 3.2 按量化对象覆盖范围分类

| 类别 | 量化的对象 | 方法 | 说明 |
|------|----------|------|------|
| **仅权重量化 (W-Only)** | 线性层权重 | AWQ, GPTQ, GGUF | 激活保持 FP16，部署简单 |
| **权重+激活量化 (W+A)** | 权重 + 输入激活 | FP8, W8A8-INT8, CompressedTensors | 更高压缩率，需要硬件支持 |
| **全模型量化** | 权重 + 激活 + KV Cache | FP8+KV, ModelOpt | 最大内存节省 |
| **MoE 专用量化** | 专家权重（3D） | ExpertsInt8, MoE-WNA16 | 针对 MoE 架构优化 |

### 3.3 按量化粒度分类

| 粒度 | 缩放因子数量 | 精度 | 方法/场景 |
|------|------------|------|----------|
| **Per-Tensor** | 1 个标量 | 最低 | FP8 基础模式, KV Cache |
| **Per-Channel** | 每输出通道 1 个 | 中等 | PTPC-FP8, ModelOpt |
| **Per-Group** | 每组 1 个 (组大小 32-128) | 较高 | AWQ, GPTQ, MXFP4 |
| **Per-Token** | 每 token 1 个 | 中等 | FP8 动态模式, W8A8 |
| **Per-Block** | 每块 1 个 (如 128×128) | 最高 | FP8 块量化, DeepGEMM |

### 3.4 按量化时机分类

| 时机 | 说明 | 方法 |
|------|------|------|
| **训练后量化 (PTQ)** | 模型训练完成后离线量化 | AWQ, GPTQ, FP8 静态 |
| **在线量化** | 加载 FP16 模型后在线量化权重 | FP8 动态, TorchAO, BitsAndBytes |
| **运行时量化** | 推理时对激活实时量化 | FP8 per-token, W8A8 动态 |

---

## 4. 量化对象 × 量化方法交叉矩阵

下表展示每种量化方法作用于哪些对象，是理解 vLLM 量化系统的核心参考：

| 量化方法 | 线性层权重 | 激活值 | KV Cache | MoE 专家权重 | Embedding | 量化位宽 |
|---------|:---------:|:-----:|:--------:|:----------:|:---------:|---------|
| **AWQ** | ✓ | ✗ | ✗ | ✓ | ✗ | 4-bit INT |
| **AWQ-Marlin** | ✓ | ✗ | ✗ | ✓ | ✗ | 4-bit INT |
| **AWQ-Triton** | ✓ | ✗ | ✗ | ✗ | ✗ | 4-bit INT |
| **GPTQ** | ✓ | ✗ | ✗ | ✓ | ✗ | 2/3/4/8-bit INT |
| **GPTQ-Marlin** | ✓ | ✗ | ✗ | ✓ | ✗ | 4/8-bit INT |
| **FP8** | ✓ | ✓ (动态/静态) | ✗ | ✓ | ✗ | 8-bit FP |
| **FBGEMM-FP8** | ✓ | ✓ (per-token) | ✗ | ✗ | ✗ | 8-bit FP |
| **PTPC-FP8** | ✓ | ✓ (per-token) | ✗ | ✗ | ✗ | 8-bit FP |
| **InputQuant-FP8** | ✓ | ✓ (多粒度) | ✗ | ✗ | ✗ | 8-bit FP |
| **BitsAndBytes** | ✓ | ✓ (LLM.int8) | ✗ | ✓ | ✗ | 4/8-bit Mixed |
| **GGUF** | ✓ | ✗ | ✗ | ✗ | ✓ | 2-8 bit Mixed |
| **CompressedTensors** | ✓ | ✓ (多方案) | ✓ | ✓ | ✗ | 4-8 bit Mixed |
| **ModelOpt** | ✓ | ✓ (可选) | ✓ | ✓ | ✗ | 4-8 bit FP |
| **TorchAO** | ✓ | ✓ | ✗ | ✗ | ✗ | 灵活 |
| **Quark** | ✓ | ✓ | ✓ | ✓ | ✗ | 4-8 bit Mixed |
| **INC** | ✓ | ✗ | ✗ | ✓ | ✗ | 2-8 bit INT |
| **MXFP4** | ✓ | ✓ | ✗ | ✓ | ✗ | 4-bit FP |
| **PETIT** | ✓ | ✗ | ✓ | ✗ | ✗ | 4-bit FP |
| **FP-Quant** | ✓ | ✓ | ✗ | ✗ | ✗ | 4-bit FP |
| **ExpertsInt8** | ✗ | ✗ | ✗ | ✓ | ✗ | 8-bit INT |
| **MoE-WNA16** | ✗ | ✗ | ✗ | ✓ | ✗ | 4/8-bit INT |
| **CPU-WNA16** | ✓ | ✗ | ✗ | ✗ | ✗ | 4/8-bit INT |

**关键发现：**

1. **线性层权重** 是最普遍的量化对象——几乎所有方法都支持
2. **激活量化** 仅约一半的方法支持，主要是 FP8 系列和 W8A8 类方法
3. **KV Cache 量化** 仅少数方法支持（CompressedTensors, ModelOpt, Quark, PETIT），独立于权重量化
4. **MoE 专家权重** 有专门的量化路径（ExpertsInt8, MoE-WNA16），也可通过通用方法的 MoE 分支处理
5. **Embedding 量化** 目前仅 GGUF 通过 `GGUFEmbeddingMethod` 实现
6. **ExpertsInt8 和 MoE-WNA16** 是纯 MoE 专用方法，不处理标准线性层

---

## 5. 架构设计与核心抽象

### 5.1 抽象基类体系

vLLM 的量化系统采用 **策略模式 (Strategy Pattern)** 和 **工厂模式 (Factory Pattern)** 构建：

```
┌─────────────────────────────────────────────────────┐
│                  QuantizationConfig (ABC)            │
│  定义在: base_config.py                              │
│                                                     │
│  核心方法:                                           │
│  - get_name() → str                                 │
│  - get_supported_act_dtypes() → list[torch.dtype]   │
│  - get_min_capability() → int                       │
│  - from_config(config: dict) → QuantizationConfig   │
│  - get_quant_method(layer, prefix) → QuantizeMethodBase │
│  - get_config_filenames() → list[str]               │
└─────────────┬───────────────────────────────────────┘
              │ 产生
              ▼
┌─────────────────────────────────────────────────────┐
│                QuantizeMethodBase (ABC)              │
│  定义在: base_config.py                              │
│                                                     │
│  核心方法:                                           │
│  - create_weights(layer, ...)                       │
│  - apply(layer, x, bias) → Tensor                  │
│  - process_weights_after_loading(layer)             │
│  - embedding(layer, ...) [可选]                      │
└─────────────────────────────────────────────────────┘
```

**设计要点：**

- **QuantizationConfig** 是配置层抽象，负责解析量化参数并为每个层选择合适的量化方法
- **QuantizeMethodBase** 是运行时抽象，负责权重创建、推理计算和权重后处理
- 两者解耦使得同一种量化配置可以根据层的类型（Linear, Attention, Embedding, MoE）返回不同的量化方法实例

### 5.2 关键设计模式

| 模式 | 应用场景 | 示例 |
|------|---------|------|
| **策略模式** | 每个量化方法封装为独立策略 | AWQLinearMethod, GPTQLinearMethod |
| **工厂模式** | 按名称创建量化配置实例 | `get_quantization_config("awq")` |
| **注册表模式** | 自定义量化方法的动态注册 | `@register_quantization_config` |
| **模板方法** | 基类定义算法骨架，子类填充细节 | `QuantizeMethodBase` 的三步流程 |
| **适配器模式** | 统一不同后端内核的调用接口 | `apply_nvfp4_linear()` 统一 7 种后端 |

---

## 6. 量化方法注册与发现机制

### 6.1 静态注册表

文件 `__init__.py` 定义了完整的量化方法映射表：

```python
QUANTIZATION_METHODS = Literal[
    "awq", "fp8", "fbgemm_fp8", "gptq", "gptq_marlin",
    "awq_marlin", "bitsandbytes", "compressed-tensors",
    "modelopt", "torchao", "gguf", "quark", "inc",
    "mxfp4", "petit", "fp_quant", "ptpc_fp8", ...
]

def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    method_to_config = {
        "awq": AWQConfig,
        "fp8": Fp8Config,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        # ... 30+ 映射
    }
    return method_to_config[quantization]
```

### 6.2 动态注册（自定义量化方法）

```python
@register_quantization_config("my_custom_quant")
class MyCustomQuantConfig(QuantizationConfig):
    # 自定义实现
    pass
```

这个装饰器将新的量化方法注入到注册表中，使其可通过 `--quantization my_custom_quant` 使用。

### 6.3 自动检测与覆盖

系统支持从模型 checkpoint 自动检测量化方法：

```python
# config/model.py
def _verify_quantization(self):
    hf_quant_cfg = getattr(self.hf_config, "quantization_config", None)
    quant_method = hf_quant_cfg.get("quant_method")
    # 自动检测并可能覆盖用户指定的方法
    quant_cfg = override_quantization_method(hf_quant_cfg, quant_method)
```

---

## 7. 完整调用流程

### 7.1 端到端流程图

```
用户命令: python -m vllm.entrypoints.openai.api_server --model model_path --quantization awq
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════╗
║  阶段 1: 配置解析 (vllm/config/model.py)                    ║
║                                                            ║
║  ModelConfig.__init__():                                   ║
║    → self.quantization = "awq"                             ║
║    → _verify_quantization():                               ║
║       ├── 从 HF config 读取 quantization_config             ║
║       ├── get_quantization_config("awq") → AWQConfig 类     ║
║       └── AWQConfig.from_config(hf_quant_cfg) → 实例化      ║
╚══════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════╗
║  阶段 2: 模型构建 (vllm/model_executor/models/*.py)         ║
║                                                            ║
║  LlamaForCausalLM.__init__(vllm_config):                   ║
║    → LlamaDecoderLayer(vllm_config):                       ║
║       → quant_config = self.get_quant_config(vllm_config)  ║
║       → LlamaMLP(quant_config=quant_config):               ║
║          → MergedColumnParallelLinear(quant_config=...):    ║
║             → LinearBase.__init__():                       ║
║                → self.quant_method =                       ║
║                   quant_config.get_quant_method(self, prefix) ║
║                   # 返回 AWQLinearMethod 实例               ║
╚══════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════╗
║  阶段 3: 权重创建与加载                                      ║
║  (vllm/model_executor/model_loader/)                       ║
║                                                            ║
║  quant_method.create_weights(layer, ...):                  ║
║    → 创建量化参数 (qweight, qzeros, scales 等)              ║
║    → 注册到 layer 的 parameters 中                          ║
║                                                            ║
║  load_weights() → weight_loader():                         ║
║    → 从 checkpoint 加载量化权重                              ║
║    → 处理 tensor parallel 分片                               ║
║                                                            ║
║  quant_method.process_weights_after_loading(layer):        ║
║    → 权重格式转换 (如 AWQ→Marlin 格式)                       ║
║    → ExLlama shuffle / 零点转换                             ║
║    → 设置运行时状态                                          ║
╚══════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════╗
║  阶段 4: 推理 (vllm/model_executor/layers/linear.py)        ║
║                                                            ║
║  LinearBase.forward(x):                                    ║
║    → output = self.quant_method.apply(self, x, bias)       ║
║       ├── 激活量化 (如 FP8 per-token 量化)                   ║
║       ├── 反量化 + 矩阵乘法 (融合内核)                       ║
║       │   或 量化 GEMM 内核                                 ║
║       └── 返回推理结果                                       ║
╚══════════════════════════════════════════════════════════════╝
```

### 7.2 get_quant_method 的分发逻辑

```python
# 在每个 QuantizationConfig 子类中实现
class AWQConfig(QuantizationConfig):
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return AWQMoEMethod(self)       # MoE 特化路径
        elif isinstance(layer, Attention):
            return None                      # 不量化注意力
        return None                          # 不量化其他层
```

**关键设计：** `prefix` 参数（如 `"model.layers.0.mlp.gate_up_proj"`）允许量化方法根据层的名称决定是否量化。某些方法（如 ModelOpt）使用通配符模式排除特定层。

---

## 8. 主要量化方法详解

### 8.1 AWQ (Activation-aware Weight Quantization)

**原理：** AWQ 是一种激活感知的权重量化方法，通过分析激活分布来确定哪些权重通道更重要，从而在量化时保持关键通道的精度。

**核心参数：**
- **weight_bits**: 4-bit（固定，打包为 int32，pack_factor=8）
- **group_size**: 量化组大小（32/64/128/-1）
- **zero_point**: 是否使用零点（非对称量化）

**存储格式：**
```python
qweight:  [input_size, output_size // 8]    # int32，8 个 4-bit 权重打包
qzeros:   [num_groups, output_size // 8]    # int32，每组零点
scales:   [num_groups, output_size]          # fp16/bf16，每组缩放因子
```

**量化/反量化公式：**
```
量化:   q = round(w / scale) + zero_point    (存储为 4-bit 无符号整数)
反量化: w' = (q - zero_point) * scale
```

#### AWQ 三种后端实现

| 特性 | AWQ (基础) | AWQ-Marlin | AWQ-Triton |
|------|-----------|-----------|-----------|
| **内核** | CUDA C++ 自定义 ops | Marlin 高性能内核 | Triton JIT 编译 |
| **策略** | 启发式选择 | 始终融合 | 可配置 |
| **激活类型** | fp16 | fp16, bf16, int8, fp8 | fp16 |
| **格式转换** | 无 | 加载时 AWQ→Marlin | 无 |
| **性能** | 良好 | 最优 | 良好（可移植） |

**AWQ 基础实现的启发式策略：**
```python
# awq.py
FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

if FP16_MATMUL_HEURISTIC_CONDITION:
    # 大批量: 先反量化再 GEMM（利用 tensor core）
    out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
    out = torch.matmul(reshaped_x, out)
else:
    # 小批量: 融合反量化 + GEMM 内核
    out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
```

**AWQ-Triton 的 4-bit 解包算法：**
```python
@triton.jit
def awq_dequantize_kernel(...):
    # 交错展开打包权重
    iweights = tl.interleave(iweights, iweights)  # 3 次交错 → 8x 展开
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # AWQ 顺序: [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_awq_order = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)
    shifts = reverse_awq_order * 4

    # 提取并反量化
    iweights = (iweights >> shifts) & 0xF         # 提取 4-bit
    zeros = (zeros >> shifts) & 0xF               # 提取零点
    result = (iweights - zeros) * scales           # 反量化
```

---

### 8.2 GPTQ 量化

**原理：** GPTQ 基于最优脑损伤（OBD）理论，使用二阶信息（Hessian 矩阵）来最小化量化误差。它逐列量化权重矩阵，同时用 Hessian 信息补偿未量化列的误差。

**核心参数：**
- **weight_bits**: 2/3/4/8-bit（比 AWQ 更灵活）
- **group_size**: 量化组大小（-1 表示按通道量化）
- **desc_act**: 是否按激活大小排序权重（提高精度但增加开销）
- **sym**: 是否对称量化

**与 AWQ 的关键区别：**

| 方面 | GPTQ | AWQ |
|------|------|-----|
| 位宽 | 2/3/4/8-bit | 仅 4-bit |
| 量化策略 | Hessian 优化逐列量化 | 激活感知权重裁剪 |
| 激活排序 | 支持 (desc_act) | 不支持 |
| GPU 要求 | SM 6.0+ | SM 7.5+ |
| 理论基础 | 最优脑损伤 (OBD) | 激活统计分析 |

**GPTQ 的 ExLlama 优化：**
```python
# gptq.py - 权重后处理
def process_weights_after_loading(self, layer):
    if self.quant_config.desc_act:
        # 按激活大小排序权重
        layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
        # ExLlama shuffle 优化内存访问模式
        ops.gptq_shuffle(layer.qweight, layer.g_idx, self.quant_config.weight_bits)
```

**GPTQ-Marlin 优化：**
- 将 GPTQ 权重重新打包为 Marlin 优化格式
- 支持 FP16/BF16 激活（基础 GPTQ 仅支持 FP16）
- 通过 `choose_mp_linear_kernel()` 自动选择最优内核
- MoE 支持通过 `GPTQMarlinMoEMethod` 实现

---

### 8.3 FP8 量化

**原理：** FP8 使用 8-bit 浮点数表示权重和/或激活。相比整数量化，浮点量化更好地保持了数值的动态范围。

#### FP8 数据格式

| 格式 | 指数位 | 尾数位 | 范围 | 用途 |
|------|-------|-------|------|------|
| E4M3 (float8_e4m3fn) | 4 | 3 | ±240 | 权重 + 激活（默认） |
| E5M2 (float8_e5m2) | 5 | 2 | ±57344 | 训练梯度（参考） |
| E4M3FNUZ (ROCm) | 4 | 3 | ±224 | AMD GPU 专用 |

**量化公式：**
```
量化:   q = clamp(x / scale, FP8_MIN, FP8_MAX).to(float8_e4m3fn)
反量化: x' = q.to(float16) * scale
缩放因子计算: scale = max(|x|) / FP8_MAX
```

#### 量化粒度

| 粒度 | 缩放因子形状 | 精度 | 硬件要求 |
|------|------------|------|---------|
| Per-Tensor `(-1, -1)` | `(1, 1)` 标量 | 最低 | 通用 |
| Per-Channel `(-1, 1)` | `(out_channels, 1)` | 中等 | Hopper+ |
| Per-Token `(1, -1)` | `(batch*seq, 1)` | 中等 | 动态计算 |
| Per-Block `(B_k, B_n)` | 块级缩放 | 最高 | Hopper+ |

#### FP8 四种实现

**1. Fp8Config (fp8.py) - 核心实现：**
```python
class Fp8Config:
    is_checkpoint_fp8_serialized: bool  # 检查点是否已 FP8 序列化
    activation_scheme: str              # "static" 或 "dynamic"
    weight_block_size: list[int]        # 块量化大小 (如 [128, 128])
```

- **静态激活量化**：使用预计算的激活缩放因子（存储在 checkpoint 中）
- **动态激活量化**：运行时 per-token 计算缩放因子
- 支持 `torch._scaled_mm`（原生 FP8 GEMM）和 Marlin 后备

**2. FBGEMMFp8Config (fbgemm_fp8.py)：**
- FBGEMM 后端实现的 FP8 量化
- 动态 per-token 激活量化
- 旧 GPU 回退到 Marlin 内核

**3. PTPCFp8Config (ptpc_fp8.py)：**
- Per-Token-Per-Channel 量化（AMD MI300 专用）
- 权重 per-channel + 激活 per-token 的组合
- 通过 `torch._scaled_mm`（hipBLASLt）实现

**4. InputQuantFp8Config (input_quant_fp8.py)：**
- 支持 per-tensor、per-token、per-channel、per-group 多种激活量化粒度
- 通过自定义 op `QuantFP8` 实现灵活的量化策略

#### FP8 后端选择

```
GPU 能力检查
  ├── SM 9.0+ (Hopper): CUTLASS Scaled MM / DeepGEMM
  │   ├── 块量化 (128x128): W8A8BlockFp8LinearOp
  │   └── 张量量化: Fp8LinearMethod
  ├── SM 8.9+ (Ada): torch._scaled_mm
  └── SM < 8.9: Marlin FP8 (仅权重量化)
```

---

### 8.4 CompressedTensors 统一框架

**原理：** CompressedTensors 是一个配置驱动的统一量化框架，允许不同层使用不同的量化方案，通过 JSON 配置指定每个层的量化策略。

#### 架构设计

```
CompressedTensorsConfig
  ├── target_scheme_map: {layer_pattern → QuantizationScheme}
  ├── sparsity_scheme_map: {layer_pattern → SparsityScheme}
  └── get_scheme(layer, prefix) → CompressedTensorsScheme
        ├── 模式匹配层名称
        ├── 提取 QuantizationArgs (权重 + 激活)
        └── 智能选择量化方案
```

#### 11 种量化方案

| 方案 | 权重 | 激活 | 描述 |
|------|------|------|------|
| **WNA16** | 4/8-bit INT | 16-bit Float | Marlin 内核，权重仅量化 |
| **W8A16_Fp8** | 8-bit FP8 | 16-bit Float | FP8 权重，支持张量/通道/块级 |
| **W8A8_Fp8** | 8-bit FP8 | 8-bit FP8 | 全 FP8，静态/动态模式 |
| **W8A8_Int8** | 8-bit INT8 | 8-bit INT8 | 对称/非对称 INT8 |
| **W4A8_Fp8** | 4-bit | 8-bit FP8 | Hopper (SM 90+) 专用 |
| **W4A8_Int** | 4-bit INT | 8-bit INT | 动态 token 级量化 |
| **W4A16_NvFp4** | 4-bit NVFP4 | 16-bit | 张量组量化，group_size=16 |
| **W4A16_MxFp4** | 4-bit MXFP4 | 16-bit | 组策略量化，group_size=32 |
| **W4A4_NvFp4** | 4-bit FP4 | 4-bit FP4 | 全 4-bit，最大压缩 |
| **2:4 稀疏** | N/A | N/A | 结构化稀疏 + 可选量化 |
| **Scheme** (基类) | - | - | 抽象接口 |

#### 方案智能选择逻辑

```python
def get_scheme(layer, prefix):
    # 1. 匹配层名称到配置
    weight_args, input_args = match_target(prefix)

    # 2. 按条件选择方案（优先级从高到低）
    if is_nvfp4_tensor_group:    return CompressedTensorsW4A16Fp4
    if is_mxfp4_group:           return CompressedTensorsW4A16Mxfp4
    if is_w4a8_fp8_sm90:         return CompressedTensorsW4A8Fp8
    if is_wna16_packed:          return CompressedTensorsWNA16
    if is_w4a4_fp4:              return CompressedTensorsW4A4Fp4
    if is_w8a8_fp8:              return CompressedTensorsW8A8Fp8  # 或回退到 W8A16
    if is_w8a16_fp8:             return CompressedTensorsW8A16Fp8
    if is_w8a8_int8:             return CompressedTensorsW8A8Int8
    if is_w4a8_int:              return CompressedTensorsW4A8Int
    raise ValueError("不支持的量化参数组合")
```

---

### 8.5 BitsAndBytes 量化

**原理：** BitsAndBytes 提供了两种量化模式：LLM.int8() 和 4-bit NF4/FP4 量化，针对消费级 GPU 优化。

**LLM.int8() (8-bit)：**
- 使用动态量化状态 (MatmulLtState)
- 支持 FP32 CPU offload
- 按分片管理量化状态

**4-bit NF4/FP4：**
- NF4 (Normal Float 4)：基于正态分布的最优 4-bit 数据类型
- FP4：标准 4-bit 浮点
- 双重量化 (double quantization)：对量化常数再次量化

```python
class BitsAndBytesConfig:
    load_in_8bit: bool           # 启用 8-bit 模式
    load_in_4bit: bool           # 启用 4-bit 模式
    bnb_4bit_quant_type: str     # "nf4" 或 "fp4"
    llm_int8_threshold: float    # 离群值检测阈值
```

**特点：** 最小 GPU 要求 (SM 7.0+)，适合消费级 GPU 和微调场景。

---

### 8.6 GGUF 量化

**原理：** GGUF (GGML Unified Format) 来自 llama.cpp 生态系统，支持 20+ 种量化方案，特别适合 CPU 和边缘设备部署。

**支持的量化类型：**
- **标准**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- **K-Quants**: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K（针对特定块大小优化）
- **I-Matrix**: IQ1_M/S, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL（重要性矩阵量化）
- **未量化**: F32, F16, BF16

**三种推理路径：**
```python
# gguf.py - 内核选择
if seq_len <= MMVQ_THRESHOLD:      # 小批量（≤8-16 tokens）
    kernel = "MMVQ"                # MatMul Vec Quantized
elif quant_type in K_QUANTS:       # K-Quants 支持的类型
    kernel = "MMQ"                 # MatMul Quantized
else:
    kernel = "DEQUANTIZE"          # 回退：先反量化
```

**特点：** 最低 GPU 要求 (SM 6.0+)，支持 Embedding 层量化，llama.cpp 兼容。

---

### 8.7 ModelOpt (NVIDIA 量化框架)

**原理：** ModelOpt 是 NVIDIA 官方的模型优化框架，提供多种量化算法和自动化工具。

**支持的量化算法：**
- **FP8**: Per-tensor 权重 + 可选静态激活缩放
- **FP8_PER_CHANNEL_PER_TOKEN**: Per-channel 权重 + 动态 per-token 激活
- **FP8_PB_WO**: Per-block 仅权重 FP8
- **NVFP4**: NVIDIA 4-bit 浮点
- **MXFP8**: 矩阵指数 FP8
- **MIXED_PRECISION**: 混合精度组合

**特点：** 要求 SM 8.9+ (Ada/Hopper)，支持 KV-Cache FP8 量化，Oracle 后端自动选择。

---

### 8.8 TorchAO 量化

**原理：** TorchAO 是 PyTorch 原生的量化库，提供灵活的模块级量化控制。

**核心特性：**
- 使用 `AOBaseConfig` 配置系统
- 支持正则表达式模块匹配（如 `re:model.layers.\d+.q_proj`）
- 检测 checkpoint 是否已预量化，避免重复量化
- 硬件感知的张量打包 (`convert_to_packed_tensor_based_on_current_hardware()`)

```python
# 模块级精度控制
ModuleFqnToConfig = {
    "re:model.layers.\d+.q_proj": Int8QuantConfig(),
    "re:model.layers.\d+.mlp":   Int4QuantConfig(),
    "lm_head": None,  # 不量化
}
```

---

### 8.9 Quark 量化框架

**原理：** Quark 是一个模块化的量化框架，采用可插拔的量化方案设计。

**支持的方案：**
- **W8A8-FP8**: Per-token/channel 动态激活 + 静态权重
- **W8A8-INT8**: 整数 8-bit 权重和激活
- **OCP-MX**: Open Compute Platform 混合指数格式
- **Dynamic MXFP4**: DeepSeek-V3 动态 FP4 量化

**架构：**
```
quark/
├── quark.py           # 主配置和线性方法
├── quark_moe.py       # MoE 专用方法
├── utils.py           # 工具函数
└── schemes/
    ├── quark_scheme.py        # 抽象基类
    ├── quark_w8a8_fp8.py      # W8A8 FP8 方案
    ├── quark_w8a8_int8.py     # W8A8 INT8 方案
    └── quark_ocp_mx.py        # OCP 混合指数方案
```

---

### 8.10 其他量化方法

#### MXFP4 (Mixed-precision FP4)
- 多后端支持（FlashInfer, Marlin, Triton, CK）
- 针对 H100/Blackwell GPU 优化
- 支持 MoE 模型

#### PETIT
- AMD GPU (ROCm) 专用的 NVFP4 实现
- SE2M1 位格式，2 个 FP4 值打包为 uint8
- 使用 FP8-E4M3 缩放因子

#### FP-Quant
- 带 Hadamard 变换的浮点量化
- 支持 mxfp4 和 nvfp4 两种执行模式
- 要求 SM 10.0+

#### INC (Intel Neural Compressor)
- 与 Auto-Round 兼容，支持 GPTQ/AWQ 后端
- 2/3/4/8-bit 整数量化
- 正则表达式层级配置
- 针对 Intel CPU/XPU 优化

#### CPU WNA16
- CPU 专用的权重 N-bit/激活 16-bit 量化
- 使用 AMX ISA 提示优化
- int32 打包张量

---

## 9. MoE 模型量化

### 9.1 MoE 与标准层的量化差异

MoE (Mixture-of-Experts) 模型的量化在多个维度上不同于标准线性层：

| 方面 | 标准线性层 | MoE 层 |
|------|----------|--------|
| 权重维度 | 2D `(in, out)` | 3D `(experts, out, in)` |
| 缩放粒度 | 单一或逐组 | 逐专家 + 逐组 |
| 激活缩放 | 张量或通道级 | 逐专家级 |
| 投影处理 | 统一 | gate/up (w13) vs down (w2) 分离 |
| 内核变体 | 少量 | 多种 (Marlin, FlashInfer, NVFP4) |

### 9.2 MoE 量化实现

**主要文件：**
- `moe_wna16.py`: GPU 通用 MoE 权重 N-bit 量化
- `compressed_tensors/compressed_tensors_moe.py`: CompressedTensors MoE 支持
- `quark/quark_moe.py`: Quark 框架 MoE 支持

**权重存储结构：**
```python
# moe_wna16.py
w13_qweight = torch.empty(
    (num_experts, 2*intermediate_size, hidden_size//pack_factor),
    dtype=torch.uint8
)  # gate + up 融合权重

w2_qweight = torch.empty(
    (num_experts, hidden_size, intermediate_size//pack_factor),
    dtype=torch.uint8
)  # down 投影权重
```

**自适应 group_size：**
```python
# 当 intermediate_size 不能整除 group_size 时动态缩小
while intermediate_size_per_partition % group_size or hidden_size % group_size:
    group_size = group_size // 2
    group_size_div_factor *= 2
```

### 9.3 MoE 内核格式差异

不同内核对 MoE 权重有不同的布局要求：
```python
# Flashinfer: (num_experts, 2*intermediate, hidden//pack)
# Marlin:     (num_experts, hidden//pack, 2*intermediate)  # 转置
```

权重加载时需要根据目标内核进行格式转换和重新打包。

---

## 10. KV Cache 量化

**文件：** `kv_cache.py`

KV Cache 量化通过将注意力机制的 Key/Value 缓存压缩为 FP8 格式来减少内存使用。

**核心参数：**
```python
class BaseKVCacheMethod:
    def create_weights(self, layer):
        # 初始化缩放因子（默认 -1.0 表示未设置）
        layer.q_scale = nn.Parameter(torch.tensor(-1.0))
        layer.k_scale = nn.Parameter(torch.tensor(-1.0))
        layer.v_scale = nn.Parameter(torch.tensor(-1.0))
        layer.prob_scale = nn.Parameter(torch.tensor(-1.0))
```

**平台处理：**
```python
# ROCm FP8 格式差异处理
if current_platform.is_fp8_fnuz():
    # e4m3fnuz 没有负零，需要 scale×2 调整
    k_scale *= 2.0
    v_scale *= 2.0
```

**约束：** 仅支持 per-tensor（标量）缩放因子，确保每个 K/V 头有一个缩放值。

---

## 11. 工具层与内核支持

### 11.1 工具文件概览

`utils/` 目录包含 24 个文件，构成量化系统的中间件层：

```
工具层架构:
┌────────────────────────────────────────────┐
│  量化方法实现层                              │
│  (awq.py, gptq.py, fp8.py 等)              │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│  工具层 (utils/)                            │
│  ├── 后端选择 (select_backend)              │
│  ├── 数据变换 (permute, repack, pad)        │
│  ├── 内核分发 (kernel dispatch)             │
│  ├── 格式转换 (format conversion)           │
│  └── 硬件检测 (capability check)            │
└───────────┬────────────────────────────────┘
            │
┌───────────▼────────────────────────────────┐
│  CUDA/Triton 内核层                         │
│  ├── cutlass_scaled_mm (CUTLASS GEMM)       │
│  ├── marlin_gemm (Marlin 内核)              │
│  ├── gptq_gemm / awq_gemm (自定义 GEMM)    │
│  ├── triton 内核 (JIT 编译)                 │
│  └── torch._scaled_mm (PyTorch 原生)        │
└────────────────────────────────────────────┘
```

### 11.2 关键工具文件

**marlin_utils.py：**
- 支持 uint4, uint8, FP8, FP4 等量化类型
- 组大小约束：-1（通道级）、32、64、128
- 权重/缩放因子排列 (permutation) 和工作空间分配
- 最低 SM 7.5 要求

**fp8_utils.py（最大的工具文件，52.7KB）：**
- 注册 `cutlass_scaled_mm`、`w8a8_triton_block_scaled_mm` 等自定义 ops
- 智能批量分发逻辑：
  ```python
  M < 32  → FlashInfer DeepGEMM (swapAB)
  M >= 32 → 官方 DeepGEMM 内核
  ```
- 块级 FP8 量化 (`W8A8BlockFp8LinearOp`) 和 Triton per-token 量化

**nvfp4_utils.py：**
- 7 种后端的统一接口：
  ```python
  NvFp4LinearBackend:
    VLLM_CUTLASS | FLASHINFER_CUTLASS | FLASHINFER_TRTLLM |
    FLASHINFER_CUDNN | FBGEMM | MARLIN | EMULATION
  ```
- 自动后端选择策略和权重填充/变换

**w8a8_utils.py：**
- GPU 能力检查（稀疏 CUTLASS、FP8 CUTLASS、块 FP8 支持）
- per-tensor 到 per-channel 缩放转换
- ROCm e4m3fn ↔ e4m3fnuz 格式规范化

---

## 12. 量化方法对比总结

### 12.1 完整对比表

| 方法 | 位宽 | 类型 | GPU 要求 | 激活量化 | MoE | 最佳场景 |
|------|------|------|---------|---------|-----|---------|
| **AWQ** | 4-bit | INT | SM 7.5+ | 无 | ✓ | 通用 4-bit 部署 |
| **AWQ-Marlin** | 4-bit | INT | SM 7.5+ | 无 | ✓ | 高性能推理 |
| **GPTQ** | 2-8 bit | INT | SM 6.0+ | 无 | ✓ | 灵活位宽选择 |
| **GPTQ-Marlin** | 4/8-bit | INT | SM 7.5+ | 无 | ✓ | GPTQ 加速推理 |
| **FP8** | 8-bit | FP | SM 8.9+ | ✓ | ✓ | Hopper GPU 最优精度 |
| **FBGEMM-FP8** | 8-bit | FP | SM 7.5+ | ✓ | ✗ | FBGEMM 生态集成 |
| **BitsAndBytes** | 4/8-bit | Mixed | SM 7.0+ | ✓ | ✓ | 消费级 GPU |
| **GGUF** | 2-8 bit | Mixed | SM 6.0+ | 无 | ✗ | CPU/边缘设备 |
| **CompressedTensors** | 4-8 bit | Mixed | SM 7.0+ | ✓ | ✓ | 配置驱动灵活部署 |
| **ModelOpt** | 4-8 bit | FP | SM 8.9+ | ✓ | ✓ | NVIDIA 官方优化 |
| **TorchAO** | 灵活 | Mixed | SM 7.5+ | ✓ | ✗ | PyTorch 原生集成 |
| **Quark** | 4-8 bit | Mixed | SM 7.0+ | ✓ | ✓ | 模块化量化研究 |
| **INC** | 2-8 bit | INT | 通用 | 无 | ✓ | Intel 平台优化 |
| **MXFP4** | 4-bit | FP | SM 9.0+ | ✓ | ✓ | 最新 GPU 极致性能 |

### 12.2 性能排序（典型场景）

**4-bit 权重量化（无激活量化）：**
```
AWQ-Marlin > GPTQ-Marlin > AWQ-Triton > AWQ > GPTQ
   (最快)                                    (最慢)
```

**8-bit 全量化（权重 + 激活）：**
```
FP8 (CUTLASS) > FP8 (Marlin) > W8A8-INT8
    (Hopper)      (通用)         (通用)
```

### 12.3 选择指南

```
你的 GPU 是什么？
├── Hopper (H100/H200)
│   ├── 追求最高性能 → FP8 + CUTLASS
│   ├── 最大压缩 → MXFP4 / NVFP4
│   └── 混合精度 → CompressedTensors W4A8
├── Ada (L40/4090)
│   ├── 通用推理 → AWQ-Marlin / GPTQ-Marlin
│   └── FP8 支持 → FP8 (Marlin fallback)
├── Ampere (A100/3090)
│   ├── 推荐 → AWQ-Marlin / GPTQ-Marlin
│   └── INT8 → W8A8-INT8
├── 消费级 (3060/3070)
│   └── 推荐 → BitsAndBytes NF4
├── AMD GPU
│   └── 推荐 → PTPC-FP8 / PETIT
└── CPU/边缘设备
    └── 推荐 → GGUF / INC
```

---

## 13. 关键发现与洞察

### 13.1 架构发现

1. **极高的模块化程度**：73+ 个 Python 文件，每种量化方法独立实现，通过统一接口集成。
2. **多层抽象**：Config → Method → Utils → Kernel 四层架构，每层职责清晰。
3. **注册表模式**：支持运行时动态注册自定义量化方法，无需修改核心代码。
4. **硬件感知设计**：每种方法声明最低 GPU 能力，运行时自动检查和回退。

### 13.2 技术发现

1. **Marlin 内核是性能关键**：AWQ、GPTQ、FP8 都有 Marlin 后端变体，加速 2-3 倍。
2. **动态内核选择**：根据批量大小、序列长度和 GPU 型号自动选择最优内核。例如 AWQ 在 batch≥256 时切换到先反量化再 GEMM。
3. **格式转换开销**：某些方法（如 AWQ-Marlin）在权重加载时进行格式转换（一次性开销），换取推理时的性能提升。
4. **MoE 专用路径**：MoE 模型需要逐专家的缩放因子和自适应 group_size，增加了量化复杂度。
5. **平台差异处理**：CUDA 和 ROCm 的 FP8 格式不同（e4m3fn vs e4m3fnuz），需要特殊的缩放因子调整。

### 13.3 工程发现

1. **CompressedTensors 是最灵活的框架**：支持 11 种方案，配置驱动，不同层可用不同量化。
2. **后端爆炸**：NVFP4 有 7 种后端，MXFP4 有 6 种后端，反映了不同硬件的优化需求。
3. **工具层是胶水**：24 个 utils 文件在量化方法和 CUDA 内核之间提供适配，是理解代码的关键。
4. **prefix 机制**：通过层名称前缀（如 `model.layers.0.mlp.gate_up_proj`）决定量化策略，实现细粒度控制。
5. **权重后处理（process_weights_after_loading）** 是很多量化方法的关键步骤，执行格式转换、排列优化等操作。

### 13.4 趋势观察

1. **从整数量化到浮点量化**：FP8/FP4 正在取代 INT4/INT8，提供更好的动态范围。
2. **混合精度量化兴起**：W4A8、W4A4 等组合方案越来越多。
3. **块级量化增长**：从 per-tensor 到 per-channel 到 per-block，粒度越来越细。
4. **MoE 量化成为焦点**：随着 MoE 模型普及，专用量化方案持续增加。
5. **硬件-软件协同设计**：新量化方法（MXFP4, NVFP4）与特定 GPU 架构（Hopper, Blackwell）紧密绑定。

---

*报告生成日期：2026-03-05*
*基于 vLLM 代码库分析*
