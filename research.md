# vLLM 编译系统深度分析：CompilationMode 工作流程研究报告

## 目录

1. [概述](#1-概述)
2. [vLLM 编译系统架构总览](#2-vllm-编译系统架构总览)
3. [CompilationMode 枚举定义与语义](#3-compilationmode-枚举定义与语义)
4. [核心组件详解](#4-核心组件详解)
5. [四种编译模式的详细工作流程](#5-四种编译模式的详细工作流程)
6. [模式间异同点对比分析](#6-模式间异同点对比分析)
7. [优化级别与编译模式的关系](#7-优化级别与编译模式的关系)
8. [CUDA Graph 与编译模式的交互](#8-cuda-graph-与编译模式的交互)
9. [编译缓存系统](#9-编译缓存系统)
10. [端到端编译流程](#10-端到端编译流程)
11. [平台特定行为](#11-平台特定行为)
12. [总结](#12-总结)

---

## 1. 概述

vLLM 是一个高性能的大语言模型（LLM）推理引擎。为了最大化推理性能，vLLM 实现了一套基于 `torch.compile` 的多层次编译系统。该系统通过 `CompilationMode` 枚举提供四种编译策略，从完全不编译的急切执行模式到高度定制化的 vLLM 专用编译模式，覆盖了不同场景下的性能与灵活性需求。

### 编译系统的设计目标

- **性能优化**：通过 Inductor 后端生成优化的 Triton 内核，减少 Python 开销
- **分片编译（Piecewise Compilation）**：将计算图按 attention 算子等分裂点切分，支持灵活的 CUDA Graph 捕获
- **编译缓存**：避免重复编译，支持 AOT（Ahead-of-Time）编译产物的持久化存储
- **自定义优化 Pass**：在 Inductor 后端的 post-grad 阶段注入 vLLM 特有的融合优化
- **Guard 消除**：消除 Dynamo 的重编译 guard，确保模型只编译一次

### 核心源码文件

| 文件路径 | 功能 |
|---------|------|
| `vllm/config/compilation.py` | `CompilationMode`、`CUDAGraphMode` 枚举定义，`CompilationConfig` 配置类 |
| `vllm/config/vllm.py` | `VllmConfig` 全局配置，编译模式自动选择逻辑 |
| `vllm/compilation/backends.py` | `VllmBackend` 编译后端，图切分逻辑，`CompilerManager` |
| `vllm/compilation/wrapper.py` | `TorchCompileWithNoGuardsWrapper`，torch.compile 包装器 |
| `vllm/compilation/decorators.py` | `@support_torch_compile` 装饰器 |
| `vllm/compilation/piecewise_backend.py` | `PiecewiseBackend` 分片编译后端 |
| `vllm/compilation/compiler_interface.py` | `CompilerInterface`、`InductorAdaptor`、`EagerAdaptor` |
| `vllm/compilation/caching.py` | 编译产物的缓存与序列化 |
| `vllm/compilation/cuda_graph.py` | CUDA Graph 捕获与重放 |
| `vllm/compilation/passes/pass_manager.py` | `PostGradPassManager` 优化 Pass 管理器 |

---

## 2. vLLM 编译系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户配置                                       │
│  VllmConfig(optimization_level=O2, compilation_config=...)          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                VllmConfig.__post_init__()                             │
│  • 自动选择 CompilationMode                                          │
│  • 应用 OptimizationLevel 默认值                                     │
│  • 验证 CUDAGraph 兼容性                                             │
│  • 设置 custom_ops、splitting_ops                                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 模型加载 & @support_torch_compile                     │
│  • 装饰器注入 TorchCompileWithNoGuardsWrapper                        │
│  • 标记动态维度 (mark_dynamic / mark_unbacked)                        │
│  • 调用 CompilationConfig.init_backend() 初始化后端                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │   NONE   │ │  STOCK   │ │ DYNAMO_TRACE │
        │  不编译   │ │  标准编译 │ │   单次追踪    │
        └──────────┘ └──────────┘ └──────────────┘
                                        │
                           ┌────────────┘
                           ▼
                    ┌──────────────┐
                    │ VLLM_COMPILE │
                    │  vLLM 定制   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │ 图切分    │ │ 分片编译  │ │ 自定义 Pass   │
        │split_graph│ │Piecewise │ │PostGradPass  │
        └──────────┘ └──────────┘ └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ CUDA Graph   │
                    │ 捕获 & 重放   │
                    └──────────────┘
```

---

## 3. CompilationMode 枚举定义与语义

### 3.1 枚举定义

`CompilationMode` 定义在 `vllm/config/compilation.py` 中，是一个 `IntEnum`：

```python
class CompilationMode(enum.IntEnum):
    """The compilation approach used for torch.compile-based compilation of the model."""

    NONE = 0
    """No torch.compile compilation is applied, model runs in fully eager pytorch mode."""

    STOCK_TORCH_COMPILE = 1
    """The standard `torch.compile` compilation pipeline."""

    DYNAMO_TRACE_ONCE = 2
    """Single Dynamo trace through the model, avoiding recompilation."""

    VLLM_COMPILE = 3
    """Custom vLLM Inductor-based backend with caching, piecewise compilation,
    shape specialization, and custom passes."""
```

`IntEnum` 的特性使得可以进行数值比较，例如 `mode < CompilationMode.VLLM_COMPILE`，这在代码中被广泛使用来判断编译等级。

### 3.2 各模式的核心语义

| 模式 | 数值 | 后端 | Guard 策略 | 图切分 | 自定义 Pass | 用途 |
|------|------|------|-----------|--------|------------|------|
| **NONE** | 0 | 无 | 无 | 无 | 无 | 调试、最快启动 |
| **STOCK_TORCH_COMPILE** | 1 | PyTorch 原生 | **保留所有 Guard** | 无 | 无 | 标准 PyTorch 编译 |
| **DYNAMO_TRACE_ONCE** | 2 | PyTorch 原生 | **丢弃所有 Guard** | 无 | 无 | 单次追踪，避免重编译 |
| **VLLM_COMPILE** | 3 | VllmBackend | **丢弃所有 Guard** | 有（分片编译） | 有 | 生产部署、最高性能 |

---

## 4. 核心组件详解

### 4.1 CompilationConfig 配置类

`CompilationConfig`（`vllm/config/compilation.py`）是编译系统的中心配置，包含三大配置域：

#### 顶层编译控制

| 字段 | 类型 | 说明 |
|------|------|------|
| `mode` | `CompilationMode` | 编译模式 |
| `backend` | `str` | 后端名称（`""`=默认, `"inductor"`, `"eager"`, `"openxla"` 等） |
| `custom_ops` | `list[str]` | 自定义算子启用/禁用列表 |
| `splitting_ops` | `list[str] \| None` | 图切分算子列表（用于分片编译） |
| `compile_mm_encoder` | `bool` | 是否编译多模态编码器 |

#### Inductor 编译控制

| 字段 | 类型 | 说明 |
|------|------|------|
| `compile_sizes` | `list[int \| str] \| None` | 指定编译的 batch size 列表 |
| `compile_ranges_endpoints` | `list[int] \| None` | 编译范围端点 |
| `inductor_compile_config` | `dict` | Inductor 附加配置 |
| `inductor_passes` | `dict[str, str]` | 附加 Inductor Pass |

#### CUDA Graph 控制

| 字段 | 类型 | 说明 |
|------|------|------|
| `cudagraph_mode` | `CUDAGraphMode` | CUDA Graph 捕获模式 |
| `cudagraph_capture_sizes` | `list[int] \| None` | CUDA Graph 捕获大小列表 |
| `cudagraph_num_of_warmups` | `int` | 预热运行次数 |
| `cudagraph_copy_inputs` | `bool` | 是否为 CUDA Graph 复制输入张量 |

#### 关键方法

**`init_backend()`** —— 根据 `CompilationMode` 初始化编译后端：

```python
def init_backend(self, vllm_config, prefix="", is_encoder=False):
    if self.mode == CompilationMode.NONE:
        raise ValueError("No compilation mode is set.")

    # STOCK_TORCH_COMPILE 和 DYNAMO_TRACE_ONCE 使用 PyTorch 原生后端
    if self.mode in [CompilationMode.STOCK_TORCH_COMPILE,
                     CompilationMode.DYNAMO_TRACE_ONCE]:
        if self.backend in torch_backends:
            return self.backend        # 返回后端名称字符串
        return resolve_obj_by_qualname(self.backend)  # 返回自定义后端函数

    # VLLM_COMPILE 使用 VllmBackend
    assert self.mode == CompilationMode.VLLM_COMPILE
    return VllmBackend(vllm_config, prefix=prefix, is_encoder=is_encoder)
```

### 4.2 TorchCompileWithNoGuardsWrapper

`TorchCompileWithNoGuardsWrapper`（`vllm/compilation/wrapper.py`）是所有编译模式的入口包装器。其核心职责：

1. **调用 `torch.compile`**：以 `fullgraph=True`, `dynamic=False` 编译模型 forward 方法
2. **Guard 管理**：根据 `CompilationMode` 决定是否丢弃 Dynamo Guard
3. **字节码钩子**：可选地注册字节码钩子，避免后续调用再经过 Dynamo

```python
# Guard 策略分支
if mode != CompilationMode.STOCK_TORCH_COMPILE:
    # 非标准模式：丢弃所有 Guard，确保只编译一次
    if self.evaluate_guards:
        options["guard_filter_fn"] = lambda x: [
            entry.guard_type == "SHAPE_ENV" for entry in x
        ]
    else:
        options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
# STOCK_TORCH_COMPILE 模式：保留所有 Guard（标准 PyTorch 行为）

self._compiled_callable = torch.compile(
    compiled_ptr,
    fullgraph=True,   # 要求完整图，不允许 graph break
    dynamic=False,     # 不使用 Dynamo 的动态形状
    backend=backend,
    options=options,
)
```

**字节码钩子机制**：在 `DYNAMO_TRACE_ONCE` 和 `VLLM_COMPILE` 模式下，注册字节码钩子截获 Dynamo 转换后的字节码。后续调用直接替换 forward 方法的字节码为编译后版本，完全绕过 Dynamo 层，消除重编译开销。

### 4.3 @support_torch_compile 装饰器

`@support_torch_compile`（`vllm/compilation/decorators.py`）装饰器是模型类与编译系统对接的入口。70+ 个模型类使用了该装饰器。

**装饰器的核心工作**：

1. **将 `TorchCompileWithNoGuardsWrapper` 注入模型类的基类**
2. **劫持 `__init__` 方法**：在模型初始化时设置编译配置
3. **标记动态维度**：通过 `torch._dynamo.mark_dynamic` 或 `mark_unbacked` 标记输入张量的动态维度
4. **重写 `__call__` 方法**：首次调用时触发编译，后续调用走编译路径

```python
# 编译跳过条件
self.do_not_compile = (
    self.compilation_config.mode in [
        CompilationMode.NONE,
        CompilationMode.STOCK_TORCH_COMPILE  # 由 model runner 层处理
    ]
    or _should_ignore_torch_compile(self.__class__)
    or not enable_compile
)
```

**关键设计**：对于 `STOCK_TORCH_COMPILE` 模式，装饰器**不参与**编译流程，而是由上层的 `gpu_model_runner.py` 直接调用 `self.model.compile()` 完成标准 torch.compile 编译。

### 4.4 VllmBackend

`VllmBackend`（`vllm/compilation/backends.py`）是 `VLLM_COMPILE` 模式的核心编译后端，作为 `torch.compile` 的 backend 参数传入。

**VllmBackend 的主要流程**：

```
torch.compile(forward, backend=VllmBackend)
    │
    ▼ Dynamo 追踪 forward → 生成 FX GraphModule
    │
    ▼ VllmBackend.__call__(graph, example_inputs)
    │
    ├─ 1. 计算 Hash（环境、配置、代码、编译器）
    │     → 确定缓存目录
    │
    ├─ 2. 图切分 split_graph()
    │     → 按 splitting_ops 将图切分为多个子图
    │     → 生成 SplitItem 列表
    │
    ├─ 3. PiecewiseCompileInterpreter 遍历子图
    │     → 为每个子图创建 PiecewiseBackend
    │     → 每个 PiecewiseBackend 编译所有 range
    │
    ├─ 4. CUDA Graph 包装
    │     → 如需要，用 CUDAGraphWrapper 包装子图
    │
    └─ 5. 返回编译后的可调用对象
```

### 4.5 PiecewiseBackend

`PiecewiseBackend`（`vllm/compilation/piecewise_backend.py`）为每个子图管理多个编译范围（compile range）。

**核心概念**：
- **Compile Range**：一个 batch size 范围 `[start, end]`，模型为每个 range 生成一个独立编译的计算图
- **运行时分发**：根据当前输入的实际 batch size，选择对应 range 的编译产物执行

```python
def __call__(self, *args):
    runtime_shape = args[self.sym_shape_indices[0]]
    range_entry = self._find_range_for_shape(runtime_shape)
    return range_entry.runnable(*args)
```

**双模式运行**：
- **冷启动（Cold-start）**：接收 FX Graph，为所有 range 编译
- **热启动（Warm-start）**：接收预编译的 runnables（从缓存加载），直接包装

### 4.6 CompilerInterface 与适配器

`CompilerInterface`（`vllm/compilation/compiler_interface.py`）定义了编译器抽象接口，有三个实现：

| 适配器 | 说明 | 适用场景 |
|--------|------|---------|
| **InductorStandaloneAdaptor** | 使用 `torch._inductor.standalone_compile`，支持 AOT 序列化 | PyTorch ≥ 2.8，默认选择 |
| **InductorAdaptor** | 使用 `torch._inductor.compile_fx`，需 monkey-patch | PyTorch 2.5-2.7 |
| **EagerAdaptor** | 不编译，直接返回图 | backend="eager" |

---

## 5. 四种编译模式的详细工作流程

### 5.1 CompilationMode.NONE (模式 0)

#### 概述
完全不使用 `torch.compile`，模型以纯 PyTorch eager 模式运行。

#### 工作流程

```
用户请求 → 模型加载 → 模型 forward() → 纯 PyTorch 急切执行 → 输出
```

1. **配置阶段**：`VllmConfig` 将 `mode` 设为 `CompilationMode.NONE`
2. **模型加载**：`@support_torch_compile` 装饰器检测到 `NONE` 模式，设置 `self.do_not_compile = True`
3. **运行时**：`__call__` 直接调用 `self.forward(*args, **kwargs)`，无编译开销

#### 适用场景
- **调试**：需要检查模型中间状态
- **最快启动**：不需等待编译
- **兼容性问题**：某些模型无法编译
- **`enforce_eager=True`** 参数会自动设置此模式

#### 关键代码路径

```python
# decorators.py - __call__ 中的跳过逻辑
def __call__(self, *args, **kwargs):
    if self.do_not_compile or torch.compiler.is_compiling():
        return self.forward(*args, **kwargs)  # 直接急切执行
```

```python
# custom_op.py - 自定义算子处理
if compilation_config.mode == CompilationMode.NONE:
    # 使用 vLLM 的自定义 CUDA 内核（不经过编译）
```

#### 特性
- ✅ 零编译延迟
- ✅ 完全可调试
- ❌ 无编译优化
- ❌ 无 CUDA Graph（`cudagraph_mode` 被强制设为 `NONE`）

---

### 5.2 CompilationMode.STOCK_TORCH_COMPILE (模式 1)

#### 概述
使用标准的 `torch.compile` 编译管线，完全保留 Dynamo 的 guard 机制。这是最接近原生 PyTorch 编译行为的模式。

#### 工作流程

```
┌────────────────────────────────────────────────────────────┐
│               STOCK_TORCH_COMPILE 工作流程                    │
└─────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. gpu_model_runner._compile_model()                        │
│     backend = compilation_config.init_backend(vllm_config)   │
│     → 返回 PyTorch 原生后端名称（如 "inductor"）                │
│     self.model.compile(fullgraph=True, backend=backend)      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. @support_torch_compile 装饰器                             │
│     do_not_compile = True  (STOCK 模式不由装饰器编译)           │
│     → 装饰器透传，不干预                                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 标准 torch.compile 执行                                   │
│     • Dynamo 追踪 forward → FX Graph                         │
│     • 保留所有 Guard（形状、类型、值 Guard）                    │
│     • Inductor 后端编译 → Triton 内核                         │
│     • 如果 Guard 不匹配 → 触发重编译                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 运行时                                                    │
│     每次调用 forward():                                       │
│     • Dynamo 评估 Guard                                      │
│     • Guard 通过 → 执行编译后代码                              │
│     • Guard 不通过 → 重新追踪并编译                            │
└─────────────────────────────────────────────────────────────┘
```

#### 关键区别
- **由 `gpu_model_runner` 负责编译**，而非 `@support_torch_compile` 装饰器
- **保留所有 Dynamo Guard**：形状变化会触发重编译
- **无字节码钩子**：不使用 vLLM 的字节码优化

```python
# gpu_model_runner.py
if self.vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
    backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
    self.model.compile(fullgraph=True, backend=backend)
    return  # 不再进行其他编译设置
```

#### wrapper.py 中的特殊处理

```python
# STOCK_TORCH_COMPILE 模式不注册字节码钩子
if envs.VLLM_USE_BYTECODE_HOOK and mode != CompilationMode.STOCK_TORCH_COMPILE:
    torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

# __call__ 中的直接调用
if self.vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
    return self._compiled_callable(*args, **kwargs)  # 直接调用，不走字节码优化
```

#### 适用场景
- 需要 Dynamo 的完整 guard 保护（例如形状频繁变化的工作负载）
- 需要 PyTorch 标准编译行为的兼容场景
- 弹性分布式执行（Elastic EP）

#### 特性
- ✅ 完整的 Dynamo guard 保护
- ✅ 支持 PyTorch 原生所有后端
- ❌ 可能触发重编译（Guard 失败时）
- ❌ 无 vLLM 自定义优化 Pass
- ❌ 无分片编译
- ❌ 不支持 Piecewise CUDA Graph

---

### 5.3 CompilationMode.DYNAMO_TRACE_ONCE (模式 2)

#### 概述
通过 Dynamo 追踪模型一次，但丢弃所有 Guard，避免任何重编译。使用 PyTorch 原生后端，但去除了 guard 开销。

#### 工作流程

```
┌────────────────────────────────────────────────────────────┐
│               DYNAMO_TRACE_ONCE 工作流程                      │
└─────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. @support_torch_compile 装饰器激活                          │
│     do_not_compile = False (参与编译)                          │
│     TorchCompileWithNoGuardsWrapper.__init__() 执行           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. init_backend() → 返回 PyTorch 原生后端                     │
│     if self.backend in torch_backends:                       │
│         return self.backend  # 如 "inductor"                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. torch.compile 配置                                       │
│     • guard_filter_fn = skip_all_guards_unsafe               │
│     • fullgraph=True, dynamic=False                          │
│     • 注册字节码钩子                                           │
│                                                              │
│     torch.compile(forward, fullgraph=True, dynamic=False,    │
│                   backend="inductor", options={              │
│                       "guard_filter_fn": skip_all_guards     │
│                   })                                         │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 首次调用 → 编译                                           │
│     • Dynamo 追踪 forward 生成 FX Graph                       │
│     • 所有 Guard 被丢弃 → 不会触发重编译                       │
│     • 原生后端（Inductor）编译图 → Triton 内核                 │
│     • 字节码钩子保存编译后字节码                                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 后续调用                                                  │
│     • 使用保存的字节码直接执行                                  │
│     • 完全绕过 Dynamo（_dispatch_to_compiled_code）            │
│     • 无 Guard 评估开销                                       │
└─────────────────────────────────────────────────────────────┘
```

#### Guard 丢弃机制

这是 `DYNAMO_TRACE_ONCE` 区别于 `STOCK_TORCH_COMPILE` 的核心特性：

```python
# wrapper.py - Guard 丢弃
if mode != CompilationMode.STOCK_TORCH_COMPILE:
    # 对于 DYNAMO_TRACE_ONCE 和 VLLM_COMPILE：
    if self.evaluate_guards:
        # 只保留 SHAPE_ENV guard（用于 backed dynamic shapes）
        options["guard_filter_fn"] = lambda x: [
            entry.guard_type == "SHAPE_ENV" for entry in x
        ]
    else:
        # 丢弃所有 guard
        options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
```

#### 与 STOCK_TORCH_COMPILE 的关键区别
1. **由 `@support_torch_compile` 装饰器驱动编译**，而非 `gpu_model_runner`
2. **丢弃所有 Dynamo Guard**：模型只编译一次
3. **使用字节码钩子**：后续调用直接替换字节码，完全绕过 Dynamo
4. **仍使用 PyTorch 原生后端**：不使用 VllmBackend

#### 适用场景
- CPU 平台（`VLLM_COMPILE` 不支持时的降级选择）
- 需要单次编译但不需要 vLLM 分片编译等高级特性
- 中等性能需求

#### 特性
- ✅ 单次编译，无重编译
- ✅ 消除 Guard 评估开销
- ✅ 字节码优化加速后续调用
- ❌ 无 vLLM 自定义优化 Pass
- ❌ 无分片编译
- ❌ 不支持 Piecewise CUDA Graph

---

### 5.4 CompilationMode.VLLM_COMPILE (模式 3)

#### 概述
vLLM 的全定制编译模式，是**最高性能的模式**，也是 vLLM 默认使用的编译策略。该模式使用 `VllmBackend` 作为 `torch.compile` 后端，支持分片编译、自定义优化 Pass、编译缓存和 CUDA Graph 集成。

#### 完整工作流程

```
┌────────────────────────────────────────────────────────────────────┐
│                    VLLM_COMPILE 完整工作流程                         │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Phase 1: 初始化                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ @support_torch_compile 装饰器                                 │  │
│  │ • TorchCompileWithNoGuardsWrapper.__init__()                 │  │
│  │ • init_backend() → VllmBackend(vllm_config)                  │  │
│  │ • guard_filter_fn = skip_all_guards_unsafe                   │  │
│  │ • 注册字节码钩子                                               │  │
│  │                                                               │  │
│  │ torch.compile(forward, fullgraph=True, dynamic=False,        │  │
│  │               backend=VllmBackend_instance, options={...})   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Phase 2: Dynamo 追踪                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • 标记输入张量的动态维度                                        │  │
│  │ • Dynamo 追踪 forward → FX GraphModule                       │  │
│  │ • 丢弃所有 Guard（只追踪一次）                                  │  │
│  │ • 记录追踪到的源文件（用于缓存验证）                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Phase 3: VllmBackend.__call__(graph, example_inputs)              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Step 3a: 计算缓存 Hash                                       │  │
│  │ • env_hash: 环境变量（VLLM_PP_LAYER_PARTITION 等）             │  │
│  │ • config_hash: VllmConfig 完整配置                            │  │
│  │ • code_hash: 追踪到的源文件内容 hash                          │  │
│  │ • compiler_hash: Triton/系统/PyTorch 版本                     │  │
│  │ → cache_dir = VLLM_CACHE_ROOT/torch_compile_cache/{hash}/   │  │
│  │              rank_{rank}_{dp_rank}/{prefix}/                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Step 3b: 图切分 split_graph()                                │  │
│  │ • 根据 splitting_ops 将图切分为多个子图                       │  │
│  │ • splitting_ops 默认为 attention 算子列表：                   │  │
│  │   - vllm::unified_attention                                  │  │
│  │   - vllm::unified_mla_attention                              │  │
│  │   - vllm::mamba_mixer 等                                     │  │
│  │ • 生成 SplitItem 列表：                                      │  │
│  │   - submod_0: attention 前的计算                              │  │
│  │   - submod_1: attention 算子（splitting graph）              │  │
│  │   - submod_2: attention 后到下一个 attention 前               │  │
│  │   - ...                                                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Step 3c: PostGradPassManager 配置                            │  │
│  │ • 根据 PassConfig 配置注册优化 Pass：                         │  │
│  │   - NoOpEliminationPass: 消除空操作                           │  │
│  │   - SequenceParallelismPass: 序列并行                         │  │
│  │   - AllReduceFusionPass: AllReduce+RMSNorm 融合              │  │
│  │   - RMSNormQuantFusionPass: RMSNorm+量化融合                 │  │
│  │   - ActQuantFusionPass: 激活+量化融合                         │  │
│  │   - RoPE+KVCache 融合、注意力+量化融合 等                     │  │
│  │ • Pass 注入到 Inductor config 中                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Step 3d: 分片编译                                            │  │
│  │ • PiecewiseCompileInterpreter 遍历切分后的子图               │  │
│  │ • 为每个非 splitting 子图创建 PiecewiseBackend               │  │
│  │ • PiecewiseBackend.compile_all_ranges():                     │  │
│  │   for range in compile_ranges:                               │  │
│  │       compiled = CompilerManager.compile(                    │  │
│  │           graph, args, inductor_config, range                │  │
│  │       )                                                      │  │
│  │       # → InductorStandaloneAdaptor.compile()                │  │
│  │       # → standalone_compile(graph, inputs)                  │  │
│  │       # → 生成 Triton 内核                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Step 3e: 编译去重                                            │  │
│  │ • CompilerManager 拦截 autograd_cache_key                    │  │
│  │ • 如果发现同构图（isometric graph），复用已编译产物            │  │
│  │ • 避免重复编译结构相同的 transformer 层                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Step 3f: CUDA Graph 包装                                     │  │
│  │ • 如果 cudagraph_mode 包含 PIECEWISE：                       │  │
│  │   每个子图用 CUDAGraphWrapper 包装                            │  │
│  │ • 如果 cudagraph_mode 包含 FULL：                             │  │
│  │   整个模型用 CUDAGraphWrapper 包装                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Phase 4: 运行时执行                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ model.__call__(*args)                                        │  │
│  │ → TorchCompileWithNoGuardsWrapper.__call__()                 │  │
│  │   → _dispatch_to_compiled_code()                             │  │
│  │     → split_gm(*args)  (编译后的拼接图)                       │  │
│  │       → submod_0(args)  [PiecewiseBackend → Triton 内核]     │  │
│  │         → CUDAGraphWrapper.capture_or_replay()               │  │
│  │       → submod_1(args)  [Attention 算子，eager 执行]          │  │
│  │       → submod_2(args)  [PiecewiseBackend → Triton 内核]     │  │
│  │         → CUDAGraphWrapper.capture_or_replay()               │  │
│  │       → ...                                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

#### 图切分详解

`split_graph()` 是 `VLLM_COMPILE` 模式的核心创新之一：

```python
def split_graph(graph, splitting_ops):
    # 1. 分解 size() 调用为逐维 sym_size.int
    _decompose_size_nodes(graph)

    # 2. 为每个节点分配子图 ID
    for node in graph.graph.nodes:
        if should_split(node, splitting_ops):
            subgraph_id += 1       # 新子图
            node_to_subgraph_id[node] = subgraph_id
            subgraph_id += 1       # splitting node 后开始新子图
        else:
            node_to_subgraph_id[node] = subgraph_id

    # 3. 合并只含 torch.empty 的空子图
    _merge_empty_only_subgraphs(...)

    # 4. 使用 PyTorch 的 split_module 切分
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True  # 保持语义正确性
    )
    return split_gm, piecewise_graphs
```

**切分策略**：默认按 attention 算子切分。这意味着 attention 算子在 CUDA Graph 之外 eager 执行（因为 attention 涉及动态内存访问模式），而线性层、归一化等计算密集型算子在 CUDA Graph 内执行。

#### 编译去重机制

Transformer 模型中的多层结构通常具有相同的计算图结构。`CompilerManager` 通过拦截 `autograd_cache_key` 实现去重：

```python
def compile(self, graph, example_inputs, ...):
    # 拦截 cache key 计算
    def autograd_cache_key(*args, **kwargs):
        result = orig(*args, **kwargs)
        cache_key = result[0]
        if cache_key in self.loaded_artifacts:
            raise StopCompiling()  # 发现同构图，跳过编译
        return result

    try:
        compiled_graph = self.compiler.compile(graph, ...)
    except StopCompiling:
        return self.loaded_artifacts[cache_key]  # 复用已有编译产物
```

#### 自定义 Inductor Pass

`VLLM_COMPILE` 模式下，`PostGradPassManager` 在 Inductor 的 post-grad 阶段运行自定义优化：

| Pass | 功能 |
|------|------|
| `NoOpEliminationPass` | 消除无操作节点 |
| `SequenceParallelismPass` | 序列并行优化 |
| `AllReduceFusionPass` | AllReduce + RMSNorm 融合 |
| `RMSNormQuantFusionPass` | RMSNorm + 量化融合 |
| `ActQuantFusionPass` | 激活函数 + 量化融合 |
| `AttnQuantFusionPass` | 注意力 + 量化融合 |
| `RoPEKVCacheFusionPass` | RoPE + KV Cache 融合（ROCm） |
| `PostCleanupPass` | 后清理 |
| `FixFunctionalizationPass` | 修复自动功能化问题 |

Pass 的执行顺序：
1. 用户/配置注册的 Pass
2. 默认 Pass（NoOp 消除、融合 Pass）
3. `post_grad_custom_post_pass`（如存在）
4. `FixFunctionalizationPass`（始终最后执行）

#### 特性
- ✅ 分片编译：attention 算子外的部分独立编译
- ✅ 自定义 Inductor Pass：丰富的融合优化
- ✅ 编译去重：同构层共享编译产物
- ✅ 编译缓存：支持持久化缓存
- ✅ 分片 CUDA Graph：灵活的 CUDA Graph 捕获
- ✅ 多 range 编译：不同 batch size 使用不同优化
- ✅ 丢弃 Guard：单次编译，无重编译
- ❌ 启动延迟较长（首次编译）
- ❌ 仅支持 `"inductor"` 和 `"eager"` 后端

---

## 6. 模式间异同点对比分析

### 6.1 编译触发方式

| 维度 | NONE | STOCK_TORCH_COMPILE | DYNAMO_TRACE_ONCE | VLLM_COMPILE |
|------|------|--------------------|--------------------|--------------|
| **编译发起者** | 无 | `gpu_model_runner` | `@support_torch_compile` 装饰器 | `@support_torch_compile` 装饰器 |
| **编译入口** | 无 | `self.model.compile()` | `torch.compile(forward)` | `torch.compile(forward)` |
| **后端** | 无 | PyTorch 原生后端 | PyTorch 原生后端 | `VllmBackend` 实例 |
| **init_backend 返回值** | 异常 | 后端名称/函数 | 后端名称/函数 | `VllmBackend` 对象 |

### 6.2 Guard 与重编译策略

| 维度 | NONE | STOCK_TORCH_COMPILE | DYNAMO_TRACE_ONCE | VLLM_COMPILE |
|------|------|--------------------|--------------------|--------------|
| **Guard 策略** | 无 | 保留所有 Guard | 丢弃所有 Guard | 丢弃所有 Guard |
| **重编译** | 无 | 可能（Guard 失败时） | 不会 | 不会 |
| **字节码钩子** | 无 | 不注册 | 注册 | 注册 |
| **Guard 评估开销** | 无 | 每次调用 | 无 | 无 |

### 6.3 编译优化特性

| 维度 | NONE | STOCK_TORCH_COMPILE | DYNAMO_TRACE_ONCE | VLLM_COMPILE |
|------|------|--------------------|--------------------|--------------|
| **图切分** | 无 | 无 | 无 | ✅ 按 attention 算子切分 |
| **分片编译** | 无 | 无 | 无 | ✅ 多 range 编译 |
| **自定义 Pass** | 无 | 无 | 无 | ✅ PostGradPassManager |
| **编译去重** | 无 | 无 | 无 | ✅ 同构图复用 |
| **编译缓存** | 无 | PyTorch 原生缓存 | PyTorch 原生缓存 | ✅ vLLM 缓存 + AOT |
| **custom_ops** | 启用全部 | 取决于后端 | 取决于后端 | 默认禁用（Inductor 替代） |

### 6.4 CUDA Graph 支持

| 维度 | NONE | STOCK_TORCH_COMPILE | DYNAMO_TRACE_ONCE | VLLM_COMPILE |
|------|------|--------------------|--------------------|--------------|
| **CUDA Graph** | 不支持 | 由 PyTorch 管理 | 不支持 Piecewise | 完整支持 |
| **Piecewise CG** | ❌ | ❌ | ❌ | ✅ |
| **Full CG** | ❌ | ❌ | ❌ | ✅ |
| **CG 与编译集成** | 无 | 分离 | 分离 | 紧密集成 |

### 6.5 运行时行为

| 维度 | NONE | STOCK_TORCH_COMPILE | DYNAMO_TRACE_ONCE | VLLM_COMPILE |
|------|------|--------------------|--------------------|--------------|
| **首次调用** | 直接执行 | Dynamo 追踪+编译 | Dynamo 追踪+编译 | Dynamo 追踪 → VllmBackend 编译 |
| **后续调用** | 直接执行 | Guard 评估+执行 | 字节码替换+执行 | 字节码替换 → PiecewiseBackend 分发 |
| **输出 padding** | 需要 | 需要 | 需要 | 不需要 |
| **启动延迟** | 最低 | 中等 | 中等 | 最高（但缓存可消除） |

### 6.6 `@support_torch_compile` 装饰器行为

| 行为 | NONE | STOCK_TORCH_COMPILE | DYNAMO_TRACE_ONCE | VLLM_COMPILE |
|------|------|--------------------|--------------------|--------------|
| **`do_not_compile`** | `True` | `True` | `False` | `False` |
| **`__init__` 初始化** | 跳过 Wrapper | 跳过 Wrapper | 初始化 Wrapper | 初始化 Wrapper |
| **`__call__` 行为** | 直接 `forward()` | 直接 `forward()` | `Wrapper.__call__()` | `Wrapper.__call__()` |
| **动态维度标记** | 不标记 | 不标记 | 标记 | 标记 |

---

## 7. 优化级别与编译模式的关系

### 7.1 OptimizationLevel 枚举

```python
class OptimizationLevel(IntEnum):
    O0 = 0  # 无优化：不编译，不捕获 CUDA Graph
    O1 = 1  # 快速优化：Dynamo+Inductor 编译 + Piecewise CUDA Graph
    O2 = 2  # 完整优化：O1 + Full 和 Piecewise CUDA Graph（默认）
    O3 = 3  # 当前与 O2 相同
```

### 7.2 模式自动选择

```python
# VllmConfig.__post_init__()
if self.compilation_config.mode is None:
    if self.optimization_level > OptimizationLevel.O0:
        self.compilation_config.mode = CompilationMode.VLLM_COMPILE
    else:
        self.compilation_config.mode = CompilationMode.NONE
```

### 7.3 优化级别配置映射

| 参数 | O0 | O1 | O2（默认） |
|------|----|----|-----------|
| **编译模式** | NONE | VLLM_COMPILE | VLLM_COMPILE |
| **CUDAGraph 模式** | NONE | PIECEWISE | FULL_AND_PIECEWISE |
| **RMSNorm+量化融合** | ❌ | ✅（条件性） | ✅（条件性） |
| **激活+量化融合** | ❌ | ✅（条件性） | ✅（条件性） |
| **AllReduce+RMS 融合** | ❌ | ❌ | ✅（条件性） |
| **序列并行** | ❌ | ❌ | ✅（仅密集模型） |
| **异步 TP** | ❌ | ❌ | ✅（仅密集模型） |
| **Flashinfer 自动调优** | ❌ | ✅ | ✅ |
| **Inductor 图分区** | ❌ | ❌ | ❌ |

### 7.4 条件性优化函数

某些优化 Pass 的启用是条件性的，由回调函数在运行时决定：

```python
def enable_norm_fusion(vllm_config):
    """仅在 custom_ops 中启用了 rms_norm 时激活 RMSNorm 融合"""
    return vllm_config.compilation_config.is_custom_op_enabled("rms_norm")

def enable_allreduce_rms_fusion(vllm_config):
    """需要 TP>1、CUDA 设备、Flashinfer 可用"""
    return (tp_size > 1
            and is_cuda
            and flashinfer_available
            and not is_amd)
```

---

## 8. CUDA Graph 与编译模式的交互

### 8.1 CUDAGraphMode 枚举

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0                         # 不捕获 CUDA Graph
    PIECEWISE = 1                    # 分片 CUDA Graph（每个子图独立捕获）
    FULL = 2                         # 全图 CUDA Graph
    FULL_DECODE_ONLY = (FULL, NONE)  # 解码阶段 FULL，预填充阶段 NONE
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)  # 解码阶段 FULL，预填充阶段 PIECEWISE
```

### 8.2 编译模式与 CUDA Graph 的兼容性

| 编译模式 | 支持的 CUDAGraph 模式 | 说明 |
|---------|---------------------|------|
| NONE | NONE | 不支持任何 CUDA Graph |
| STOCK_TORCH_COMPILE | NONE | 不支持 vLLM 管理的 CUDA Graph |
| DYNAMO_TRACE_ONCE | NONE | 不支持 Piecewise（需要 VllmBackend 的图切分） |
| VLLM_COMPILE | **全部支持** | 唯一支持 Piecewise CUDA Graph 的模式 |

```python
# 兼容性检查
if (cudagraph_mode.requires_piecewise_compilation()
    and mode != CompilationMode.VLLM_COMPILE):
    logger.info("Cudagraph mode %s is not compatible with compilation mode %s.",
                cudagraph_mode, mode)
    cudagraph_mode = CUDAGraphMode.NONE  # 降级
```

### 8.3 Piecewise CUDA Graph 工作原理

在 `VLLM_COMPILE` 模式下，图切分将模型分为多个子图：

```
模型 forward → split_graph() → [submod_0] [submod_1(attn)] [submod_2] [submod_3(attn)] [submod_4]
                                   │              │              │              │              │
                                   ▼              ▼              ▼              ▼              ▼
                              CUDAGraph        Eager         CUDAGraph       Eager        CUDAGraph
                              捕获+重放       直接执行       捕获+重放      直接执行      捕获+重放
```

**Piecewise 的优势**：
1. **内存效率**：多个小图比一个大图使用更少的 GPU 内存峰值
2. **灵活性**：attention 算子可以 eager 执行（支持动态内存模式）
3. **GC 优化**：仅第一个分区启用 GC，后续分区禁用

### 8.4 Full CUDA Graph 工作原理

Full 模式在整个模型外层包装一个 CUDAGraphWrapper：

```python
# gpu_model_runner.py
if cudagraph_mode.has_full_cudagraphs():
    self.model = CUDAGraphWrapper(
        self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
    )
```

运行时根据 batch size 捕获或重放完整的模型前向传播。

---

## 9. 编译缓存系统

### 9.1 缓存层次

vLLM 的编译缓存系统有多个层次：

| 层次 | 文件 | 功能 |
|------|------|------|
| **vLLM 编译缓存** | `CompilerManager` | 管理子图级别的编译产物 |
| **Inductor FX 缓存** | PyTorch 内置 | Inductor 的图级别缓存 |
| **AOT 编译产物** | `caching.py` | 完整模型的 AOT 序列化 |
| **StandaloneCompiledArtifacts** | `caching.py` | 基于内容去重的产物存储 |

### 9.2 缓存键生成

```python
# 四要素 Hash
factors = [
    env_hash,       # 环境变量（VLLM_PP_LAYER_PARTITION 等）
    config_hash,    # VllmConfig 完整配置 hash
    code_hash,      # 追踪到的源文件内容 hash
    compiler_hash,  # Triton/系统/PyTorch 版本 hash
]
hash_key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
```

缓存目录结构：
```
$VLLM_CACHE_ROOT/
└── torch_compile_cache/
    ├── {hash_key}/                    # vLLM 编译缓存
    │   └── rank_{rank}_{dp_rank}/
    │       └── {prefix}/
    │           ├── vllm_compile_cache.py   # 缓存索引
    │           └── {compiled artifacts}     # 编译产物
    └── torch_aot_compile/             # AOT 编译缓存
        └── {aot_hash}/
            └── rank_{rank}_{dp_rank}/
                └── model                   # 完整 AOT 产物
```

### 9.3 缓存加载流程

1. **计算 Hash**：根据环境、配置、代码、编译器版本计算缓存键
2. **查找缓存**：检查缓存目录是否存在
3. **验证源码**：验证追踪到的源文件是否发生变化
4. **加载产物**：反序列化编译产物并恢复 PiecewiseBackend
5. **热启动**：直接使用预编译的 runnables，跳过编译

### 9.4 内容去重

`StandaloneCompiledArtifacts` 实现了两级去重：

```python
class StandaloneCompiledArtifacts:
    submodule_bytes: dict[str, str]        # "{submod}_{shape}" → SHA256 hash
    submodule_bytes_store: dict[str, bytes] # SHA256 hash → 编译字节码

    def insert(self, submod_name, shape, data_bytes):
        content_hash = hashlib.sha256(data_bytes).hexdigest()
        self.submodule_bytes[f"{submod_name}_{shape}"] = content_hash
        if content_hash not in self.submodule_bytes_store:
            self.submodule_bytes_store[content_hash] = data_bytes
        # 否则复用已有字节码（去重）
```

---

## 10. 端到端编译流程

### 10.1 从 VllmConfig 初始化到模型就绪

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: VllmConfig 构造                                          │
│  VllmConfig(optimization_level=O2,                               │
│             compilation_config=CompilationConfig(...))            │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2: VllmConfig.__post_init__()                               │
│  2a. enforce_eager 检查 → 可能设为 NONE                            │
│  2b. 平台默认值 → current_platform.apply_config_platform_defaults │
│  2c. 编译模式自动选择 → NONE 或 VLLM_COMPILE                      │
│  2d. custom_ops 默认值 → "all" 或 "none"                          │
│  2e. 优化级别配置应用 → 融合 Pass、CUDAGraph 模式                  │
│  2f. CUDAGraph 兼容性验证                                         │
│  2g. 序列并行配置                                                  │
│  2h. CUDA Graph 大小设置                                          │
│  2i. 编译范围设置                                                  │
│  2j. splitting_ops 设置                                           │
│  2k. 最终验证                                                     │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3: 模型加载 (gpu_model_runner._compile_model())             │
│                                                                   │
│  if STOCK_TORCH_COMPILE:                                         │
│      self.model.compile(fullgraph=True, backend=backend)         │
│  else:                                                            │
│      模型带有 @support_torch_compile 装饰器                        │
│      → 首次 model(*inputs) 触发编译                                │
│      如需要，包装 CUDA Graph Wrapper                               │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 4: 首次推理触发编译（DYNAMO_TRACE_ONCE/VLLM_COMPILE）       │
│                                                                   │
│  model.__call__(*args)                                            │
│  → _mark_dynamic_inputs()                                        │
│  → TorchCompileWithNoGuardsWrapper.__call__()                    │
│    → torch.compile 触发 Dynamo 追踪                               │
│    → [VLLM_COMPILE] VllmBackend.__call__(graph)                  │
│      → split_graph()                                              │
│      → PiecewiseBackend.compile_all_ranges()                     │
│      → CompilerManager.compile() → Inductor                     │
│    → 字节码钩子保存编译后字节码                                    │
│  → 编译完成，self.compiled = True                                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 5: 后续推理（已编译）                                        │
│                                                                   │
│  model.__call__(*args)                                            │
│  → _dispatch_to_compiled_code()                                  │
│    → 字节码替换为编译版本                                          │
│    → [VLLM_COMPILE] split_gm(args)                               │
│      → PiecewiseBackend 按 batch size 分发到正确的编译 range      │
│      → CUDAGraphWrapper 捕获或重放                                │
│  → 返回输出                                                       │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 自定义算子（Custom Ops）与编译模式的交互

在 `VLLM_COMPILE` 模式下，默认配置为 `custom_ops = ["none"]`（禁用所有自定义算子），因为 Inductor 后端会生成等效的 Triton 内核。在非 Inductor 模式下，默认为 `custom_ops = ["all"]`（启用所有自定义 CUDA 内核）。

```python
# IntEnum 比较用于决定行为
pad_output = vllm_config.mode < CompilationMode.VLLM_COMPILE
# NONE(0), STOCK_TORCH_COMPILE(1), DYNAMO_TRACE_ONCE(2) → True (需要 padding)
# VLLM_COMPILE(3) → False (不需要 padding，Inductor 处理)
```

---

## 11. 平台特定行为

### 11.1 CPU 平台

```python
# vllm/platforms/cpu.py
if vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
    # CPU 不支持 VLLM_COMPILE → 降级为 DYNAMO_TRACE_ONCE
    compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
else:
    compilation_config.mode = CompilationMode.NONE
```

### 11.2 默认后端选择

```python
if self.backend == "":
    self.backend = current_platform.get_compile_backend()
    # CUDA 平台 → "inductor"
    # CPU 平台 → 平台特定后端
    # XLA 平台 → "openxla"
```

---

## 12. 总结

### 12.1 模式选择指南

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| **生产部署** | `VLLM_COMPILE` (O2) | 最高性能，全部优化 |
| **快速原型** | `NONE` (O0) | 零启动延迟 |
| **调试编译问题** | `STOCK_TORCH_COMPILE` | 标准 PyTorch 行为 |
| **CPU 部署** | `DYNAMO_TRACE_ONCE` | 自动降级的 CPU 选择 |
| **弹性分布式** | `STOCK_TORCH_COMPILE` | 需要 Guard 保护 |

### 12.2 设计亮点

1. **渐进式复杂度**：从 NONE → STOCK → DYNAMO_TRACE → VLLM_COMPILE，每一级增加更多优化，用户可按需选择
2. **Guard 消除**：DYNAMO_TRACE_ONCE 和 VLLM_COMPILE 通过丢弃 Guard 确保单次编译
3. **分片编译**：VLLM_COMPILE 独有的图切分+分片编译策略，平衡了 CUDA Graph 的性能优势和 attention 算子的灵活性需求
4. **多层缓存**：从 Inductor FX 缓存到 vLLM 编译缓存到 AOT 产物，多层次缓存机制最大化避免重复编译
5. **编译去重**：利用 `autograd_cache_key` 拦截实现同构图（如 Transformer 重复层）的编译复用
6. **IntEnum 设计**：允许通过数值比较判断编译级别，代码简洁

### 12.3 架构设计模式

1. **策略模式**：通过 `CompilerInterface` 抽象接口和多个适配器（Inductor、Eager）实现后端切换
2. **装饰器模式**：`@support_torch_compile` 无侵入地为模型类添加编译支持
3. **工厂模式**：`init_backend()` 根据模式创建不同后端
4. **解释器模式**：`PiecewiseCompileInterpreter` 遍历切分后的图
5. **缓存模式**：`CompilerManager` 和 `StandaloneCompiledArtifacts` 实现多级缓存
