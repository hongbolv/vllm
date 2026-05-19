# vLLM Examples 梳理报告

> 本文档全面梳理了 vLLM 项目 `examples/` 目录下的所有示例文件，按目录分类说明每个示例的**测试对象**和**目的**。

---

## 📁 一、`examples/basic/` — 基础入门示例

### 1. `basic/offline_inference/` — 离线推理基础

| 文件 | 测试对象/目的 |
|------|-------------|
| `basic.py` | 使用 `LLM` 类 + `SamplingParams` 进行最基本的文本生成 |
| `chat.py` | 基于对话格式（chat）的离线推理，支持 CLI 参数 |
| `classify.py` | 使用 pooling runner 进行文本分类 |
| `embed.py` | 使用 pooling runner 生成文本 embedding |
| `generate.py` | 高级文本生成，展示可配置的采样参数 |
| `score.py` | 使用 cross-encoder 进行文档评分/重排序 |

### 2. `basic/online_serving/` — 在线服务基础

| 文件 | 测试对象/目的 |
|------|-------------|
| `openai_chat_completion_client.py` | OpenAI 兼容的 Chat Completion API 客户端 |
| `openai_completion_client.py` | OpenAI 兼容的 Text Completion API 客户端 |

---

## 📁 二、`examples/features/` — 特性功能示例

### 1. 自动前缀缓存 (Automatic Prefix Caching)

| 文件 | 测试对象/目的 |
|------|-------------|
| `automatic_prefix_caching_offline.py` | APC 功能：共享前缀的 KV cache 复用 |
| `prefix_caching_offline.py` | 手动前缀缓存配置 |

### 2. 可复现性 (Reproducibility)

| 文件 | 测试对象/目的 |
|------|-------------|
| `reproducibility_offline.py` | 通过环境变量实现确定性生成，保证结果可复现 |

### 3. 上下文扩展 (Context Extension)

| 文件 | 测试对象/目的 |
|------|-------------|
| `context_extension_offline.py` | 使用 YARN 方法扩展上下文长度 |

### 4. 数据并行 (Data Parallelism)

| 文件 | 测试对象/目的 |
|------|-------------|
| `data_parallel_offline.py` | 多 GPU/多节点的数据并行推理 |
| `multi_instance_data_parallel.py` | 多实例异步数据并行 |

### 5. KV Cache 事件监控

| 文件 | 测试对象/目的 |
|------|-------------|
| `kv_events_subscriber.py` | 使用 msgspec/zmq 监控 KV cache 事件 |

### 6. Logits 处理器 (Logits Processors)

| 文件 | 测试对象/目的 |
|------|-------------|
| `custom.py` | 自定义 batch 级别的 logits processor |
| `custom_req.py` | 请求级别的 logits processor 包装 |
| `custom_req_init.py` | 带引擎配置初始化的 logits processor |

### 7. LoRA 适配器

| 文件 | 测试对象/目的 |
|------|-------------|
| `lora_with_quantization_offline.py` | LoRA + 量化推理 |
| `multilora_offline.py` | 多 LoRA 适配器同时使用 |

### 8. 暂停/恢复 (Pause/Resume)

| 文件 | 测试对象/目的 |
|------|-------------|
| `pause_resume_offline.py` | 异步生成的暂停和恢复，含 token 追踪 |
| `data_parallel_pause_resume.py` | 数据并行下的暂停/恢复 |

### 9. 性能分析 (Profiling)

| 文件 | 测试对象/目的 |
|------|-------------|
| `run_one_batch_offline.py` | 单 batch 性能分析 |
| `simple_profiling_offline.py` | PyTorch profiler 集成 |

### 10. Prompt Embeddings

| 文件 | 测试对象/目的 |
|------|-------------|
| `prompt_embed_offline.py` | 使用预计算的 prompt embeddings 作为输入 |
| `prompt_embed_inference_with_openai_client.py` | 通过 OpenAI API 使用 base64 编码的 embeddings |

### 11. KV Cache 重置

| 文件 | 测试对象/目的 |
|------|-------------|
| `reset_kv_offline.py` | KV cache 重置下的抢占行为 |

### 12. 分片状态 (Sharded State)

| 文件 | 测试对象/目的 |
|------|-------------|
| `load_sharded_state_offline.py` | 加载分片模型检查点 |
| `save_sharded_state_offline.py` | 保存分片模型状态 |

### 13. 投机解码 (Speculative Decoding)

| 文件 | 测试对象/目的 |
|------|-------------|
| `extract_hidden_states_offline.py` | 提取隐藏状态用于投机解码 |
| `mlpspeculator_offline.py` | MLPSpeculator 投机解码（已弃用） |
| `spec_decode_offline.py` | 使用 draft model 的投机解码 |

### 14. 结构化输出 (Structured Outputs)

| 文件 | 测试对象/目的 |
|------|-------------|
| `structured_outputs_offline.py` | JSON schema、正则、语法、选择约束等结构化输出 |
| `structured_outputs_client.py` | 通过 OpenAI API 的流式结构化输出 |

### 15. TorchRun 启动

| 文件 | 测试对象/目的 |
|------|-------------|
| `torchrun_example_offline.py` | 使用 torchrun 的张量并行推理 |
| `torchrun_dp_example_offline.py` | 使用 torchrun 的数据并行推理 |

---

## 📁 三、`examples/generate/` — 生成类示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `batched_chat_completions_online.py` | 批量 chat completion 端点 |
| `qwen_1m_offline.py` | Qwen2.5 超长上下文（1M tokens）处理 |
| `token_generation_client.py` | 自定义端点的 token 级生成 |

### 多模态 (Multimodal)

| 文件 | 测试对象/目的 |
|------|-------------|
| `audio_language_offline.py` | 音频语言模型（AudioFlamingo 等） |
| `encoder_decoder_multimodal_offline.py` | 编码器-解码器多模态模型（Whisper 等） |
| `mistral-small_offline.py` | Mistral Small 视觉语言模型 |
| `openai_chat_completion_client_for_multimodal.py` | 多模态 OpenAI 兼容客户端 |
| `vision_language_offline.py` | 视觉语言模型（LLaVA、Phi-3.5、Aria 等） |
| `vision_language_multi_image_offline.py` | 多图像视觉语言推理 |
| `qwen2_5_omni/only_thinker.py` | Qwen2.5-Omni 音频/视频/图像多模态 |
| `qwen3_omni/only_thinker.py` | Qwen3-Omni 多模态推理 |

---

## 📁 四、`examples/observability/` — 可观测性示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `metrics/offline.py` | Counter、Gauge、Histogram、Vector 等指标采集 |
| `opentelemetry/dummy_client.py` | OpenTelemetry 追踪导出（OTLP/gRPC + 控制台） |

---

## 📁 五、`examples/offline_inference/` — 离线推理高级示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `async_llm_streaming.py` | AsyncLLM 流式生成（DELTA 模式） |
| `batch_llm_inference.py` | 使用 Ray Data 进行分布式批量推理 |
| `disaggregated_prefill.py` | 分离式预填充/解码与 KV 传输 |
| `llm_engine_example.py` | 底层 LLMEngine API 使用 |
| `prefix_caching_flexkv.py` | FlexKV 分布式 KV cache 集成 |

### 分离式预填充 v1 (`disaggregated-prefill-v1/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `prefill_example.py` | 预填充节点的 KV 传输 |
| `decode_example.py` | 解码节点接收 KV cache |

### KV 加载失败恢复 (`kv_load_failure_recovery/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `prefill_example.py` | 带故障处理的预填充 |
| `decode_example.py` | 异步加载恢复的解码 |
| `load_recovery_example_connector.py` | 自定义恢复连接器 |

---

## 📁 六、`examples/online_serving/` — 在线服务示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `api_client.py` | vLLM API 服务器 HTTP 客户端 |
| `gradio_webserver.py` | Gradio UI 文本补全界面 |
| `gradio_openai_chatbot_webserver.py` | Gradio UI 聊天界面 |
| `ray_serve_deepseek.py` | Ray Serve 部署 DeepSeek 模型 |
| `retrieval_augmented_generation_with_langchain.py` | LangChain + Milvus 的 RAG |
| `retrieval_augmented_generation_with_llamaindex.py` | LlamaIndex 的 RAG |
| `streamlit_openai_chatbot_webserver.py` | Streamlit 聊天 UI |

### 分离式编码器 (`disaggregated_encoder/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `disagg_epd_proxy.py` | 编码器-预填充-解码器代理 |

### 分离式服务 (`disaggregated_serving/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `disagg_proxy_demo.py` | XpYd 预填充/解码代理 |
| `disagg_proxy_multiturn.py` | 多轮对话 KV 复用 |
| `example_mm_serve.py` | 多模态渲染/生成 |
| `mooncake_connector_proxy.py` | Mooncake 连接器代理 |
| `moriio_toy_proxy_server.py` | Moriio P2P 代理 |

### P2P NCCL 分离式服务

| 文件 | 测试对象/目的 |
|------|-------------|
| `disagg_proxy_p2p_nccl_xpyd.py` | P2P NCCL 注册代理 |

### 弹性专家并行 (`elastic_ep/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `scale.py` | 动态 EP 缩放控制 |

---

## 📁 七、`examples/others/` — 其他工具示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `tensorize_vllm_model.py` | 使用 Tensorizer 序列化/反序列化模型 |

### LMCache 集成

| 文件 | 测试对象/目的 |
|------|-------------|
| `cpu_offload_lmcache.py` | LMCache CPU 卸载 |
| `disagg_prefill_lmcache_v0.py` | LMCache 分离式预填充 |
| `disagg_prefill_lmcache_v1/disagg_proxy_server.py` | LMCache v1 代理 |
| `kv_cache_sharing_lmcache_v1.py` | 远程 KV cache 共享 |

---

## 📁 八、`examples/pooling/` — 池化模型示例

### 分类 (`classify/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `classification_online.py` | 在线文本分类 API |
| `vision_classification_online.py` | 多模态视觉分类 |

### 嵌入 (`embed/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `embed_jina_embeddings_v3_offline.py` | Jina 多语言 embeddings |
| `embed_matryoshka_fy_offline.py` | Matryoshka 维度缩减 |
| `embedding_requests_base64_online.py` | Base64 编码 embeddings |
| `embedding_requests_bytes_online.py` | 字节编码 embeddings |
| `openai_embedding_client.py` | OpenAI 兼容 embeddings API |
| `openai_embedding_long_text/client.py` | 长文本分块 embedding |
| `openai_embedding_matryoshka_fy_client.py` | Matryoshka 维度控制 |
| `vision_embedding_offline.py` | 视觉语言 embedding（CLIP） |
| `vision_embedding_online.py` | 多模态 embeddings API |

### 插件 (`plugin/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `prithvi_geospatial_mae_offline.py` | 地理空间影像编码 |
| `prithvi_geospatial_mae_io_processor.py` | GeoTiff 预处理 |
| `prithvi_geospatial_mae_online.py` | 在线地理空间推理 |

### 奖励模型 (`reward/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `sequence_reward_offline.py` | 序列级别奖励模型 |
| `sequence_reward_online.py` | 在线序列奖励 |
| `token_reward_offline.py` | Token 级别奖励模型 |
| `token_reward_online.py` | 在线 token 奖励 |

### 评分/重排序 (`score/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `cohere_rerank_client.py` | Cohere 重排序兼容 |
| `colbert_rerank_online.py` | ColBERT 延迟交互重排序 |
| `colmodernvbert_rerank_online.py` | ModernBERT ColBERT 重排序 |
| `colqwen3_5_rerank_online.py` | Qwen3.5 ColBERT 重排序 |
| `colqwen3_rerank_online.py` | Qwen3 ColBERT 重排序 |
| `convert_model_to_seq_cls.py` | LLM 转序列分类模型 |
| `qwen3_reranker_offline.py` | Qwen3 重排序器离线 |
| `qwen3_reranker_online.py` | Qwen3 重排序器 API |
| `rerank_api_online.py` | 通用重排序 API |
| `score_api_online.py` | 通用评分 API |
| `using_template_offline.py` | 模板评分离线 |
| `using_template_online.py` | 模板评分 API |
| `vision_rerank_api_online.py` | 多模态重排序 |
| `vision_reranker_offline.py` | 视觉重排序器 |
| `vision_score_api_online.py` | 视觉评分 API |

### Token 分类 (`token_classify/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `forced_alignment_offline.py` | 强制 token 对齐 |
| `ner_offline.py` | 命名实体识别（离线） |
| `ner_online.py` | 命名实体识别（在线） |

### Token Embeddings (`token_embed/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `colqwen3_token_embed_online.py` | ColQwen3 多向量 embeddings |
| `jina_embeddings_v4_offline.py` | Jina v4 token embeddings |
| `jina_reranker_v3_offline.py` | Jina reranker v3 |
| `multi_vector_retrieval_offline.py` | 多向量检索 |
| `multi_vector_retrieval_online.py` | 在线多向量检索 |

---

## 📁 九、`examples/reasoning/` — 推理链示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `openai_chat_completion_with_reasoning.py` | DeepSeekR1 推理输出 |
| `openai_chat_completion_with_reasoning_streaming.py` | 流式推理输出 |
| `openai_chat_completion_tool_calls_with_reasoning.py` | QwQ 工具调用 + 推理 |
| `openai_responses_client.py` | Responses API 推理 |

---

## 📁 十、`examples/rl/` — 强化学习（RLHF）示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `rlhf_async_new_apis.py` | 异步 RLHF（Ray + NCCL 权重同步） |
| `rlhf_http_ipc.py` | HTTP + IPC 权重传输的 RLHF |
| `rlhf_http_nccl.py` | HTTP + NCCL 权重传输的 RLHF |
| `rlhf_ipc.py` | Ray + IPC 同 GPU 的 RLHF |
| `rlhf_nccl.py` | Ray + NCCL 不同 GPU 的 RLHF |
| `rlhf_nccl_fsdp_ep.py` | FSDP2 训练 + 专家并行推理 RLHF |
| `routed_experts_e2e.py` | MoE 专家路由捕获验证 |
| `skip_loading_weights_in_engine_init.py` | 虚拟模型加载 + 权重更新 |

---

## 📁 十一、`examples/speech_to_text/` — 语音转文本示例

### 语种识别 (`lid/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `openai_lid_client.py` | FireRedLID 语种识别 |

### OpenAI 转录/翻译 (`openai/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `openai_transcription_client.py` | Whisper 转录（同步 + 流式） |
| `openai_translation_client.py` | Whisper 翻译 |

### 实时 (`realtime/`)

| 文件 | 测试对象/目的 |
|------|-------------|
| `openai_realtime_client.py` | WebSocket 实时转录 |
| `openai_realtime_microphone_client.py` | Gradio 麦克风实时转录 |

---

## 📁 十二、`examples/tool_calling/` — 工具调用示例

| 文件 | 测试对象/目的 |
|------|-------------|
| `chat_with_tools_offline.py` | Mistral 离线工具调用 |
| `openai_chat_completion_client_with_tools.py` | Mistral/Hermes 在线工具调用 |
| `openai_chat_completion_client_with_tools_required.py` | 结构化必选参数工具调用 |
| `openai_chat_completion_client_with_tools_xlam.py` | xLAM-2 工具调用 |
| `openai_chat_completion_client_with_tools_xlam_streaming.py` | xLAM-2 流式工具调用 |
| `openai_responses_client_with_mcp_tools.py` | MCP 工具 + Responses API |
| `openai_responses_client_with_tools.py` | Responses API + 工具 + 推理 |

---

## 📁 十三、Jinja 模板文件（`examples/` 根目录）

### 聊天模板 (`template_*.jinja`)

用于各种模型的聊天 prompt 格式化，覆盖模型包括：
- Alpaca、Baichuan、ChatGLM/ChatGLM2、ChatML、Falcon/Falcon-180B、InkBot、TeleFLM

### 工具调用聊天模板 (`tool_chat_template_*.jinja`)

用于工具调用场景下的 prompt 格式化，覆盖主流模型包括：
- DeepSeek R1 / V3 / V3.1
- Gemma 3 (Pythonic) / Gemma 4
- GLM4
- Granite / Granite 20B FC
- Hermes
- 华为 Hunyuan A13B
- InternLM2 Tool
- Llama 3.1 (JSON) / 3.2 (JSON & Pythonic) / 4 (JSON & Pythonic)
- MiniMax M1
- Mistral / Mistral 3 / Mistral Parallel
- Phi4 Mini
- Qwen3 Coder
- ToolAce
- xLAM (Llama / Qwen)
- Function Gemma

---

## 📊 总结统计

| 类别 | Python 文件数 | 核心测试对象 |
|------|-------------|------------|
| basic | 8 | LLM/pooling 基础 API |
| features | ~30 | 各种高级功能（缓存、并行、LoRA、投机解码等） |
| generate | ~11 | 文本/多模态生成 |
| observability | 2 | 监控指标与追踪 |
| offline_inference | ~10 | 异步/分布式/分离式离线推理 |
| online_serving | ~15 | API 服务器、代理、分离式服务 |
| others | ~5 | Tensorizer、LMCache |
| pooling | ~40 | 嵌入、分类、评分、重排序、NER |
| reasoning | 4 | 推理链输出 |
| rl | 8 | RLHF 训练推理协同 |
| speech_to_text | 5 | 语音转录/翻译/识别 |
| tool_calling | 7 | 工具调用与 MCP |
| Jinja 模板 | ~35 | 聊天/工具调用 prompt 格式 |

**共计约 180+ Python 文件 + 35+ Jinja 模板**，全面覆盖了 vLLM 从基础推理到分布式部署、多模态、RLHF、工具调用等几乎所有功能模块。
