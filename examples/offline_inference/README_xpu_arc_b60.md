# Running vLLM on 4x Intel ARC B60 GPUs with TP=2, DP=2, EP=true

This guide demonstrates how to run vLLM with full parallelism on 4 Intel ARC B60 GPUs
using the Qwen3.5-35B-A3B model.

## Configuration Overview

| Parameter | Value | Description |
|-----------|-------|-------------|
| **TP (Tensor Parallelism)** | 2 | Each model replica is split across 2 GPUs |
| **DP (Data Parallelism)** | 2 | 2 independent replicas process data in parallel |
| **EP (Expert Parallelism)** | true | MoE experts are partitioned across TP ranks within each DP group |
| **Total GPUs** | 4 | TP × DP = 2 × 2 = 4 |
| **Model** | Qwen3.5-35B-A3B | MoE model with ~35B params, 3B active |

### Architecture Diagram

```
4x Intel ARC B60 GPUs
├── DP Rank 0 (GPU 0, GPU 1)
│   ├── TP Rank 0 (GPU 0) ─── Experts [0, 1, ...]
│   └── TP Rank 1 (GPU 1) ─── Experts [2, 3, ...]
└── DP Rank 1 (GPU 2, GPU 3)
    ├── TP Rank 0 (GPU 2) ─── Experts [0, 1, ...]
    └── TP Rank 1 (GPU 3) ─── Experts [2, 3, ...]
```

## Known Limitation: XPU DP>1 with Multiprocessing Backend

When `DP > 1` on XPU using the multiprocessing backend (`--distributed-executor-backend mp`),
vLLM sets different `ZE_AFFINITY_MASK` values for each DP group (e.g., `0,1` for DP rank 0
and `2,3` for DP rank 1). This creates isolated Level Zero device namespaces. XCCL's IPC
mechanism uses `zeMemGetIpcHandle()` / `zeMemOpenIpcHandle()` for cross-process GPU memory
sharing, but the IPC handles contain device indices that are **local to each process's
ZE_AFFINITY_MASK namespace**. This causes GPU memory address mismatches and allreduce ring
deadlocks.

**Workaround:** Use `torchrun` with `distributed_executor_backend="external_launcher"` for
DP>1 offline inference. torchrun does NOT set `ZE_AFFINITY_MASK`, so all processes share a
unified Level Zero device namespace and XCCL IPC works correctly.

**Limitation:** `vllm serve` (OpenAI API server) uses the multiprocessing backend internally
and is therefore limited to DP=1 on XPU until the cross-affinity IPC issue is resolved.

See [PR #15](https://github.com/hongbolv/vllm/pull/15) for the full root cause analysis.

## Prerequisites

### Hardware
- 4x Intel ARC B60 GPUs
- Intel GPU drivers installed ([driver installation guide](https://dgpu-docs.intel.com/driver/installation.html))

### Software
- Python 3.12 (required for vllm-xpu-kernels)
- PyTorch with XPU support (torch >= 2.8 for xccl backend)

### Installation

```bash
# Clone vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install dependencies
pip install --upgrade pip
pip install -v -r requirements/xpu.txt

# Install triton-xpu (NOT regular triton)
pip uninstall -y triton triton-xpu
pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu

# Build vLLM for XPU
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```

### Model

The default model path is `/home/media/Hongbo/models/Qwen3.5-35B-A3B`.
Qwen3.5-35B-A3B is a Mixture-of-Experts model with ~35B total parameters
and ~3B active parameters per token. It is a ConditionalGeneration
(multimodal) model; for text-only inference, `language_model_only=True`
is required.

## Usage

### Offline Inference with DP=2 (torchrun)

```bash
torchrun --nproc-per-node=4 \
    examples/offline_inference/xpu_arc_b60_dp_ep.py
```

With custom model path:
```bash
torchrun --nproc-per-node=4 \
    examples/offline_inference/xpu_arc_b60_dp_ep.py \
    --model="/home/media/Hongbo/models/Qwen3.5-35B-A3B" \
    --max-model-len=256
```

Process layout (WORLD_SIZE=4 = TP × DP = 2 × 2):
```
RANK 0: vllm dp_rank=0, tp_rank=0   (DP group 0, TP leader)
RANK 1: vllm dp_rank=0, tp_rank=1   (DP group 0, TP follower)
RANK 2: vllm dp_rank=1, tp_rank=0   (DP group 1, TP leader)
RANK 3: vllm dp_rank=1, tp_rank=1   (DP group 1, TP follower)
```

### Online Serving (API Server, TP=2 only)

Due to the XPU DP limitation, `vllm serve` currently runs with TP=2, DP=1:

```bash
bash examples/online_serving/xpu_arc_b60_serve.sh
```

Query the server:
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/media/Hongbo/models/Qwen3.5-35B-A3B",
        "prompt": "The future of AI is",
        "max_tokens": 64
    }'
```

## Key Parameters Explained

### `--tensor-parallel-size 2` (TP=2)
Splits the model's weight tensors across 2 GPUs. Each GPU holds half of each
layer's parameters and performs partial computation, then communicates results
via all-reduce operations using the xccl backend.

### `--data-parallel-size 2` (DP=2, torchrun only)
Creates 2 independent model replicas. Each replica processes different requests
concurrently, effectively doubling throughput. Only available with torchrun
launch for offline inference.

### `--enable-expert-parallel` (EP=true)
For Mixture-of-Experts (MoE) models, distributes experts across the TP group
instead of replicating them. With TP=2 and a model having N experts, each GPU
holds N/2 experts, reducing memory per GPU.

### `--distributed-executor-backend external_launcher`
Used with torchrun for DP>1 on XPU. torchrun manages process spawning and
sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE` environment variables.

### `--enforce-eager`
Disables graph compilation. Recommended for Intel ARC B60 since XPU graph
support requires explicit opt-in via `VLLM_XPU_ENABLE_XPU_GRAPH=1`.

### `--language-model-only`
Required for Qwen3.5-35B-A3B which is a ConditionalGeneration (multimodal)
model. This flag skips multimodal components for text-only inference.

### `--dtype float16`
Uses float16 precision for inference on ARC B60.

### `--gpu-memory-utilization 0.95`
Uses 95% of available GPU memory. ARC B60 has limited VRAM, so maximize
utilization while leaving headroom for the runtime.

## Troubleshooting

### GPU Not Detected
```bash
# Verify Intel GPUs are visible
python -c "import torch; print(torch.xpu.device_count())"
# Should print: 4
```

### Out of Memory
Reduce `--max-model-len` or `--gpu-memory-utilization`:
```bash
torchrun --nproc-per-node=4 \
    examples/offline_inference/xpu_arc_b60_dp_ep.py --max-model-len=128
```

### Communication Errors
Ensure xccl backend is available:
```bash
python -c "import torch.distributed; print('xccl' in torch.distributed.Backend.backend_list)"
```
