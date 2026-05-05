# Running vLLM on 4x Intel ARC B60 GPUs with TP=2, DP=2, EP=true

This guide demonstrates how to run vLLM with full parallelism on 4 Intel ARC B60 GPUs.

## Configuration Overview

| Parameter | Value | Description |
|-----------|-------|-------------|
| **TP (Tensor Parallelism)** | 2 | Each model replica is split across 2 GPUs |
| **DP (Data Parallelism)** | 2 | 2 independent replicas process data in parallel |
| **EP (Expert Parallelism)** | true | MoE expert layers are distributed across TP ranks |
| **Total GPUs** | 4 | TP × DP = 2 × 2 = 4 |

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

Each DP rank holds a complete model replica (split across 2 GPUs via TP).
With EP enabled, MoE experts are distributed across the TP ranks instead of
being replicated, reducing memory usage for MoE models.

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

## Usage

### Offline Inference (Batch Processing)

```bash
python examples/offline_inference/xpu_arc_b60_dp_ep.py
```

With a custom model:
```bash
python examples/offline_inference/xpu_arc_b60_dp_ep.py \
    --model="ibm-research/PowerMoE-3b" \
    --max-model-len=1024
```

### Online Serving (API Server)

```bash
bash examples/online_serving/xpu_arc_b60_serve.sh
```

Query the server:
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ibm-research/PowerMoE-3b",
        "prompt": "The future of AI is",
        "max_tokens": 64
    }'
```

### Direct vllm serve Command

```bash
vllm serve ibm-research/PowerMoE-3b \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --distributed-executor-backend mp \
    --enforce-eager
```

## Key Parameters Explained

### `--tensor-parallel-size 2` (TP=2)
Splits the model's weight tensors across 2 GPUs. Each GPU holds half of each
layer's parameters and performs partial computation, then communicates results
via all-reduce operations using the xccl backend.

### `--data-parallel-size 2` (DP=2)
Creates 2 independent model replicas. Each replica processes different requests
concurrently, effectively doubling throughput. The scheduler distributes
incoming requests across DP ranks.

### `--enable-expert-parallel` (EP=true)
For Mixture-of-Experts (MoE) models, distributes experts across the TP group
instead of replicating them. With TP=2 and a model having N experts, each GPU
holds N/2 experts. This reduces memory per GPU while maintaining model capacity.

### `--distributed-executor-backend mp`
Uses Python multiprocessing for launching workers. This is the recommended
backend for single-node XPU deployments.

### `--enforce-eager`
Disables graph compilation. Recommended for Intel ARC B60 since XPU graph
support requires explicit opt-in via `VLLM_XPU_ENABLE_XPU_GRAPH=1`.

### `--dtype bfloat16`
Uses bfloat16 precision for inference. Intel ARC B60 supports bfloat16
natively, providing good performance with lower memory usage.

## Performance Considerations

1. **Memory**: ARC B60 has limited VRAM compared to data center GPUs.
   Use `--max-model-len` to limit context length and reduce memory usage.

2. **Communication**: The xccl backend handles all inter-GPU communication.
   Ensure all 4 GPUs are on the same PCIe root complex for best performance.

3. **Expert Parallelism**: EP reduces memory per GPU for MoE models but adds
   all-to-all communication overhead. The `allgather_reducescatter` backend
   is used by default on XPU.

4. **Model Selection**: Choose MoE models (e.g., PowerMoE-3b, Mixtral) to
   benefit from Expert Parallelism. Non-MoE models will still work with
   TP=2, DP=2 but EP has no effect.

## Troubleshooting

### GPU Not Detected
```bash
# Verify Intel GPUs are visible
python -c "import torch; print(torch.xpu.device_count())"
# Should print: 4
```

### Out of Memory
Reduce `--max-model-len` or choose a smaller model:
```bash
python examples/offline_inference/xpu_arc_b60_dp_ep.py --max-model-len=512
```

### Communication Errors
Ensure xccl backend is available:
```bash
python -c "import torch.distributed; print('xccl' in torch.distributed.Backend.backend_list)"
```

## Supported Models

Any MoE model supported by vLLM can benefit from this EP configuration:
- `ibm-research/PowerMoE-3b` (recommended for testing, small model)
- `mistralai/Mixtral-8x7B-v0.1` (requires more VRAM)
- Other MoE architectures supported by vLLM

For non-MoE models, remove `--enable-expert-parallel` and use TP=2, DP=2:
```bash
vllm serve meta-llama/Llama-3.2-1B \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --distributed-executor-backend mp \
    --enforce-eager
```
