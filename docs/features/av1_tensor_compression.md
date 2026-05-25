# Experimental AV1 Tensor Compression

vLLM now includes an **experimental** tensor compression interface for
KV-cache-like tensors at:

- `vllm.v1.kv_offload.tensor_compression`

This feature is **disabled by default**. Existing inference behavior does not
change unless a compressor is explicitly created and used.

## Available codecs

- `none` (default): pass-through (`NoOpTensorCompressor`)
- `av1`: AV1-backed experimental compressor (`AV1TensorCompressor`)

## Optional dependency

The AV1 path depends on [PyAV](https://pyav.org/), which is not installed by
default.

```bash
pip install av
```

If `av` is unavailable, AV1 encode/decode raises a clear runtime error with the
install hint above.

## Example

```python
import torch

from vllm.v1.kv_offload.tensor_compression import create_tensor_compressor

# Disabled-by-default behavior:
noop = create_tensor_compressor()

# Explicitly opt into AV1:
av1 = create_tensor_compressor("av1")

x = torch.randn(1, 8, 16, dtype=torch.float16)
encoded = av1.encode(x)
decoded = av1.decode(encoded)
```

## Limitations

- Experimental API and format; not yet integrated into default vLLM execution.
- AV1 encoding introduces extra CPU work and latency.
- Round-trip behavior depends on the selected AV1 backend and its capabilities.
