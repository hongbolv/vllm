# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.kv_offload.tensor_compression import (
    AV1TensorCompressor,
    NoOpTensorCompressor,
    create_tensor_compressor,
)


def test_tensor_compressor_factory_default_is_noop():
    compressor = create_tensor_compressor()
    assert isinstance(compressor, NoOpTensorCompressor)


def test_noop_tensor_compressor_round_trip():
    tensor = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    compressor = NoOpTensorCompressor()
    encoded = compressor.encode(tensor)
    decoded = compressor.decode(encoded)

    assert decoded.shape == tensor.shape
    assert decoded.dtype == tensor.dtype
    assert torch.equal(decoded, tensor)


def test_tensor_compressor_factory_rejects_unknown_codec():
    with pytest.raises(ValueError, match="Unsupported tensor compression codec"):
        create_tensor_compressor("unknown_codec")


def test_av1_tensor_compressor_dependency_check(monkeypatch):
    def raise_missing_av():
        raise RuntimeError(
            "AV1 tensor compression requires optional dependency 'av'. "
            "Install it with: pip install av"
        )

    compressor = AV1TensorCompressor()
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tensor_compression._load_pyav",
        raise_missing_av,
    )

    tensor = torch.zeros((2, 2), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="pip install av"):
        compressor.encode(tensor)
