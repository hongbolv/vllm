# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.kv_offload.tensor_compression import (
    AV1TensorCompressor,
    EncodedTensor,
    NoOpTensorCompressor,
    create_tensor_compressor,
)

MISSING_AV_ERROR = (
    "AV1 tensor compression requires optional dependency 'av'. "
    "Install it with: uv pip install av"
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


def test_av1_tensor_compressor_rejects_non_positive_frame_width():
    with pytest.raises(ValueError, match="max_frame_width must be positive"):
        AV1TensorCompressor(max_frame_width=0)


def test_av1_tensor_compressor_dependency_check(monkeypatch):
    def raise_missing_av():
        raise RuntimeError(MISSING_AV_ERROR)

    compressor = AV1TensorCompressor()
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tensor_compression._load_pyav",
        raise_missing_av,
    )

    tensor = torch.zeros((2, 2), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="uv pip install av"):
        compressor.encode(tensor)


def test_av1_tensor_compressor_decode_dependency_check(monkeypatch):
    def raise_missing_av():
        raise RuntimeError(MISSING_AV_ERROR)

    compressor = AV1TensorCompressor()
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tensor_compression._load_pyav",
        raise_missing_av,
    )
    encoded = EncodedTensor(
        payload=b"not-used",
        metadata={
            "shape": [1],
            "numpy_dtype": "uint8",
            "original_num_bytes": 1,
        },
    )
    with pytest.raises(RuntimeError, match="uv pip install av"):
        compressor.decode(encoded)


def test_av1_tensor_compressor_round_trip_when_available():
    av = pytest.importorskip("av")
    compressor = AV1TensorCompressor(max_frame_width=16)
    tensor = torch.arange(64, dtype=torch.uint8).reshape(8, 8)

    codec_module = getattr(av, "codec", None)
    codec_cls = getattr(codec_module, "Codec", None)
    if codec_cls is None:
        pytest.skip("PyAV codec introspection is unavailable in this environment")
    try:
        codec_cls("libaom-av1", "w")
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"AV1 backend unavailable in this environment: {exc}")

    encoded = compressor.encode(tensor)
    decoded = compressor.decode(encoded)

    assert decoded.shape == tensor.shape
    assert decoded.dtype == tensor.dtype
    assert torch.equal(decoded, tensor)


def test_av1_tensor_compressor_decode_raises_when_no_frames(monkeypatch):
    class _FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, video: int = 0):
            del video
            return iter(())

    class _FakeAv:
        @staticmethod
        def open(*_args, **_kwargs):
            return _FakeContainer()

    compressor = AV1TensorCompressor()
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tensor_compression._load_pyav",
        lambda: _FakeAv,
    )
    encoded = EncodedTensor(
        payload=b"not-used",
        metadata={
            "shape": [1],
            "numpy_dtype": "uint8",
            "original_num_bytes": 1,
        },
    )
    with pytest.raises(RuntimeError, match="Failed to decode AV1 tensor payload"):
        compressor.decode(encoded)
