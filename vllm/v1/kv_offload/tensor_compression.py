# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental tensor compression helpers for KV-cache-like tensors."""

from __future__ import annotations

import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class EncodedTensor:
    payload: bytes
    metadata: dict[str, Any]


class TensorCompressor(ABC):
    """Interface for tensor compression backends."""

    @abstractmethod
    def encode(self, tensor: torch.Tensor) -> EncodedTensor:
        """
        Encode a tensor into a compressed payload.

        Args:
            tensor: Input tensor to compress.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        encoded: EncodedTensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Decode a tensor payload back to a tensor.

        Args:
            encoded: Encoded payload and metadata.
            device: Optional target device for the decoded tensor.
                If omitted, the tensor remains on CPU.
        """
        raise NotImplementedError


def _encode_tensor_bytes(tensor: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    cpu_tensor = tensor.detach().contiguous().cpu()
    np_tensor = cpu_tensor.numpy()
    payload = np_tensor.view(np.uint8).reshape(-1)
    metadata = {
        "shape": list(cpu_tensor.shape),
        "numpy_dtype": np_tensor.dtype.name,
    }
    return payload, metadata


def _decode_tensor_bytes(
    payload: bytes,
    metadata: dict[str, Any],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    shape = tuple(int(v) for v in metadata["shape"])
    np_dtype = np.dtype(metadata["numpy_dtype"])
    np_tensor = np.frombuffer(payload, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(np_tensor.copy())
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _load_pyav():
    try:
        import av
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AV1 tensor compression requires optional dependency 'av'. "
            "Install it with: uv pip install av"
        ) from exc
    return av


class NoOpTensorCompressor(TensorCompressor):
    """A pass-through compressor."""

    def encode(self, tensor: torch.Tensor) -> EncodedTensor:
        payload, metadata = _encode_tensor_bytes(tensor)
        metadata["codec"] = "none"
        return EncodedTensor(payload.tobytes(), metadata)

    def decode(
        self,
        encoded: EncodedTensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return _decode_tensor_bytes(encoded.payload, encoded.metadata, device=device)


class AV1TensorCompressor(TensorCompressor):
    """AV1-backed tensor compressor for experimentation."""

    def __init__(
        self,
        max_frame_width: int = 1024,
        codec_name: str = "libaom-av1",
        codec_options: dict[str, str] | None = None,
    ):
        """
        Initialize an AV1 tensor compressor.

        Args:
            max_frame_width: Width used to pack byte-stream rows into a frame.
                A moderate default avoids excessively wide frames.
            codec_name: AV1 codec name recognized by FFmpeg/PyAV.
            codec_options: Optional codec configuration (e.g. lossless mode).
        """
        if max_frame_width <= 0:
            raise ValueError(
                f"max_frame_width must be positive, got {max_frame_width}"
            )
        self.max_frame_width = max_frame_width
        self.codec_name = codec_name
        self.codec_options = codec_options or {"lossless": "1"}

    def encode(self, tensor: torch.Tensor) -> EncodedTensor:
        av = _load_pyav()
        raw_bytes, metadata = _encode_tensor_bytes(tensor)
        num_bytes = int(raw_bytes.size)

        width = self.max_frame_width
        height = max(1, math.ceil(num_bytes / width))
        padded = np.zeros(width * height, dtype=np.uint8)
        padded[:num_bytes] = raw_bytes
        image = padded.reshape(height, width)

        output = io.BytesIO()
        with av.open(output, mode="w", format="matroska") as container:
            stream = container.add_stream(self.codec_name, rate=1)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "gray"
            stream.options = self.codec_options

            frame = av.VideoFrame.from_ndarray(image, format="gray")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

        metadata.update({
            "codec": "av1",
            "height": height,
            "width": width,
            "original_num_bytes": num_bytes,
        })
        return EncodedTensor(output.getvalue(), metadata)

    def decode(
        self,
        encoded: EncodedTensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        av = _load_pyav()
        with av.open(
            io.BytesIO(encoded.payload),
            mode="r",
            format="matroska",
        ) as container:
            frame = next(container.decode(video=0), None)
        if frame is None:
            raise RuntimeError("Failed to decode AV1 tensor payload")

        decoded = frame.to_ndarray(format="gray").reshape(-1)
        original_num_bytes = int(encoded.metadata["original_num_bytes"])
        raw = decoded[:original_num_bytes].tobytes()
        return _decode_tensor_bytes(raw, encoded.metadata, device=device)


def create_tensor_compressor(
    codec: str | None = None,
    **kwargs: Any,
) -> TensorCompressor:
    """
    Create a tensor compressor.

    The default mode (``codec=None``) is pass-through and does not alter data.
    """
    codec_name = (codec or "none").lower()
    if codec_name in ("none", "raw"):
        return NoOpTensorCompressor()
    if codec_name == "av1":
        return AV1TensorCompressor(**kwargs)
    raise ValueError(f"Unsupported tensor compression codec: {codec_name}")
