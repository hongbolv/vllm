# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


class XpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

            else:  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on XPU. "
                    "Falling back to AgRs manager for XPU, "
                    "which is the Default backend",
                    self.all2all_backend,  # type: ignore[has-type]
                )
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        output = input_.clone() if torch.compiler.is_compiling() else input_
        dist.all_reduce(output, group=self.device_group)
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        import sys
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            # XCCL does not support variable-size dist.reduce_scatter.
            # Use all_reduce + slice as a workaround: all_reduce the full
            # concatenated tensor so every rank holds the reduced result,
            # then each rank extracts its own variable-size slice.
            print(
                f"[XPU reduce_scatterv] rank={self.rank_in_group} "
                f"BEFORE all_reduce (variable-size) sizes={sizes} "
                f"input_tensor.shape={input_tensor.shape}",
                flush=True,
                file=sys.stderr,
            )
            dist.all_reduce(input_tensor, group=self.device_group)
            print(
                f"[XPU reduce_scatterv] rank={self.rank_in_group} "
                f"AFTER all_reduce (variable-size)",
                flush=True,
                file=sys.stderr,
            )
            offset = sum(sizes[: self.rank_in_group])
            output.copy_(input_tensor[offset : offset + chunk_size])
        else:
            print(
                f"[XPU reduce_scatterv] rank={self.rank_in_group} "
                f"BEFORE reduce_scatter_tensor (equal-size) "
                f"input_tensor.shape={input_tensor.shape}",
                flush=True,
                file=sys.stderr,
            )
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
            print(
                f"[XPU reduce_scatterv] rank={self.rank_in_group} "
                f"AFTER reduce_scatter_tensor (equal-size)",
                flush=True,
                file=sys.stderr,
            )
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        import sys
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        print(
            f"[XPU all_gatherv] rank={self.rank_in_group} ENTER "
            f"sizes={sizes} "
            f"input_type={'list' if isinstance(input_, list) else 'tensor'} "
            f"input_shape={[x.shape for x in input_] if isinstance(input_, list) else input_.shape}",
            flush=True,
            file=sys.stderr,
        )

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

            if sizes is not None:
                # XCCL does not support variable-size dist.all_gather.
                # Workaround: pad this rank's input to max(sizes), perform an
                # equal-size all_gather_into_tensor (which XCCL supports), then
                # strip the padding from each rank's slice.
                max_size = max(sizes)
                padded = torch.empty(
                    (max_size,) + input_.shape[1:],
                    dtype=input_.dtype,
                    device=input_.device,
                )
                padded[: sizes[self.rank_in_group]].copy_(input_)
                gathered = torch.empty(
                    (world_size * max_size,) + input_.shape[1:],
                    dtype=input_.dtype,
                    device=input_.device,
                )
                print(
                    f"[XPU all_gatherv] rank={self.rank_in_group} "
                    f"BEFORE all_gather_into_tensor (variable-size) "
                    f"sizes={sizes} max_size={max_size} "
                    f"padded.shape={padded.shape} gathered.shape={gathered.shape}",
                    flush=True,
                    file=sys.stderr,
                )
                dist.all_gather_into_tensor(
                    gathered, padded, group=self.device_group
                )
                print(
                    f"[XPU all_gatherv] rank={self.rank_in_group} "
                    f"AFTER all_gather_into_tensor (variable-size)",
                    flush=True,
                    file=sys.stderr,
                )
                # Extract each rank's real (unpadded) rows.
                chunks = [
                    gathered[r * max_size : r * max_size + size]
                    for r, size in enumerate(sizes)
                ]
                output_tensor = torch.cat(chunks, dim=0)
            else:
                # Equal-size path: use all_gather_into_tensor (XCCL supported)
                # instead of dist.all_gather(list, ...) which may hang on XCCL.
                print(
                    f"[XPU all_gatherv] rank={self.rank_in_group} "
                    f"BEFORE all_gather_into_tensor (equal-size) "
                    f"input_.shape={input_.shape} output_tensor.shape={output_tensor.shape}",
                    flush=True,
                    file=sys.stderr,
                )
                dist.all_gather_into_tensor(
                    output_tensor, input_, group=self.device_group
                )
                print(
                    f"[XPU all_gatherv] rank={self.rank_in_group} "
                    f"AFTER all_gather_into_tensor (equal-size)",
                    flush=True,
                    file=sys.stderr,
                )
            return output_tensor

        if isinstance(input_, torch.Tensor):
            result = _all_gather_single(input_, sizes)
            print(
                f"[XPU all_gatherv] rank={self.rank_in_group} EXIT "
                f"output.shape={result.shape}",
                flush=True,
                file=sys.stderr,
            )
            return result

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        print(
            f"[XPU all_gatherv] rank={self.rank_in_group} EXIT "
            f"output_list len={len(output_list)}",
            flush=True,
            file=sys.stderr,
        )
        return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        dist.broadcast(input_, src=src, group=self.device_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )
