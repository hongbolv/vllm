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
        # Per-rank counter for MoE collective diagnosis:
        # incremented (+1) before each collective+sync, decremented (-1) after.
        # If the last printed counter is 0, the collective completed normally.
        # A stuck counter (last value > 0) identifies the hanging collective.
        self._collective_counter: int = 0
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
            # if inputs shape in different ranks is not the same using reduce_scatter
            input_splits = list(input_tensor.split(sizes, dim=0))
            self._collective_counter += 1
            print(
                f"[COUNTER] rank={self.rank_in_group} "
                f"reduce_scatterv/variable-size counter={self._collective_counter}",
                flush=True,
            )
            torch.xpu.synchronize()
            dist.reduce_scatter(output, input_splits, group=self.device_group)
            torch.xpu.synchronize()
            self._collective_counter -= 1
            print(
                f"[COUNTER] rank={self.rank_in_group} "
                f"reduce_scatterv/variable-size counter={self._collective_counter}",
                flush=True,
            )
        else:
            self._collective_counter += 1
            print(
                f"[COUNTER] rank={self.rank_in_group} "
                f"reduce_scatterv/uniform counter={self._collective_counter}",
                flush=True,
            )
            torch.xpu.synchronize()
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
            torch.xpu.synchronize()
            self._collective_counter -= 1
            print(
                f"[COUNTER] rank={self.rank_in_group} "
                f"reduce_scatterv/uniform counter={self._collective_counter}",
                flush=True,
            )
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

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
                all_gather_list = []
                for size in sizes:
                    all_gather_list.append(
                        torch.empty(
                            (size,) + input_.shape[1:],
                            dtype=input_.dtype,
                            device=input_.device,
                        )
                    )
                self._collective_counter += 1
                print(
                    f"[COUNTER] rank={self.rank_in_group} "
                    f"all_gatherv/variable-size counter={self._collective_counter}",
                    flush=True,
                )
                torch.xpu.synchronize()
                dist.all_gather(all_gather_list, input_, group=self.device_group)
                torch.xpu.synchronize()
                self._collective_counter -= 1
                print(
                    f"[COUNTER] rank={self.rank_in_group} "
                    f"all_gatherv/variable-size counter={self._collective_counter}",
                    flush=True,
                )
                output_tensor = torch.cat(all_gather_list, dim=0)
            else:
                self._collective_counter += 1
                print(
                    f"[COUNTER] rank={self.rank_in_group} "
                    f"all_gatherv/uniform counter={self._collective_counter}",
                    flush=True,
                )
                torch.xpu.synchronize()
                dist.all_gather_into_tensor(
                    output_tensor, input_, group=self.device_group
                )
                torch.xpu.synchronize()
                self._collective_counter -= 1
                print(
                    f"[COUNTER] rank={self.rank_in_group} "
                    f"all_gatherv/uniform counter={self._collective_counter}",
                    flush=True,
                )
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        # For list inputs: concatenate all tensors into ONE combined tensor,
        # perform a SINGLE collective, then split the output back.
        #
        # Sequential calls (one per tensor in the list) create multiple XCCL
        # operations on the same communicator group.  On XPU,
        # torch.xpu.synchronize() only drains local compute ops and does NOT
        # wait for cross-device XCCL collectives to globally complete.  So a
        # faster DP group can finish tensor-1's collective and submit tensor-2's
        # collective before the slower group has even called tensor-1 — XCCL
        # call-order mismatch → deadlock.  A single bulk collective eliminates
        # all such ordering hazards.
        orig_shapes = [inp.shape for inp in input_]
        # Flatten each tensor to 2-D [tokens, features] so we can concatenate
        # along the features dimension regardless of original trailing shape.
        tensors_2d = [inp.reshape(inp.shape[0], -1) for inp in input_]
        feature_sizes = [t.shape[1] for t in tensors_2d]
        combined = torch.cat(tensors_2d, dim=1)  # [tokens, sum_features]

        # ONE collective for the combined tensor.
        gathered = _all_gather_single(combined, sizes=sizes)

        # Split back along the features dimension and restore original shapes.
        results = []
        offset = 0
        for orig_shape, fsz in zip(orig_shapes, feature_sizes):
            chunk = gathered[:, offset : offset + fsz]
            new_shape = (chunk.shape[0],) + orig_shape[1:]
            results.append(chunk.reshape(new_shape))
            offset += fsz
        return results

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
