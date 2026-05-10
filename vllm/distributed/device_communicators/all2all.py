# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.import_utils import has_deep_ep, has_mori

from .base_device_communicator import All2AllManagerBase, Cache

if has_flashinfer_nvlink_two_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_alltoall import (
        MnnvlMoe,  # type: ignore[import-not-found]
    )

if has_flashinfer_nvlink_one_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_moe_alltoall import (
        MoeAlltoAll,  # type: ignore[import-not-found]
        moe_a2a_get_workspace_size_per_rank,
    )


logger = init_logger(__name__)


class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

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
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        dist.barrier(group=dist_group.device_group)
        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

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
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        dist.barrier(group=dist_group.device_group)
        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        dist.barrier(group=dist_group.device_group)
        input_shape_before = hidden_states.shape

        # --- Force-zero padding positions before reduce_scatterv ---
        # When DP padding is active (all sizes equal), padding tokens have
        # zero input but produce non-zero expert output due to bias terms.
        # These non-zero values can corrupt real token hidden states through
        # the reduce_scatter operation. Zero them out before scattering.
        all_sizes_equal = (len(set(sizes)) == 1) if sizes else False
        if all_sizes_equal:
            num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu
            dp_size = len(num_tokens_across_dp)
            num_chunks = len(sizes)
            # sizes may have more entries than dp_size when sequence
            # parallelism is enabled (dp_size * sp_size entries).
            # Map each chunk back to its DP rank to get real token count.
            sp_size = num_chunks // dp_size if dp_size > 0 else 1
            offset = 0
            for i, chunk_size in enumerate(sizes):
                dp_rank_idx = i // sp_size if sp_size > 0 else i
                if dp_rank_idx < dp_size:
                    real_count = int(
                        num_tokens_across_dp[dp_rank_idx].item())
                    # For SP chunks, real_count is the full DP rank count;
                    # each SP chunk gets ceil(real_count / sp_size) tokens.
                    if sp_size > 1:
                        sp_chunk_idx = i % sp_size
                        # Distribute real tokens across SP chunks
                        per_sp = (real_count + sp_size - 1) // sp_size
                        sp_real = min(per_sp,
                                      max(0, real_count - sp_chunk_idx
                                           * per_sp))
                    else:
                        sp_real = real_count
                    if sp_real < chunk_size:
                        pad_start = offset + sp_real
                        pad_end = offset + chunk_size
                        hidden_states[pad_start:pad_end, :] = 0
                offset += chunk_size
        # --- End force-zero ---

        # --- NaN detection BEFORE reduce_scatterv ---
        dp_rank = get_dp_group().rank_in_group
        has_nan_before = bool(torch.isnan(hidden_states).any().item())
        has_inf_before = bool(torch.isinf(hidden_states).any().item())
        if has_nan_before or has_inf_before:
            nan_count = int(torch.isnan(hidden_states).sum().item())
            inf_count = int(torch.isinf(hidden_states).sum().item())
            # Find which rows contain NaN
            nan_rows = torch.isnan(hidden_states).any(dim=-1)
            nan_row_indices = torch.where(nan_rows)[0].tolist()
            print(
                f"[NAN_CHECK] dp_rank={dp_rank} "
                f"NaN/Inf detected BEFORE reduce_scatterv! "
                f"nan_count={nan_count} inf_count={inf_count} "
                f"input_shape={list(hidden_states.shape)} "
                f"nan_row_indices={nan_row_indices[:10]}"
                f"{'... ' if len(nan_row_indices) > 10 else ' '}"
                f"total_nan_rows={len(nan_row_indices)}",
                flush=True,
            )
        else:
            print(
                f"[NAN_CHECK] dp_rank={dp_rank} "
                f"No NaN/Inf BEFORE reduce_scatterv. "
                f"input_shape={list(hidden_states.shape)} "
                f"input_norm={hidden_states.float().norm().item():.6f}",
                flush=True,
            )
        # --- End NaN detection BEFORE ---

        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)

        # --- NaN detection AFTER reduce_scatterv ---
        has_nan_after = bool(torch.isnan(hidden_states).any().item())
        has_inf_after = bool(torch.isinf(hidden_states).any().item())
        if has_nan_after or has_inf_after:
            nan_count = int(torch.isnan(hidden_states).sum().item())
            inf_count = int(torch.isinf(hidden_states).sum().item())
            nan_rows = torch.isnan(hidden_states).any(dim=-1)
            nan_row_indices = torch.where(nan_rows)[0].tolist()
            print(
                f"[NAN_CHECK] dp_rank={dp_rank} "
                f"NaN/Inf detected AFTER reduce_scatterv! "
                f"nan_count={nan_count} inf_count={inf_count} "
                f"output_shape={list(hidden_states.shape)} "
                f"nan_row_indices={nan_row_indices[:10]}"
                f"{'... ' if len(nan_row_indices) > 10 else ' '}"
                f"total_nan_rows={len(nan_row_indices)} "
                f"had_nan_before={has_nan_before}",
                flush=True,
            )
        # --- End NaN detection AFTER ---

        # --- Diagnostic: check padding positions after reduce_scatterv ---
        all_sizes_equal = (len(set(sizes)) == 1) if sizes else False
        if all_sizes_equal and sizes[dp_rank] > 0:
            output_rows = hidden_states.shape[0]
            # Compute per-row norms to detect padding positions with
            # non-zero values (indicating expert bias producing non-zero
            # output for zero-padded input)
            row_norms = hidden_states.float().norm(dim=-1)
            nonzero_rows = int((row_norms > 0).sum().item())
            last_n = min(4, output_rows)
            last_row_norms = [
                f"{row_norms[output_rows - last_n + i].item():.6f}"
                for i in range(last_n)
            ]
            print(
                f"[REDUCE_SCATTER_CHECK] dp_rank={dp_rank} "
                f"sizes={sizes} "
                f"input_shape={list(input_shape_before)} "
                f"output_shape={list(hidden_states.shape)} "
                f"output_rows={output_rows} "
                f"nonzero_rows={nonzero_rows} "
                f"output_norm={hidden_states.float().norm().item():.6f} "
                f"output_min={hidden_states.min().item():.6f} "
                f"output_max={hidden_states.max().item():.6f} "
                f"last_{last_n}_row_norms=[{', '.join(last_row_norms)}]",
                flush=True,
            )
            # Error check: if nonzero_rows equals output_rows when DP
            # padding is active, it means ALL rows (including padding
            # positions) have non-zero values. This indicates expert
            # bias terms produced non-zero output for zero-padded input,
            # which may corrupt real token hidden states through
            # reduce_scatter.
            if nonzero_rows == output_rows:
                print(
                    f"[REDUCE_SCATTER_CHECK] ERROR: dp_rank={dp_rank} "
                    f"ALL {output_rows} output rows are non-zero "
                    f"(nonzero_rows={nonzero_rows}). "
                    f"Padding positions likely have non-zero values "
                    f"after expert computation (expert bias on zero "
                    f"input). This may corrupt real token hidden states "
                    f"through reduce_scatter cut boundary shift.",
                    flush=True,
                )
        # --- End diagnostic ---

        return hidden_states

    def destroy(self):
        pass


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_deep_ep(), (
            "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md"
            " to install DeepEP kernels."
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself."
        )

        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    def set_num_sms(self, num_sms: int):
        import deep_ep  # type: ignore[import-not-found]

        # Right now the buffers are sized for only what the kernels were
        # created with. So we can only reduce the number of SMS used
        # but not increase it.
        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """
        max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import deep_ep  # type: ignore[import-not-found]

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    # DeepEP LL uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0


class NixlEPAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NIXL EP kernels.
    This backend supports elastic EP with dynamic rank connection/disconnection.
    """

    # (nixl_ep_buffer, ep_size)
    _buffer: tuple[Any, int] | None = None
    _lock = threading.Lock()

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

        self.max_num_ep_ranks = envs.VLLM_NIXL_EP_MAX_NUM_RANKS

    def _init_buffer(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_experts_per_rank: int,
    ) -> None:
        from nixl_ep import Buffer  # type: ignore[import-not-found]

        max_num_global_experts = self.max_num_ep_ranks * num_experts_per_rank
        num_rdma_bytes = Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=self.max_num_ep_ranks,
            num_experts=max_num_global_experts,
        )
        assert NixlEPAll2AllManager._buffer is None, (
            "NIXL EP buffer already initialized"
        )
        buffer = Buffer(
            rank=self.rank,
            tcp_store_group=self.tcp_store_group.store,
        )
        buffer.update_memory_buffers(
            num_ranks=self.max_num_ep_ranks,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )
        ranks_to_connect = list(range(self.cpu_group.size()))
        buffer.connect_ranks(ranks_to_connect)
        NixlEPAll2AllManager._buffer = (buffer, self.cpu_group.size())

    def _update_buffer(self):
        assert NixlEPAll2AllManager._buffer is not None
        buffer, current_ep_size = NixlEPAll2AllManager._buffer
        current_ranks = list(range(current_ep_size))
        new_ep_size = self.cpu_group.size()
        buffer.set_tcp_store_group(self.tcp_store_group.store)
        if new_ep_size > len(current_ranks):
            ranks_to_connect = list(range(len(current_ranks), new_ep_size))
            buffer.connect_ranks(ranks_to_connect)
        else:
            ranks_to_disconnect = current_ranks[new_ep_size:]
            buffer.disconnect_ranks(ranks_to_disconnect)
        NixlEPAll2AllManager._buffer = (buffer, new_ep_size)

    def get_handle(self, kwargs):
        with NixlEPAll2AllManager._lock:
            if (
                NixlEPAll2AllManager._buffer is not None
                and NixlEPAll2AllManager._buffer[1] == self.cpu_group.size()
            ):
                return NixlEPAll2AllManager._buffer[0]

            num_experts_per_rank = (
                kwargs["num_global_experts"] // kwargs["num_ep_ranks"]
            )
            nixl_kwargs = dict(
                max_num_tokens_per_dp_rank=kwargs["max_num_tokens_per_dp_rank"],
                token_hidden_size=kwargs["token_hidden_size"],
                num_experts_per_rank=num_experts_per_rank,
            )
            if NixlEPAll2AllManager._buffer is None:
                self._init_buffer(**nixl_kwargs)
            else:
                self._update_buffer()

            assert NixlEPAll2AllManager._buffer is not None
            handle = NixlEPAll2AllManager._buffer[0]
            return handle

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
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        # NOTE(yongji): NIXLEPAll2AllManager instance is recreated during
        # scale-up/down, so we cannot destroy the persistent buffer here.
        assert NixlEPAll2AllManager._buffer is not None
        buffer = NixlEPAll2AllManager._buffer[0]
        buffer.set_tcp_store_group(None)

    # NIXL EP uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0


class FlashInferNVLinkTwoSidedManager(All2AllManagerBase):
    """
    All2All communication based on flashinfer all2allv/two-sided NVLink kernels.
    """

    # This type lint could be removed after all of the work in
    # https://github.com/vllm-project/vllm/issues/26533 done.
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_flashinfer_nvlink_two_sided(), (
            "flashinfer all2all module not found. Please install/check flashinfer"
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
        logger.debug(
            "Initialize for flashinfer All2All rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.alltoall_info = None

    def initialize(
        self,
        world_size: int,
        rank: int,
        gpus_per_node: int,
    ):
        """Initialize workspace"""
        if self.initialized:
            return

        self.cleanup()
        logger.debug("making map: rank=%d, world size=%d", rank, world_size)
        self.mapping = Mapping(
            world_size,
            rank,
            gpus_per_node,
            tp_size=world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        # MNNVL workspace is allocated per rank in the comm_backend's group; the
        # flashinfer kernel asserts workspace.size(0) == moe_ep_size, so the backend
        # must span the EP group (= DP*PCP*TP), not the DP group.
        ep_config = MnnvlConfig(
            comm_backend=CustomCommunicator(self.cpu_group),
            fabric_page_size=1 << 29,  # 512MB
            allocation_granularity=0,  # Auto-detect
        )

        self.workspace_tensor = MnnvlMoe.get_moe_workspaces(self.mapping, ep_config)
        self.prepare_workspace_tensor = MnnvlMoe.get_moe_prepare_workspace(
            self.mapping, ep_config
        )

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            "FlashInfer All2All initialized for rank %s, size %s", rank, world_size
        )

    def ensure_alltoall_workspace_initialized(self):
        """Ensure workspace is initialized"""
        if not has_flashinfer_nvlink_two_sided():
            return False

        if self.world_size <= 1:
            return False

        if not self.initialized:
            self.initialize(
                world_size=self.world_size,
                rank=self.rank,
                gpus_per_node=torch.accelerator.device_count,
            )
        return self.initialized

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up workspace"""
        if (
            self.initialized
            and self.workspace_tensor is not None
            and self.prepare_workspace_tensor is not None
        ):
            try:
                del self.workspace_tensor
                del self.prepare_workspace_tensor
            except Exception as e:
                logger.warning("Failed to cleanup FlashInfer workspace: %s", e)
            finally:
                self.workspace_tensor = None
                self.prepare_workspace_tensor = None
                self.mapping = None
                self.initialized = False


class FlashInferNVLinkOneSidedManager(All2AllManagerBase):
    """
    All2All communication based on FlashInfer's MoeAlltoAll/One-sided NVLink kernel.
    This is a newer kernel from trtllm that should perform better than the kernel
    used by flashinfer_nvlink_two_sided.
    """

    rank: int
    world_size: int

    def __init__(self, cpu_group):
        assert has_flashinfer_nvlink_one_sided(), (
            "flashinfer trtllm_moe_alltoall module not found. "
            "Please install/check flashinfer"
        )
        super().__init__(cpu_group)
        logger.debug(
            "Initialize FlashInfer One-sided NVLink rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.moe_alltoall: MoeAlltoAll | None = None
        self.mapping = None

    def initialize(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dispatch_dtype_bytes_per_elem: int = 0,
        dispatch_scale_bytes_per_token: int = 0,
    ):
        """Initialize the MoeAlltoAll workspace."""
        if self.initialized:
            return

        self.cleanup()
        gpus_per_node = torch.accelerator.device_count()
        logger.debug(
            "Making One-sided NVLink mapping: rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.mapping = Mapping(
            self.world_size,
            self.rank,
            gpus_per_node,
            tp_size=self.world_size,
            moe_ep_size=self.world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        # MNNVL workspace is allocated per rank in the comm_backend's group; the
        # flashinfer kernel asserts workspace.size(0) == moe_ep_size, so the backend
        # must span the EP group (= DP*PCP*TP), not the DP group.
        ep_config = MnnvlConfig(
            comm_backend=CustomCommunicator(self.cpu_group),
        )
        if dispatch_dtype_bytes_per_elem == 0:
            hidden_bytes = hidden_size // 2
        else:
            hidden_bytes = hidden_size * dispatch_dtype_bytes_per_elem
        total_dispatch_payload_size_per_token = (
            hidden_bytes
            + dispatch_scale_bytes_per_token
            + top_k * 4  # int32 topks ids
            + top_k * 4  # float32 topk weights
        )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states
        self.workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.world_size,
            max_num_tokens=max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )

        self.moe_alltoall = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=ep_config,
        )

        self.gpus_per_node = gpus_per_node
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.initialized = True

        logger.info(
            "FlashInfer One-sided NVLink initialized for rank %s, size %s",
            self.rank,
            self.world_size,
        )
        dist.barrier()

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up resources."""
        if self.initialized and self.moe_alltoall is not None:
            try:
                del self.moe_alltoall
            except Exception as e:
                logger.warning(
                    "Failed to cleanup FlashInfer One-sided NVLink workspace: %s", e
                )
            finally:
                self.moe_alltoall = None
                self.mapping = None
                self.initialized = False


class MoriAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )  # noqa
        import mori

        super().__init__(cpu_group)
        self.handle_cache = Cache()

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def _make_all2all_kwargs(
        self,
        rank: int,
        num_ep_ranks: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        token_hidden_size: int,
        scale_dim: int,
        scale_type_size: int,
        max_num_tokens_per_dp_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
    ):
        import mori  # type: ignore[import-not-found]

        from vllm.platforms.rocm import on_gfx942, on_gfx950

        assert on_gfx942() or on_gfx950(), (
            "mori currently only support arch gfx942 and gfx950"
        )

        if not self.internode:
            # single node
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            rdma_block_num = 0
            warp_num_per_block = 16
            block_num = 80
        else:
            # multi node
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
            if on_gfx942():
                warp_num_per_block = 16
                block_num = 32
                rdma_block_num = 16
            elif on_gfx950():
                warp_num_per_block = 8
                block_num = 64
                rdma_block_num = 32
            else:
                raise NotImplementedError(
                    "mori currently only support arch gfx942 and gfx950"
                )

        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            data_type=quant_dtype,
            hidden_dim=token_hidden_size,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=input_dtype.itemsize,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            kernel_type=kernel_type,
            rdma_block_num=rdma_block_num,
            gpu_per_node=min(8, num_ep_ranks),
        )

    def _make_handle(self, **kwargs):
        import mori  # type: ignore[import-not-found]

        mori_config = mori.ops.EpDispatchCombineConfig(**kwargs)
        handle = mori.ops.EpDispatchCombineOp(mori_config)
        return handle

    def get_handle(self, kwargs):
        import mori  # type: ignore[import-not-found]

        mori_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("MoRI all2all args %s", mori_kwargs)
        handle: mori.ops.EpDispatchCombineOp = self.handle_cache.get_or_create(
            mori_kwargs, self._make_handle
        )
        return handle
