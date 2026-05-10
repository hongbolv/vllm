# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group, get_ep_group

# NaN check diagnostics with smart sampling:
# - First 50 NaN detections: print full detail
# - After that: print summary every 100 calls
# This gives early detail + global coverage across the entire inference.
_NAN_DETAIL_LIMIT = 50  # Detailed prints for first N NaN detections
_NAN_SUMMARY_INTERVAL = 100  # Print summary every N calls
_nan_detail_count = 0  # How many detailed NaN lines printed so far
_nan_check_call_counter = 0  # Total MoE layer calls
_nan_total_pre = 0  # Cumulative pre-dispatch NaN detections
_nan_total_post = 0  # Cumulative post-dispatch NaN detections
_nan_first_pre_call = None  # call_id of first pre-dispatch NaN
_nan_first_post_call = None  # call_id of first post-dispatch NaN
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    # Defer input quantization to the MoE kernel.
    if defer_input_quant:
        a1q = a1
        a1q_scale = None
    else:
        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )

        # NOTE: swizzling pads the scales to multiple of 128
        # which makes the scales tensor different shape than
        # the hidden states, breaking the A2A kernel. So, we
        # delay the swizzling until after the A2A.
        a1q, a1q_scale = a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_fp4_scale_swizzled=False,
            mx_alignment=quant_config.mx_alignment,
        )

    # Skip gathering scales if we have static quantization
    # (the scale is a scalar, replicated on all ranks) or
    # if quantization is deferred.
    skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
    scales = None if skip_gather_scales else [a1q_scale]

    return a1q, scales


def _unwrap_scale_and_prepare_for_moe(
    scales: list[torch.Tensor] | None,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    assert scales is not None and len(scales) == 1
    a1q_scale = scales[0]
    # Apply swizzling after a2a if the MoE kernel needs it.
    if quant_config.quant_dtype == "nvfp4" and quant_config.is_nvfp4_scale_swizzled:
        assert a1q_scale is not None
        if a1q_scale.element_size() == 1:
            a1q_scale = a1q_scale.view(torch.uint8)
        a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

    return a1q_scale


class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Quantize and Dispatch Topk Weights and Topk Ids."""

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            # Note: do not use inplace for shared experts overlap
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales = _quantize_and_setup_dispatch(a1, quant_config, defer_input_quant)

        # --- NaN detection BEFORE dispatch (Modular path) ---
        global _nan_detail_count, _nan_check_call_counter
        global _nan_total_pre, _nan_total_post
        global _nan_first_pre_call, _nan_first_post_call
        _nan_check_call_counter += 1
        call_id = _nan_check_call_counter
        dp_rank = get_dp_group().rank_in_group
        a1q_has_nan = bool(torch.isnan(a1q).any().item())
        a1q_has_inf = bool(torch.isinf(a1q).any().item())
        if a1q_has_nan or a1q_has_inf:
            _nan_total_pre += 1
            if _nan_first_pre_call is None:
                _nan_first_pre_call = call_id
            if _nan_detail_count < _NAN_DETAIL_LIMIT:
                nan_count = int(torch.isnan(a1q).sum().item())
                inf_count = int(torch.isinf(a1q).sum().item())
                nan_rows = torch.isnan(a1q).any(dim=-1)
                nan_row_indices = torch.where(nan_rows)[0].tolist()
                print(
                    f"[NAN_CHECK_PRE_DISPATCH] call={call_id} "
                    f"dp_rank={dp_rank} "
                    f"NaN/Inf in hidden_states BEFORE dispatch "
                    f"(Modular path)! "
                    f"nan_count={nan_count} inf_count={inf_count} "
                    f"shape={list(a1q.shape)} "
                    f"nan_row_indices={nan_row_indices[:10]}"
                    f"{'... ' if len(nan_row_indices) > 10 else ' '}"
                    f"total_nan_rows={len(nan_row_indices)}",
                    flush=True,
                )
                _nan_detail_count += 1
        # Print summary every _NAN_SUMMARY_INTERVAL calls
        if call_id % _NAN_SUMMARY_INTERVAL == 0:
            print(
                f"[NAN_SUMMARY] call={call_id} dp_rank={dp_rank} "
                f"pre_nan_total={_nan_total_pre} "
                f"post_nan_total={_nan_total_post} "
                f"first_pre_nan_call={_nan_first_pre_call} "
                f"first_post_nan_call={_nan_first_post_call}",
                flush=True,
            )
        # --- End NaN detection BEFORE dispatch ---

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            a1q_scale = None
        else:
            assert len(res) == 4
            a1q, topk_weights, topk_ids, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        # --- NaN detection AFTER dispatch (Modular path) ---
        a1q_has_nan = bool(torch.isnan(a1q).any().item())
        a1q_has_inf = bool(torch.isinf(a1q).any().item())
        if a1q_has_nan or a1q_has_inf:
            _nan_total_post += 1
            if _nan_first_post_call is None:
                _nan_first_post_call = call_id
            if _nan_detail_count < _NAN_DETAIL_LIMIT:
                nan_count = int(torch.isnan(a1q).sum().item())
                inf_count = int(torch.isinf(a1q).sum().item())
                nan_rows = torch.isnan(a1q).any(dim=-1)
                nan_row_indices = torch.where(nan_rows)[0].tolist()
                print(
                    f"[NAN_CHECK_DISPATCH] call={call_id} "
                    f"dp_rank={dp_rank} "
                    f"NaN/Inf detected AFTER dispatch "
                    f"(Modular path)! "
                    f"nan_count={nan_count} inf_count={inf_count} "
                    f"shape={list(a1q.shape)} "
                    f"nan_row_indices={nan_row_indices[:10]}"
                    f"{'... ' if len(nan_row_indices) > 10 else ' '}"
                    f"total_nan_rows={len(nan_row_indices)}",
                    flush=True,
                )
                _nan_detail_count += 1
        # --- End NaN detection AFTER dispatch ---

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        output.copy_(
            get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )


class MoEPrepareAndFinalizeNaiveDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the router logits (the MoE kernel runs the router internally).
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        """Quantize and Dispatch Router Logits."""

        a1q, scales = _quantize_and_setup_dispatch(a1, quant_config, defer_input_quant)

        # --- NaN detection BEFORE dispatch (Monolithic path) ---
        global _nan_detail_count, _nan_check_call_counter
        global _nan_total_pre, _nan_total_post
        global _nan_first_pre_call, _nan_first_post_call
        _nan_check_call_counter += 1
        call_id = _nan_check_call_counter
        dp_rank = get_dp_group().rank_in_group
        a1q_has_nan = bool(torch.isnan(a1q).any().item())
        a1q_has_inf = bool(torch.isinf(a1q).any().item())
        rl_has_nan = bool(torch.isnan(router_logits).any().item())
        rl_has_inf = bool(torch.isinf(router_logits).any().item())
        if a1q_has_nan or a1q_has_inf or rl_has_nan or rl_has_inf:
            _nan_total_pre += 1
            if _nan_first_pre_call is None:
                _nan_first_pre_call = call_id
            if _nan_detail_count < _NAN_DETAIL_LIMIT:
                hs_nan = int(torch.isnan(a1q).sum().item())
                hs_inf = int(torch.isinf(a1q).sum().item())
                rl_nan = int(torch.isnan(router_logits).sum().item())
                rl_inf = int(torch.isinf(router_logits).sum().item())
                print(
                    f"[NAN_CHECK_PRE_DISPATCH] call={call_id} "
                    f"dp_rank={dp_rank} "
                    f"NaN/Inf BEFORE dispatch (Monolithic path)! "
                    f"hidden_states: nan={hs_nan} inf={hs_inf} "
                    f"shape={list(a1q.shape)} | "
                    f"router_logits: nan={rl_nan} inf={rl_inf} "
                    f"shape={list(router_logits.shape)}",
                    flush=True,
                )
                _nan_detail_count += 1
        # Print summary every _NAN_SUMMARY_INTERVAL calls
        if call_id % _NAN_SUMMARY_INTERVAL == 0:
            print(
                f"[NAN_SUMMARY] call={call_id} dp_rank={dp_rank} "
                f"pre_nan_total={_nan_total_pre} "
                f"post_nan_total={_nan_total_post} "
                f"first_pre_nan_call={_nan_first_pre_call} "
                f"first_post_nan_call={_nan_first_post_call}",
                flush=True,
            )
        # --- End NaN detection BEFORE dispatch ---

        res = get_ep_group().dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 2
            a1q, router_logits = res
            a1q_scale = None
        else:
            assert len(res) == 3
            a1q, router_logits, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        # --- NaN detection AFTER dispatch (Monolithic path) ---
        a1q_has_nan = bool(torch.isnan(a1q).any().item())
        a1q_has_inf = bool(torch.isinf(a1q).any().item())
        if a1q_has_nan or a1q_has_inf:
            _nan_total_post += 1
            if _nan_first_post_call is None:
                _nan_first_post_call = call_id
            if _nan_detail_count < _NAN_DETAIL_LIMIT:
                nan_count = int(torch.isnan(a1q).sum().item())
                inf_count = int(torch.isinf(a1q).sum().item())
                nan_rows = torch.isnan(a1q).any(dim=-1)
                nan_row_indices = torch.where(nan_rows)[0].tolist()
                print(
                    f"[NAN_CHECK_DISPATCH] call={call_id} "
                    f"dp_rank={dp_rank} "
                    f"NaN/Inf detected AFTER dispatch "
                    f"(Monolithic path)! "
                    f"nan_count={nan_count} inf_count={inf_count} "
                    f"shape={list(a1q.shape)} "
                    f"nan_row_indices={nan_row_indices[:10]}"
                    f"{'... ' if len(nan_row_indices) > 10 else ' '}"
                    f"total_nan_rows={len(nan_row_indices)}",
                    flush=True,
                )
                _nan_detail_count += 1
        # --- End NaN detection AFTER dispatch ---

        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        out = get_ep_group().combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )
        return out


def make_moe_prepare_and_finalize_naive_dp_ep(
    use_monolithic: bool,
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular | MoEPrepareAndFinalizeNaiveDPEPMonolithic:
    return (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        if use_monolithic
        else MoEPrepareAndFinalizeNaiveDPEPModular(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
    )
