# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.worker.xpu_model_runner import XPUModelRunner, XPUModelRunnerV2

from .utils import request_memory

logger = init_logger(__name__)


class XPUWorker(Worker):
    """A XPU worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        import sys as _sys
        print(f"[=====VLLM_DEBUG=====] XPUWorker.__init__: ENTERED, "
              f"rank={rank}, local_rank={local_rank}, "
              f"pid={os.getpid()}",
              file=_sys.stderr, flush=True)
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        print(f"[=====VLLM_DEBUG=====] XPUWorker.__init__: super().__init__ done, "
              f"pid={os.getpid()}",
              file=_sys.stderr, flush=True)
        device_config = self.device_config
        assert device_config.device_type == "xpu"
        assert current_platform.is_xpu()

        # Torch profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU", "XPU"],
            )

    def init_device(self):
        import sys as _sys
        print(f"[VLLM_DEBUG] XPUWorker.init_device: starting, "
              f"rank={self.rank}, local_rank={self.local_rank}, "
              f"device_type={self.device_config.device_type}, "
              f"pid={os.getpid()}",
              file=_sys.stderr, flush=True)

        device = self.device_config.device
        if (
            isinstance(device, torch.device)
            and device.type == "xpu"
            and current_platform.is_xpu()
        ):
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                print(f"[VLLM_DEBUG] XPUWorker.init_device: "
                      f"DP adjusted local_rank={self.local_rank}, "
                      f"dp_local_rank={dp_local_rank}, "
                      f"tp_pp_world_size={tp_pp_world_size}, "
                      f"pid={os.getpid()}",
                      file=_sys.stderr, flush=True)
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                    f"device_count={torch.accelerator.device_count()}"
                )

            print(f"[VLLM_DEBUG] XPUWorker.init_device: setting device "
                  f"xpu:{self.local_rank}, pid={os.getpid()}",
                  file=_sys.stderr, flush=True)
            self.device = torch.device(f"xpu:{self.local_rank}")
            torch.accelerator.set_device_index(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            torch.accelerator.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank
            ).total_memory
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv(
            "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        )
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        print(f"[VLLM_DEBUG] XPUWorker.init_device: calling "
              f"init_worker_distributed_environment, "
              f"rank={self.rank}, local_rank={self.local_rank}, "
              f"backend={current_platform.dist_backend}, "
              f"world_size={self.parallel_config.world_size}, "
              f"distributed_init_method={self.distributed_init_method}, "
              f"pid={os.getpid()}",
              file=_sys.stderr, flush=True)
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        print(f"[VLLM_DEBUG] XPUWorker.init_device: "
              f"init_worker_distributed_environment done, "
              f"pid={os.getpid()}",
              file=_sys.stderr, flush=True)

        # global all_reduce needed for overall oneccl warm up
        print(f"[VLLM_DEBUG] XPUWorker.init_device: "
              f"xccl warm up all_reduce, pid={os.getpid()}",
              file=_sys.stderr, flush=True)
        if torch.distributed.is_xccl_available():
            torch.distributed.all_reduce(torch.zeros(1).xpu())
        print(f"[VLLM_DEBUG] XPUWorker.init_device: "
              f"xccl warm up done, pid={os.getpid()}",
              file=_sys.stderr, flush=True)

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        model_runner = XPUModelRunnerV2 if self.use_v2_model_runner else XPUModelRunner
        self.model_runner = model_runner(  # type: ignore
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

        print(f"[VLLM_DEBUG] XPUWorker.init_device: all done, "
              f"rank={self.rank}, local_rank={self.local_rank}, "
              f"pid={os.getpid()}",
              file=_sys.stderr, flush=True)
