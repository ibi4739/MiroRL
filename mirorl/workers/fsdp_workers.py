# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 MiroMind Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import datetime
import logging
import os
import warnings

import torch
import torch.distributed
from omegaconf import DictConfig

from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, is_cuda_available
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


# Adapted from verl/workers/fsdp_workers.py
# Updated by MiroMind Team
# 1. Init sglang rollout with MCPSGLangRollout class
class MCPActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl"
                if is_cuda_available
                else "cpu:gloo,npu:hccl",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(days=1),
            )

        super().__init__(config, role)

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert (
            self.world_size % infer_tp == 0
        ), f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(
                module=self.actor_module_fsdp, config=self.config.rollout
            )
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(
                f"Before building {rollout_name} rollout", logger=logger
            )
            local_path = copy_to_local(
                self.config.model.path, use_shm=self.config.model.get("use_shm", False)
            )
            lora_kwargs = (
                {
                    "lora_kwargs": {
                        "enable_lora": True,
                        "max_loras": 1,
                        "max_lora_rank": self._lora_rank,
                    }
                }
                if self._is_lora
                else {}
            )
            # lora_kwargs = {}
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs,
                )
            elif vllm_mode == "spmd":
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = (
                    vLLMRollout
                    if self.config.rollout.mode == "sync"
                    else vLLMAsyncRollout
                )
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs,
                )
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(
                f"After building {rollout_name} rollout", logger=logger
            )
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name in ["sglang", "sglang_async"]:
            if rollout_name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/\
            # python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import (
                FSDPSGLangShardingManager,
            )

            from mirorl.workers.rollout.sglang_rollout import MCPSGLangRollout

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(
                f"Before building {rollout_name} rollout", logger=logger
            )
            rollout = MCPSGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(
                f"After building {rollout_name} rollout", logger=logger
            )

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(
                f"Rollout name: {self.config.rollout.name} is not supported"
            )

        return rollout, rollout_sharding_manager
