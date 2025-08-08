# Copyright 2023-2024 SGLang Team
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
from __future__ import annotations

import asyncio
import logging
import os
import time
from copy import deepcopy
from datetime import timedelta
from json import JSONDecodeError
from uuid import uuid4

import sglang.srt.entrypoints.engine
import torch.distributed as dist
from omegaconf import DictConfig
from sglang.srt.openai_api.protocol import Tool
from sglang.srt.utils import (
    get_ip,
    get_open_port,
)

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
)
from verl.utils.net_utils import is_ipv6
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout import AsyncEngine, SGLangRollout, get_tool_call_parser_type
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj
from mirorl.workers.rollout.schemas import MCPAsyncRolloutRequest

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Adapted from verl/workers/rollout/sglang_rollout/sglang_rollout.py
# Updated by MiroMind Team
# 1. Support MCP tool for _initialize_tools function
# 2. resolve the port conflict issue in _init_inference_engine function
class MCPSGLangRollout(SGLangRollout):
    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            tokenizer: The tokenizer instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__(actor_module, config, tokenizer, model_hf_config, port, trust_remote_code, device_mesh, **kwargs)

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        # NOTE(carsonchen): Sleep to avoid port conflicts. We make different ranks within the same node
        #  sleep for a short period to avoid conflicts when multiple ranks detect the same port as available
        #  but conflict occurs during actual port allocation.
        # We will have a better solution once the SGLang PR (https://github.com/sgl-project/sglang/pull/7418) is released.
        dist.barrier()
        time.sleep((dist.get_rank() % 8 ) * 0.1)

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=10017 + rank % 8,
                # NOTE(Chenyang): if you want to debug the SGLang engine output
                # please set the following parameters
                # Otherwise, it will make the engine run too slow
                # log_level="INFO",
                # log_requests=True,
                # log_requests_level=2,
                # max_running_requests=1,
            )
        else:
            self._engine = None

        dist.barrier()

        self.sharding_manager = None
        self.is_sleep = True

    def _initialize_tools(self, config, tokenizer):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        import importlib.util
        import sys

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from omegaconf import OmegaConf

        from verl.tools.schemas import OpenAIFunctionToolSchema

        async def get_mcp_tools_schema(config):
            params = StdioServerParameters(
                command=config.command,
                args=config.args if hasattr(config, "args") else [],
                env={e: os.environ.get(e) for e in config.env} if hasattr(config, "env") else {},
            )
            max_retries = config.get("max_retries", 5)
            delay_between_retries = config.get("delay_between_retries", 1)  # in seconds
            connection_timeout = config.get("connection_timeout", 5)
            tools_schema = []

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=connection_timeout)) as session:
                    for attempt in range(max_retries):
                        try:
                            await session.initialize()

                            # The list_tools operation inherits the timeout from ClientSession
                            response = await session.list_tools()
                            for tool in response.tools:
                                tool_schema_dict = {
                                    "type": "function", 
                                    "function": {
                                        "name": tool.name, 
                                        "description": tool.description, 
                                        "parameters": tool.inputSchema,
                                    },
                                }
                                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                                tools_schema.append(tool_schema)
                            break  # Exit loop if successful
                        except Exception as e:
                            logger.error(f"Attempt {attempt + 1} failed: {e}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(delay_between_retries)  # Wait before retrying
                            else:
                                raise
            return tools_schema

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []

            for tool_config in tools_config.tools:
                cls_name = tool_config.class_name
                module_name, class_name = cls_name.rsplit(".", 1)

                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]

                tool_cls = getattr(module, class_name)

                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                if tool_schema_dict["type"] == "function":
                    tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)

                    tool = tool_cls(
                        config=OmegaConf.to_container(tool_config.config, resolve=True),
                        tool_schema=tool_schema,
                    )
                    tool_list.append(tool)
                elif tool_schema_dict["type"] == "mcp":
                    # Get tools schema from MCP server
                    # Call async function following the way in FSDPSGLangShardingManager
                    loop = asyncio.get_event_loop()
                    tools_schema = loop.run_until_complete(get_mcp_tools_schema(tool_config.config))

                    for tool_schema in tools_schema:
                        tool = tool_cls(
                            config=OmegaConf.to_container(tool_config.config, resolve=True),
                            tool_schema=tool_schema,
                        )
                        tool_list.append(tool)
                else:
                    raise NotImplementedError

            return tool_list

        tools_config_file = config.multi_turn.tool_config_path
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        # configs for multi-turn rollout
        use_mcp_tool_call = self.config.multi_turn.use_mcp_tool_call
        keep_think_text_for_last_round_only = self.config.multi_turn.keep_think_text_for_last_round_only
        think_block_close_tag = self.config.multi_turn.think_block_close_tag

        current_turns = 0
        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    if use_mcp_tool_call:
                        _req.messages[-1].tool_calls = None  # remove tool_calls from message object to avoid input_ids is not consistent with messages
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )

                    _req.add_tool_response_messages(
                        self.tokenizer, 
                        [resp[:self.config.multi_turn.tool_response_cut_off_length] for resp, _, _ in tool_call_results], 
                        use_mcp_tool_call=use_mcp_tool_call,
                    )
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results):
                        _req.update_metrics(metrics, tool_call.function.name)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra token accounts for the EOS token).
                # The +5 tokens is for SGLang's input length limit (as seen in sglang/srt/managers/tp_worker.py: self.max_req_input_len = self.max_req_len - 5).
                if len(_req.get_generation_prompt_ids(self.tokenizer)) + 5 + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break
                output = await self._handle_engine_call(_req, do_sample, is_validate, **kwargs)
                content = output["text"]

                # TODO(yuntao): ideally one should not modify verl for support new function call parser, one should add a new function detector to sglang like 
                # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/qwen25_detector.py#L18
                # but to make things self-contained in verl, we modify the assistant-generated content from <use_mcp_tool></use_mcp_tool> to 
                # <tool_call></tool_call> for support Qwen2.5 function call parser.
                # Transform <use_mcp_tool> format to <tool_call> format using linear string processing
                def transform_tool_calls(text):
                    result = []
                    i = 0
                    while i < len(text):
                        # Find start of tool call
                        start = text.find('<use_mcp_tool>', i)
                        if start == -1:
                            result.append(text[i:])
                            break
                        
                        # Add text before tool call
                        result.append(text[i:start])
                        
                        # Find end of tool call
                        end = text.find('</use_mcp_tool>', start)
                        if end == -1:
                            result.append(text[start:])
                            break
                        end += len('</use_mcp_tool>')
                        
                        # Extract tool call content
                        tool_call_text = text[start:end]
                        
                        # Extract tool_name
                        tool_name_start = tool_call_text.find('<tool_name>') + len('<tool_name>')
                        tool_name_end = tool_call_text.find('</tool_name>')
                        if tool_name_start == -1 or tool_name_end == -1:
                            result.append(tool_call_text)
                            i = end
                            continue
                        tool_name = tool_call_text[tool_name_start:tool_name_end]
                        
                        # Extract arguments
                        args_start = tool_call_text.find('<arguments>\n') + len('<arguments>\n')
                        args_end = tool_call_text.find('\n</arguments>')
                        if args_start == -1 or args_end == -1:
                            result.append(tool_call_text)
                            i = end
                            continue
                        arguments = tool_call_text[args_start:args_end]
                        
                        # Add transformed tool call
                        result.append(f'<tool_call>\n{{"name":"{tool_name}", "arguments":{arguments}}}\n</tool_call>')
                        
                        i = end
                    
                    return ''.join(result)

                if self.config.multi_turn.use_mcp_tool_call:
                    transformed_content = transform_tool_calls(content)
                else:
                    transformed_content = content
                
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(
                        self.tokenizer, 
                        content, 
                        use_mcp_tool_call=use_mcp_tool_call, 
                        keep_think_text_for_last_round_only=keep_think_text_for_last_round_only, 
                        think_block_close_tag=think_block_close_tag,
                    )
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(transformed_content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(transformed_content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            # Drop the tool call if its arguments has decode error
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            _req.add_assistant_message(
                                self.tokenizer, 
                                content, 
                                tool_calls=parsed_tool_calls, 
                                use_mcp_tool_call=use_mcp_tool_call, 
                                keep_think_text_for_last_round_only=keep_think_text_for_last_round_only, 
                                think_block_close_tag=think_block_close_tag,
                            )
                        else:
                            _req.add_assistant_message(
                                self.tokenizer, 
                                content, 
                                use_mcp_tool_call=use_mcp_tool_call, 
                                keep_think_text_for_last_round_only=keep_think_text_for_last_round_only, 
                                think_block_close_tag=think_block_close_tag,
                            )
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(
                            self.tokenizer, 
                            content, 
                            use_mcp_tool_call=use_mcp_tool_call, 
                            keep_think_text_for_last_round_only=keep_think_text_for_last_round_only, 
                            think_block_close_tag=think_block_close_tag,
                        )
                        break

        if current_turns >= self.config.multi_turn.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        _req.finalize(self.tokenizer, tool_reward_scores, finish_reason_type, use_mcp_tool_call=self.config.multi_turn.use_mcp_tool_call)

        return _req

    async def _handle_engine_call(self, _req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, override_n: bool = True, **kwargs) -> dict:
        generation_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
        # Adjust max_new_tokens to ensure it is not greater than max_model_len - 1
        # SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra token accounts for the EOS token).
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        kwargs["max_new_tokens"] = max_new_tokens
        if "n" not in kwargs or (kwargs["n"] > 1 and override_n):  # group size is supported in preprocess
            kwargs["n"] = 1
        # users can customize different sampling_params at different run
        params = deepcopy(self.sampling_params)
        params.update(kwargs)
        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=params,
            return_logprob=False,
        )
        return output

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, "need data.return_raw_chat=True, due to no official way do parse_messages"
        req_list = []
        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for rollout_offset in range(n):
                if self._tool_schemas:
                    _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                    _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
                    _input_ids = None
                    _attention_mask = None
                else:
                    _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                    _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                    _tools_kwargs = {}
                    _tool_schemas = None

                req = MCPAsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=raw_prompt.tolist(),
                    tool_schemas=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=_input_ids,
                    response_ids=[],
                    attention_mask=_attention_mask,
                    response_attention_mask=[],
                    response_position_ids=[],
                    response_loss_mask=[],
                    reward_scores={},
                    max_prompt_len=self.config.prompt_length,
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                    use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                    enable_tokenization_sanity_check=self.config.multi_turn.enable_tokenization_sanity_check,
                    tokenizer=self.tokenizer,
                    use_mcp_tool_call=self.config.multi_turn.use_mcp_tool_call,
                )

                error_message = (
                    f"Request {req.request_id} has mismatched lengths: "
                    f"input_ids={len(req.input_ids)}, "
                    f"attention_mask={len(req.attention_mask)}, "
                    f"position_ids={len(req.position_ids)}, "
                    f"loss_mask={len(req.loss_mask)}"
                )
                assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), error_message

                req_list.append(req)

        return req_list
