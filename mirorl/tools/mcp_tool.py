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

import asyncio
import json
import logging
import os
from datetime import timedelta
from typing import Any, Optional, Tuple
from uuid import uuid4

import exceptiongroup

import ray
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

from mirorl.utils.debug.exception_helper import extract_exception_details

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def is_timeout_error(error: Exception) -> bool:
    """Check if an error is a timeout-related error."""
    error_str = str(error)
    return any(
        keyword in error_str
        for keyword in ["ETIMEDOUT", "ECONNRESET", "Timeout", "Timed out"]
    )


class MCPTool(BaseTool):
    """A tool for calling MCP tools.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.max_retries = config.get("max_retries", 5)
        self.delay_between_retries = config.get(
            "delay_between_retries", 1
        )  # in seconds
        self.connection_timeout = config.get("connection_timeout", 5)
        self.execution_timeout = config.get("execution_timeout", 60)

        # Get command, args, and env from config
        self.params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env={e: os.environ.get(e) for e in config["env"]}
            if "env" in config.keys()
            else {},
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        # TODO: Add all create_kwargs to dict
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def _execute(self, parameters: dict[str, Any]):
        response = ""

        async with stdio_client(self.params) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=self.connection_timeout),
            ) as session:
                for attempt in range(self.max_retries):
                    try:
                        await session.initialize()

                        result = await session.call_tool(
                            self.name,
                            arguments=parameters,
                            read_timeout_seconds=timedelta(
                                seconds=self.execution_timeout
                            ),
                        )
                        response = result.content[0].text if result.content else ""
                        if attempt > 0:
                            logger.error(f"Attempt {attempt + 1} success")
                        break  # Exit loop if successful
                    except Exception as e:
                        # The error type is McpError, consistently having an error code of -32603.
                        # To determine if the failed connection is network-related, we check the message.
                        if is_timeout_error(e) and attempt < self.max_retries - 1:
                            logger.error(f"Attempt {attempt + 1} failed: {e}")
                            await asyncio.sleep(self.delay_between_retries)
                        else:
                            response = f"Tool execution failed: {e}"

                            try:
                                mcp_tool_checker = ray.get_actor(
                                    name="mcp_tool_checker", namespace="monitor"
                                )
                                ray.get(
                                    mcp_tool_checker.record_error.remote(
                                        str(e), self.name
                                    )
                                )
                            except Exception as e:
                                print(
                                    f"MCP tool checker not found, skip recording error: {e}"
                                )

                            break  # Exit loop if the error is not timed out

        return response

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        # Call MCP tool with retry logic
        response = ""

        for attempt in range(self.max_retries):
            try:
                response = await self._execute(parameters)
                break
            except Exception as e:
                if isinstance(e, exceptiongroup.ExceptionGroup):
                    details = extract_exception_details(e)
                    logger.error(
                        f"MCPTool.execute attempt {attempt + 1} failed: {e}\n"
                        f"exception group details: {json.dumps(details, indent=4)}"
                    )
                else:
                    logger.error(f"MCPTool.execute attempt {attempt + 1} failed: {e}")

        if attempt == self.max_retries:
            logger.error(
                f"MCPTool.execute failed after {self.max_retries} attempts, returning empty response"
            )

        self._instance_dict[instance_id]["response"] = response

        # NOTE: tool_reward is not used in anywhere
        return response, 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # NOTE: tool_reward is not used in anywhere
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
