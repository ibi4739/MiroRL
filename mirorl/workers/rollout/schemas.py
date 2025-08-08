# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import difflib
import logging
import os
import re
from typing import Dict, List, Optional

import torch
from pydantic import model_validator
from transformers import PreTrainedTokenizer
from verl.tools.schemas import OpenAIFunctionToolCall
from verl.utils.model import compute_position_id_with_mask

from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    BASE_CHAT_HISTORY,
    FinishReasonTypeEnum,
    Message,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MCPAsyncRolloutRequest(AsyncRolloutRequest):
    """The data model for async rollout."""

    use_mcp_tool_call: bool = False

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError(
                "messages is required for AsyncRolloutRequest initialization"
            )
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError(
                "max_prompt_len is required for AsyncRolloutRequest initialization"
            )
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError(
                "tokenizer is required for AsyncRolloutRequest initialization"
            )

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        use_mcp_tool_call = values.get("use_mcp_tool_call", False)

        tools = (
            [tool.model_dump() for tool in tool_schemas]
            if (
                (tool_schemas := values.get("tool_schemas", []))
                and (not use_mcp_tool_call)
            )
            else None
        )
        tokens_without_prompt = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=False, tokenize=True
        )
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(
                messages,
                tools=(
                    [tool.model_dump() for tool in tool_schemas]
                    if (not use_mcp_tool_call)
                    else None
                ),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )
            values["input_ids"], values["attention_mask"] = (
                tokenization_dict_with_prompt["input_ids"],
                tokenization_dict_with_prompt["attention_mask"],
            )
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt.
                # Consider raising an error for this case in the future.
                logger.warning(
                    f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} "
                    f"greater than max_prompt_len {max_prompt_len} after applied chat template with tools."
                )

        values["prompt_ids"], values["prompt_attention_mask"] = (
            values["input_ids"],
            values["attention_mask"],
        )
        values["position_ids"] = values[
            "prompt_position_ids"
        ] = compute_position_id_with_mask(
            torch.tensor(values["attention_mask"])
        ).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(
            values["input_ids"]
        )
        values["generation_prompt_ids"] = values["input_ids"][
            len(tokens_without_prompt) :
        ]
        values["base_conv_wo_gen_prompt_end_pos"] = len(
            tokenizer.apply_chat_template(
                BASE_CHAT_HISTORY,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
        )
        values["base_conv_with_gen_prompt_end_pos"] = len(
            tokenizer.apply_chat_template(
                BASE_CHAT_HISTORY,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        )
        return values

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        use_mcp_tool_call: bool = False,
        keep_think_text_for_last_round_only: bool = False,
        think_block_close_tag: str = "</think>",
    ) -> None:
        if (
            keep_think_text_for_last_round_only and tool_calls is not None
        ):  # assistant message w/ tool calls is not the last round
            content = content.split(think_block_close_tag)[-1].lstrip(
                "\n"
            )  # use the same strip logic of Qwen3 chat template

        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )
        if use_mcp_tool_call:
            last_message = Message(role="assistant", content=content)
        else:
            last_message = self.messages[-1]
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, last_message],
            tools=(
                [tool.model_dump() for tool in self.tool_schemas]
                if (not use_mcp_tool_call and self.tool_schemas)
                else None
            ),
            add_generation_prompt=False,
            tokenize=False,
        )
        content_ids = tokenizer.encode(
            content[self.base_conv_with_gen_prompt_end_pos :], add_special_tokens=False
        )
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(
        self,
        tokenizer: PreTrainedTokenizer,
        contents: list[str],
        use_mcp_tool_call: bool = False,
    ) -> None:
        if not contents:
            return
        if use_mcp_tool_call:
            self.messages.extend(
                [Message(role="user", content=content) for content in contents]
            )
        else:
            self.messages.extend(
                [Message(role="tool", content=content) for content in contents]
            )
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]],
            tools=(
                [tool.model_dump() for tool in self.tool_schemas]
                if (not use_mcp_tool_call and self.tool_schemas)
                else None
            ),
            add_generation_prompt=False,
            tokenize=False,
        )
        content_ids = tokenizer.encode(
            content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False
        )
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def tokenization_sanity_check(
        self,
        tokenizer: PreTrainedTokenizer,
        use_mcp_tool_call: bool = False,
        ignore_think_block: bool = True,
    ):
        # Generate tokens based on current messages
        full_tokens = tokenizer.apply_chat_template(
            [msg.model_dump() for msg in self.messages],
            tools=(
                [tool.model_dump() for tool in self.tool_schemas]
                if (not use_mcp_tool_call and self.tool_schemas)
                else None
            ),
            add_generation_prompt=False,
            tokenize=True,
        )

        # Early return if sequences match
        if self.input_ids == full_tokens:
            return

        # Pattern to identify thinking blocks
        pattern = r"<think>.+?</think>\n\n"
        s = difflib.SequenceMatcher(None, self.input_ids, full_tokens)

        # Iterate over the opcodes generated by difflib.SequenceMatcher to check for differences
        # between input_ids and full_tokens.
        # Each opcode is a tuple: (tag, i1, i2, j1, j2)
        #   - tag: the type of difference ('replace', 'delete', 'insert', 'equal')
        #   - i1, i2: the start and end indices in input_ids for this operation
        #   - j1, j2: the start and end indices in full_tokens for this operation
        # This allows us to precisely locate and analyze mismatches between the two token sequences,
        # and log detailed information for debugging tokenization inconsistencies.
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == "equal":
                continue  # skip equal parts
            elif tag == "delete":
                diff_text = tokenizer.decode(
                    self.input_ids[i1:i2], skip_special_tokens=True
                )
                if re.fullmatch(pattern, diff_text, re.DOTALL):
                    if not ignore_think_block:
                        logger.warning(
                            "Inconsistent tokenization: input_ids contain extra <think>...</think> block not in full_tokens"
                        )
                else:
                    logger.warning(
                        f"Inconsistent tokenization: input_ids contain extra text: `{diff_text}`"
                    )
            elif tag == "insert":
                diff_text = tokenizer.decode(
                    full_tokens[j1:j2], skip_special_tokens=True
                )
                logger.warning(
                    f"Inconsistent tokenization: full_tokens contain extra text: `{diff_text}`"
                )
            elif tag == "replace":
                diff_text_input_ids = tokenizer.decode(
                    self.input_ids[i1:i2], skip_special_tokens=True
                )
                diff_text_full_tokens = tokenizer.decode(
                    full_tokens[j1:j2], skip_special_tokens=True
                )
                logger.warning(
                    f"Inconsistent tokenization: `{diff_text_input_ids}` in input_ids "
                    f"replaced by `{diff_text_full_tokens}` in full_tokens"
                )

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
        use_mcp_tool_call: bool = False,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        if self.enable_tokenization_sanity_check:
            self.tokenization_sanity_check(tokenizer, use_mcp_tool_call)

        # In case we failed to generate the assistant message and the generation prompt ids
        # were already added to input_ids, remove them from the end of input_ids
        if (
            self.input_ids[-len(self.generation_prompt_ids) :]
            == self.generation_prompt_ids
        ):
            self.input_ids = self.input_ids[: -len(self.generation_prompt_ids)]
            self.attention_mask = self.attention_mask[
                : -len(self.generation_prompt_ids)
            ]
            self.position_ids = self.position_ids[: -len(self.generation_prompt_ids)]
            self.loss_mask = self.loss_mask[: -len(self.generation_prompt_ids)]

        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(
                f"Unsupported finalize finish reason type: {finish_reason_type}"
            )
        self.truncate_output_ids(tokenizer)
        assert (
            len(self.input_ids)
            == len(self.attention_mask)
            == len(self.position_ids)
            == len(self.loss_mask)
        ), f"""Request {self.request_id} has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""
