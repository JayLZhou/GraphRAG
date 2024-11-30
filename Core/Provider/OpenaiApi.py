# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:08
@Author  : alexanderwu
@File    : openai.py
@Modified By: mashenquan, 2023/11/21. Fix bug: ReadTimeout.
@Modified By: mashenquan, 2023/12/1. Fix bug: Unclosed connection caused by openai 0.x.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Union

from openai import APIConnectionError, AsyncOpenAI, AsyncStream
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from Core.Common.LLMConfig import LLMConfig, LLMType
from Core.Common.Constants import USE_CONFIG_TIMEOUT
from Core.Common.Logger import log_llm_stream, logger
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.LLM_Provider_Register import register_provider
from Core.Common.Utils import  log_and_reraise
from Core.Common.CostManager import CostManager
from Core.Utils.Exceptions import handle_exception
from Core.Utils.TokenCounter import (
    count_input_tokens,
    count_output_tokens,
    get_max_completion_tokens,
)


@register_provider(
    [
        LLMType.OPENAI,
        LLMType.FIREWORKS,
        LLMType.OPEN_LLM,
        LLMType.MOONSHOT,
        LLMType.MISTRAL,
        LLMType.YI,
        LLMType.OPENROUTER,
    ]
)
class OpenAILLM(BaseLLM):
    """Check https://platform.openai.com/examples for examples"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_client()
        self.auto_max_tokens = False
        self.cost_manager: Optional[CostManager] = None

    def _init_client(self):
        """https://github.com/openai/openai-python#async-usage"""
        self.model = self.config.model  # Used in _calc_usage & _cons_kwargs
        self.pricing_plan = self.config.pricing_plan or self.model
        kwargs = self._make_client_kwargs()
        self.aclient = AsyncOpenAI(**kwargs)

    def _make_client_kwargs(self) -> dict:
        kwargs = {"api_key": self.config.api_key, "base_url": self.config.base_url}

        # to use proxy, openai v1 needs http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _get_proxy_params(self) -> dict:
        params = {}
        if self.config.proxy:
            params = {"proxies": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None) -> str:
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens), stream=True
        )
        usage = None
        collected_messages = []
        has_finished = False
        async for chunk in response:
            chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
            finish_reason = (
                chunk.choices[0].finish_reason if chunk.choices and hasattr(chunk.choices[0], "finish_reason") else None
            )
            log_llm_stream(chunk_message)
            collected_messages.append(chunk_message)
            chunk_has_usage = hasattr(chunk, "usage") and chunk.usage
            if has_finished:
                # for oneapi, there has a usage chunk after finish_reason not none chunk
                if chunk_has_usage:
                    usage = CompletionUsage(**chunk.usage) if isinstance(chunk.usage, dict) else chunk.usage
            if finish_reason:
                if chunk_has_usage:
                    # Some services have usage as an attribute of the chunk, such as Fireworks
                    if isinstance(chunk.usage, CompletionUsage):
                        usage = chunk.usage
                    else:
                        usage = CompletionUsage(**chunk.usage)
                elif hasattr(chunk.choices[0], "usage"):
                    # The usage of some services is an attribute of chunk.choices[0], such as Moonshot
                    usage = CompletionUsage(**chunk.choices[0].usage)
                has_finished = True

        log_llm_stream("\n")
        full_reply_content = "".join(collected_messages)
        if not usage:
            # Some services do not provide the usage attribute, such as OpenAI or OpenLLM
            usage = self._calc_usage(messages, full_reply_content)

        self._update_costs(usage)
        return full_reply_content

    def _cons_kwargs(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None, **extra_kwargs) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._get_max_tokens(messages),
            # "n": 1,  # Some services do not provide this parameter, such as mistral
            "stop": ["[/INST]", "<<SYS>>"] ,  # default it's None and gpt4-v can't have this one
            "temperature": self.config.temperature,
            "model": self.model,
            "timeout": self.get_timeout(timeout),
        }
        if "o1-" in self.model:
            # compatible to openai o1-series
            kwargs["temperature"] = 1
            kwargs.pop("max_tokens")
        if max_tokens != None:
            kwargs["max_tokens"] = max_tokens
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout))

        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)
        self._update_costs(rsp.usage)
        return rsp

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT, max_tokens = None) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._achat_completion_stream(messages, timeout=timeout, max_tokens = max_tokens)

        rsp = await self._achat_completion(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens)
        return self.get_choice_text(rsp)


 

    def get_choice_text(self, rsp: ChatCompletion) -> str:
        """Required to provide the first text of choice"""
        return rsp.choices[0].message.content if rsp.choices else ""

    def _calc_usage(self, messages: list[dict], rsp: str) -> CompletionUsage:
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if not self.config.calc_usage:
            return usage

        try:
            usage.prompt_tokens = count_input_tokens(messages, self.pricing_plan)
            usage.completion_tokens = count_output_tokens(rsp, self.pricing_plan)
        except Exception as e:
            logger.warning(f"usage calculation failed: {e}")

        return usage

    def _get_max_tokens(self, messages: list[dict]):
        if not self.auto_max_tokens:
            return self.config.max_token
        # FIXME
        # https://community.openai.com/t/why-is-gpt-3-5-turbo-1106-max-tokens-limited-to-4096/494973/3
        return min(get_max_completion_tokens(messages, self.model, self.config.max_token), 4096)

    @handle_exception
    async def amoderation(self, content: Union[str, list[str]]):
        """Moderate content."""
        return await self.aclient.moderations.create(input=content)

    async def atext_to_speech(self, **kwargs):
        """text to speech"""
        return await self.aclient.audio.speech.create(**kwargs)

    async def aspeech_to_text(self, **kwargs):
        """speech to text"""
        return await self.aclient.audio.transcriptions.create(**kwargs)

   
    def get_maxtokens(self) -> int:
       self._cons_kwargs['max_tokens']

    async def openai_embedding(self, text):
        response = await self.aclient.embeddings.create(
            model = model, input = text, encoding_format = "float"
        )
        return np.array([dp.embedding for dp in response.data])
