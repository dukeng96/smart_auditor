from __future__ import annotations

import json
import logging
from typing import List, Optional

from openai import AsyncOpenAI

from settings import AppSettings

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích mâu thuẫn (CONFLICT), trùng lặp (DUPLICATE), thay đổi (UPDATE) giữa các văn bản."""


class LLMClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.llm_api.api_key, base_url=settings.llm_api.base_url)
        self.system_prompt = SYSTEM_PROMPT
        self.logger = logging.getLogger(__name__)
        self.verbose_io = bool(getattr(settings, "logging", None) and settings.logging.log_external_io)

    def _safe_json_loads(self, content: str) -> List[dict]:
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    async def compare(
        self,
        draft_text: str,
        reference_list_formatted: str,
        prompt_template: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> List[dict]:
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {
                "role": "user",
                "content": prompt_template.format(
                    draft_text=draft_text,
                    reference_list_formatted=reference_list_formatted,
                ),
            },
        ]
        self.logger.info(
            "LLM compare request",
            extra={
                "model": self.settings.llm_api.model,
                "temperature": self.settings.llm_api.temperature,
                "stream": stream,
            },
        )
        if self.verbose_io:
            self.logger.info("LLM compare payload", extra={"messages": messages})
        response = await self.client.chat.completions.create(
            model=self.settings.llm_api.model,
            messages=messages,
            temperature=self.settings.llm_api.temperature,
            stream=stream,
        )
        if stream:
            collected: List[str] = []
            async for chunk in response:  # type: ignore[operator]
                delta = chunk.choices[0].delta.content or ""
                collected.append(delta)
            content = "".join(collected) or "[]"
        else:
            content = response.choices[0].message.content or "[]"
        if self.verbose_io:
            self.logger.info("LLM compare response", extra={"raw": content})
        return self._safe_json_loads(content)
