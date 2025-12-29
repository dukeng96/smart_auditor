from __future__ import annotations

import json
from typing import Iterable, List

from openai import AsyncOpenAI

from settings import AppSettings

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích mâu thuẫn (CONFLICT), trùng lặp (DUPLICATE), thay đổi (UPDATE) giữa các văn bản."""


class LLMClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.llm_api.api_key, base_url=settings.llm_api.base_url)

    async def compare(self, draft_text: str, reference_list_formatted: str, prompt_template: str) -> List[dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt_template.format(
                    draft_text=draft_text,
                    reference_list_formatted=reference_list_formatted,
                ),
            },
        ]
        response = await self.client.chat.completions.create(
            model=self.settings.llm_api.model,
            messages=messages,
            temperature=self.settings.llm_api.temperature,
            stream=False,
        )
        content = response.choices[0].message.content or "[]"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return []
