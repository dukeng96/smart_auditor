from __future__ import annotations

import json
import logging
from typing import List

import httpx

from settings import AppSettings


class LLMClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.api_url = settings.llm_api.base_url.rstrip("/")
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
        stream: bool = False,
    ) -> List[dict]:
        query_text = prompt_template.format(
            draft_text=draft_text,
            reference_list_formatted=reference_list_formatted,
        )

        payload = {"query": query_text}
        self.logger.info(
            "LLM compare request",
            extra={
                "endpoint": self.api_url,
                "stream": stream,
            },
        )
        if self.verbose_io:
            self.logger.info("LLM compare payload", extra={"payload": payload})

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception:
            self.logger.exception("LLM compare call failed")
            raise

        raw_content = data.get("response", "") if isinstance(data, dict) else ""
        if self.verbose_io:
            self.logger.info("LLM compare response", extra={"raw": raw_content, "meta": data})

        return self._safe_json_loads(raw_content)
