from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from settings import AppSettings


class KnowledgeBaseClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.base_url = settings.kb_api.base_url.rstrip("/")
        self.logger = logging.getLogger(__name__)
        self.verbose_io = bool(getattr(settings, "logging", None) and settings.logging.log_external_io)

    async def list_folders(self) -> Dict[str, Any]:
        payload = {"bot_id": self.settings.kb_api.bot_id}
        if self.verbose_io:
            self.logger.info("KB list_folders request", extra={"payload": payload})
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{self.base_url}/list_folders", json=payload)
            response.raise_for_status()
            data = response.json()
            if self.verbose_io:
                self.logger.info("KB list_folders response", extra={"response": data})
            return data

    async def search(
        self,
        query: str,
        kb_folder_path: str | None,
        top_n_retrieval: int,
        top_n_reranking: int,
        ) -> List[Dict[str, Any]]:
        payload = {
            "bot_id": self.settings.kb_api.bot_id,
            "query": query,
            "top_n_retrieval": str(top_n_retrieval),
            "top_n_reranking": str(top_n_reranking),
            "rank": "0",
            "enable_retrieval_detail_return": True,
        }
        if kb_folder_path:
            payload["selected_paths"] = [f".{kb_folder_path}"]
        self.logger.info(
            "KB search",
            extra={
                "query": query[:100],
                "folder": kb_folder_path or "all",
                "top_n_retrieval": top_n_retrieval,
                "top_n_reranking": top_n_reranking,
            },
        )
        if self.verbose_io:
            self.logger.info("KB search payload", extra={"payload": payload})
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/search", json=payload)
            response.raise_for_status()
            data = response.json()
            if self.verbose_io:
                self.logger.info("KB search response", extra={"response": data})
            return data.get("knowledge_retrieval", [])
