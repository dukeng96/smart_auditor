from __future__ import annotations

from typing import Any, Dict, List

import httpx

from settings import AppSettings


class KnowledgeBaseClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.base_url = settings.kb_api.base_url.rstrip("/")

    async def list_folders(self) -> Dict[str, Any]:
        payload = {"bot_id": self.settings.kb_api.bot_id}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{self.base_url}/list_folders", json=payload)
            response.raise_for_status()
            return response.json()

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
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/search", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("knowledge_retrieval", [])
