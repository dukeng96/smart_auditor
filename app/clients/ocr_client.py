from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

from settings import AppSettings


class OcrClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    async def _add_file(self, file_path: Path) -> str:
        headers = {
            "Token-id": self.settings.ocr_api.token_id,
            "Token-key": self.settings.ocr_api.token_key,
            "mac-address": "EGOV-DIGDOC-WEB-API",
            "Authorization": self.settings.ocr_api.authorization,
        }
        data = {"title": "Hashing document", "description": "Hashing document"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            with file_path.open("rb") as file_handle:
                files = {"file": (file_path.name, file_handle, "application/pdf")}
                response = await client.post(self.settings.ocr_api.add_file_url, headers=headers, data=data, files=files)
        response.raise_for_status()
        payload = response.json()
        file_hash = payload.get("object", {}).get("hash")
        if not file_hash:
            raise RuntimeError("OCR add_file returned no hash")
        return file_hash

    async def _get_session_id(self, file_hash: str, file_type: str) -> str:
        payload = {
            "file_hash": file_hash,
            "file_type": file_type,
            "token": self.settings.ocr_api.token,
            "client_session": self.settings.ocr_api.client_session,
            "details": True,
            "exporter": "json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.settings.ocr_api.session_id_url, json=payload)
        response.raise_for_status()
        data = response.json()
        session_id = data.get("object", {}).get("session_id")
        if not session_id:
            raise RuntimeError("OCR session_id not found")
        return session_id

    @staticmethod
    def _flatten_html(paragraphs: list[dict]) -> str:
        parts: list[str] = []
        for page in paragraphs:
            for cell in page.get("cells", []):
                parts.append(cell.get("html") or cell.get("text") or "")
        return " ".join(part for part in parts if part)

    async def _poll_result(
        self,
        session_id: str,
        on_progress: Optional[Callable[[int, int], Awaitable[None]]],
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Token-id": self.settings.ocr_api.token_id,
            "Token-key": self.settings.ocr_api.token_key,
            "Authorization": self.settings.ocr_api.authorization,
        }
        timeout_seconds = self.settings.ocr_api.timeout_seconds
        poll_interval = self.settings.ocr_api.poll_interval_seconds

        start = asyncio.get_event_loop().time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed >= timeout_seconds:
                    raise TimeoutError("OCR result polling timed out")

                response = await client.post(
                    self.settings.ocr_api.result_url,
                    json={"session_id": session_id},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                if "errors" in payload:
                    raise RuntimeError("OCR result returned errors")

                obj = payload.get("object", {})
                total_pages = int(obj.get("num_of_pages") or 0)
                processed_pages = int(obj.get("num_of_processed_page") or 0)
                if on_progress:
                    await on_progress(processed_pages, total_pages)

                link = obj.get("link")
                if link:
                    link_response = await client.get(link)
                    link_response.raise_for_status()
                    link_payload = link_response.json()
                    paragraphs = link_payload.get("object", {}).get("paragraphs", [])
                    if paragraphs:
                        return self._flatten_html(paragraphs)

                await asyncio.sleep(poll_interval)

    async def run_ocr(
        self,
        file_path: Path,
        on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> str:
        file_type = file_path.suffix.lstrip(".").lower() or "pdf"
        file_hash = await self._add_file(file_path)
        session_id = await self._get_session_id(file_hash, file_type=file_type)
        return await self._poll_result(session_id, on_progress)
