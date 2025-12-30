from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
import time
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from markdownify import markdownify as md

from app.clients.kb_client import KnowledgeBaseClient
from app.clients.llm_client import LLMClient
from app.models.schemas import (
    AnalysisResultResponse,
    AnalysisType,
    DocLinkage,
    DraftChunk,
    Finding,
    LinkageItem,
    OverviewReport,
    ProcessingPhase,
    ReferenceDoc,
)
from app.pipeline.state import AuditState
from settings import AppSettings

COMPARE_PROMPT = (
    "Bạn là chuyên gia phân tích mâu thuẫn (CONFLICT), trùng lặp (DUPLICATE), thay đổi (UPDATE) giữa các văn bản. "
    "So sánh đoạn DỰ THẢO với các đoạn THAM CHIẾU.\n\n"
    "DỰ THẢO:\n\"\"\"{draft_text}\"\"\"\n\n"
    "DANH SÁCH CÁC ĐOẠN THAM CHIẾU:\n{reference_list_formatted} \n\n"
    "YÊU CẦU:\n"
    "Với MỖI đoạn tham chiếu, hãy xác định mối quan hệ với đoạn dự thảo theo các loại sau:\n"
    "1. DUPLICATE: Giống hệt hoặc tương đương hoàn toàn về nghĩa.\n"
    "2. CONFLICT: Mâu thuẫn, trái ngược quy định.\n"
    "3. UPDATE: Dự thảo thay đổi (ngày, tiền, %...) so với tham chiếu.\n"
    "4. IRRELEVANT: Không liên quan.\n\n"
    "HÃY TRẢ VỀ DUY NHẤT MỘT MẢNG JSON GỒM N PHẦN TỬ (Tương ứng với số đoạn tham chiếu):\n"
    "- Nếu type là DUPLICATE, CONFLICT hoặc UPDATE:\n  {{ \"ref_id\": \"ID\", \"type\": \"...\", \"summary\": \"< 20 từ\", \"risk_score\": 0.0-1.0 }}\n"
    "- Nếu type là IRRELEVANT:\n  {{ \"ref_id\": \"ID\", \"type\": \"IRRELEVANT\" }} (Bỏ qua summary và risk_score)"
)


class AuditWorkflow:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.checkpointer = MemorySaver()
        self.kb_client = KnowledgeBaseClient(settings)
        self.llm_client = LLMClient(settings)
        self.folder_cache: Dict[str, str] = {}
        self.state_store: Dict[str, AuditState] = {}
        self.workflow = self._build_workflow()
        self.logger = logging.getLogger(__name__)

    def _build_workflow(self):
        graph = StateGraph(AuditState)
        graph.add_node("ingest", self._ingest_and_chunk)
        graph.add_node("retrieve", self._retrieve_references)
        graph.add_node("compare", self._compare_chunks)
        graph.add_node("complete", self._complete)

        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "retrieve")
        graph.add_edge("retrieve", "compare")
        graph.add_edge("compare", "complete")
        graph.add_edge("complete", END)

        return graph.compile(checkpointer=self.checkpointer)

    async def start(self, file_path: Path, kb_folder_id: str) -> str:
        request_id = f"req_{uuid4().hex}"
        self.logger.info("Starting audit workflow", extra={"request_id": request_id, "kb_folder_id": kb_folder_id})
        initial_state: AuditState = {
            "request_id": request_id,
            "file_path": str(file_path),
            "kb_folder_id": kb_folder_id,
            "processing_phase": ProcessingPhase.INIT,
            "draft_chunks": [],
            "knowledge_references": {},
            "chunk_references": {},
            "chunk_primary_sources": {},
            "findings": [],
            "search_processed": 0,
            "total_chunks": 0,
            "processed_chunks": 0,
        }
        self.state_store[request_id] = initial_state
        asyncio.create_task(
            self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": request_id}},
            )
        )
        return request_id

    def _persist_state(self, state: AuditState) -> AuditState:
        request_id = state.get("request_id")
        if request_id:
            self.state_store[request_id] = state
        return state

    def _normalize_ref_id(self, candidate: str) -> str:
        normalized = unicodedata.normalize("NFKD", candidate)
        ascii_text = normalized.encode("ascii", "ignore").decode().lower()
        sanitized = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
        compact = re.sub(r"-+", "-", sanitized)
        return compact or f"ref-{uuid4().hex[:8]}"

    async def _resolve_kb_folder_path(self, kb_folder_id: str) -> str | None:
        if not kb_folder_id:
            return None
        if kb_folder_id in self.folder_cache:
            return self.folder_cache[kb_folder_id]
        try:
            folders = await self.kb_client.list_folders()
        except Exception:
            return None
        for item in folders.get("folders_in_details", []):
            if item.get("id") == kb_folder_id or item.get("path") == kb_folder_id:
                folder_path = item.get("path")
                if folder_path:
                    self.folder_cache[kb_folder_id] = folder_path
                return folder_path
        return None

    async def _ingest_and_chunk(self, state: AuditState) -> AuditState:
        updated_state = dict(state)
        updated_state["processing_phase"] = ProcessingPhase.OCR
        await asyncio.sleep(self.settings.processing.ocr_delay)

        file_path = Path(updated_state["file_path"])
        text_content: List[str] = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_content.append(page_text)
        full_text = "\n".join(text_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.processing.chunk_size,
            chunk_overlap=self.settings.processing.chunk_overlap,
        )
        chunks = splitter.split_text(full_text)
        draft_chunks = [
            DraftChunk(id=f"draft_chunk_{idx:04d}", content=chunk, page_number=idx + 1)
            for idx, chunk in enumerate(chunks)
        ]

        updated_state["draft_chunks"] = draft_chunks
        updated_state["total_chunks"] = len(draft_chunks)
        updated_state["search_processed"] = 0
        updated_state["processing_phase"] = ProcessingPhase.SEARCH
        self.logger.info(
            "Ingestion completed",
            extra={"request_id": updated_state.get("request_id"), "total_chunks": len(draft_chunks)},
        )
        return self._persist_state(updated_state)

    async def _retrieve_references(self, state: AuditState) -> AuditState:
        updated_state = dict(state)
        updated_state["processing_phase"] = ProcessingPhase.SEARCH
        knowledge_references: Dict[str, ReferenceDoc] = dict(updated_state.get("knowledge_references", {}))
        chunk_references: Dict[str, List[str]] = dict(updated_state.get("chunk_references", {}))
        chunk_primary_sources: Dict[str, str] = dict(updated_state.get("chunk_primary_sources", {}))

        folder_path = await self._resolve_kb_folder_path(updated_state.get("kb_folder_id", ""))
        top_n_retrieval = getattr(self.settings.kb_api, "top_n_retrieval", self.settings.kb_api.top_k_retrieval)
        top_n_reranking = getattr(self.settings.kb_api, "top_n_reranking", self.settings.kb_api.top_k_retrieval)

        self.logger.info(
            "Retrieval phase started",
            extra={
                "request_id": updated_state.get("request_id"),
                "folder_path": folder_path or "all",
                "top_n_retrieval": top_n_retrieval,
                "top_n_reranking": top_n_reranking,
            },
        )

        for draft_chunk in updated_state.get("draft_chunks", []):
            await asyncio.sleep(self.settings.processing.search_delay)
            try:
                results = await self.kb_client.search(
                    query=draft_chunk.content,
                    kb_folder_path=folder_path,
                    top_n_retrieval=top_n_retrieval,
                    top_n_reranking=top_n_reranking,
                )
            except Exception:
                results = []
                self.logger.exception(
                    "Retrieval failed for chunk",
                    extra={"request_id": updated_state.get("request_id"), "chunk_id": draft_chunk.id},
                )
            ref_ids_for_chunk: List[str] = []
            for idx, item in enumerate(results):
                raw_id = str(item.get("id") or f"{item.get('file_name', 'ref')}_{item.get('page_id', idx)}_{idx}")
                normalized_id = self._normalize_ref_id(raw_id)
                final_ref_id = normalized_id
                if final_ref_id in knowledge_references:
                    final_ref_id = self._normalize_ref_id(f"{normalized_id}_{draft_chunk.id}_{idx}")
                ref_ids_for_chunk.append(final_ref_id)
                knowledge_references.setdefault(
                    final_ref_id,
                    ReferenceDoc(
                        ref_id=str(final_ref_id),
                        doc_code=item.get("filed", ""),
                        file_name=item.get("file_name", "unknown.pdf"),
                        title=item.get("passage_title", ""),
                        content=md(item.get("passage_content", "")),
                        page_number=item.get("page_id", 1),
                    ),
                )
            if results:
                primary_file = results[0].get("file_name", "unknown.pdf")
                chunk_primary_sources[draft_chunk.id] = primary_file
            chunk_references[draft_chunk.id] = ref_ids_for_chunk[:top_n_reranking]
            updated_state["search_processed"] = updated_state.get("search_processed", 0) + 1
            self._persist_state(updated_state)

        self.logger.info(
            "Retrieval completed",
            extra={
                "request_id": updated_state.get("request_id"),
                "total_references": len(knowledge_references),
            },
        )

        updated_state["knowledge_references"] = knowledge_references
        updated_state["chunk_references"] = chunk_references
        updated_state["chunk_primary_sources"] = chunk_primary_sources
        updated_state["processing_phase"] = ProcessingPhase.COMPARE
        return self._persist_state(updated_state)

    async def _compare_chunks(self, state: AuditState) -> AuditState:
        updated_state = dict(state)
        updated_state["processing_phase"] = ProcessingPhase.COMPARE
        findings: List[Finding] = list(updated_state.get("findings", []))
        references_map = updated_state.get("knowledge_references", {})
        chunk_references = updated_state.get("chunk_references", {})

        start_time = time.time()
        self.logger.info(
            "Comparison phase started",
            extra={"request_id": updated_state.get("request_id"), "total_chunks": len(updated_state.get("draft_chunks", []))},
        )
        for draft_chunk in updated_state.get("draft_chunks", []):
            reference_ids = chunk_references.get(draft_chunk.id, [])
            references_for_chunk = [references_map[ref_id] for ref_id in reference_ids if ref_id in references_map]
            reference_list_formatted = "\n".join(
                [
                    f"- ID: {ref.ref_id}\n  Tiêu đề: {ref.title}\n  Nội dung: {ref.content}"
                    for ref in references_for_chunk
                ]
            )
            updated_state["current_compare_target"] = updated_state.get("chunk_primary_sources", {}).get(
                draft_chunk.id, "Đang đối chiếu các tham chiếu"
            )
            self._persist_state(updated_state)
            try:
                llm_results = await self.llm_client.compare(
                    draft_text=draft_chunk.content,
                    reference_list_formatted=reference_list_formatted,
                    prompt_template=COMPARE_PROMPT,
                )
            except Exception:
                llm_results = []
                self.logger.exception(
                    "LLM compare failed",
                    extra={"request_id": updated_state.get("request_id"), "chunk_id": draft_chunk.id},
                )

            if not llm_results:
                findings.append(
                    Finding(
                        finding_id=f"find_{uuid4().hex}",
                        draft_chunk_id=draft_chunk.id,
                        type=AnalysisType.IRRELEVANT,
                        related_ref_id=None,
                        summary="Không thu được kết quả so sánh",
                        risk_score=0.0,
                    )
                )
            else:
                for result in llm_results:
                    result_type = result.get("type", AnalysisType.IRRELEVANT)
                    if isinstance(result_type, str):
                        result_type = AnalysisType(result_type)
                    findings.append(
                        Finding(
                            finding_id=f"find_{uuid4().hex}",
                            draft_chunk_id=draft_chunk.id,
                            type=result_type,
                            related_ref_id=result.get("ref_id"),
                            summary=result.get("summary", ""),
                            risk_score=float(result.get("risk_score", 0.0)),
                        )
                    )
            updated_state["processed_chunks"] = updated_state.get("processed_chunks", 0) + 1
            self._persist_state(updated_state)

        self.logger.info(
            "Comparison phase completed",
            extra={
                "request_id": updated_state.get("request_id"),
                "processed_chunks": updated_state.get("processed_chunks", 0),
            },
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        updated_state["findings"] = findings
        updated_state["processing_time_ms"] = elapsed_ms
        return self._persist_state(updated_state)

    async def _complete(self, state: AuditState) -> AuditState:
        updated_state = dict(state)
        updated_state["processing_phase"] = ProcessingPhase.DONE
        return self._persist_state(updated_state)

    def get_state(self, request_id: str) -> AuditState | None:
        return self.state_store.get(request_id)

    def build_result(self, request_id: str) -> AnalysisResultResponse | None:
        state = self.get_state(request_id)
        if not state:
            return None
        findings: List[Finding] = state.get("findings", [])
        filtered_findings = [f for f in findings if f.type != AnalysisType.IRRELEVANT]
        stats: Dict[str, int] = {
            AnalysisType.CONFLICT.value.lower(): len([f for f in filtered_findings if f.type == AnalysisType.CONFLICT]),
            AnalysisType.UPDATE.value.lower(): len([f for f in filtered_findings if f.type == AnalysisType.UPDATE]),
            AnalysisType.DUPLICATE.value.lower(): len([f for f in filtered_findings if f.type == AnalysisType.DUPLICATE]),
        }
        risk_level = "Thấp"
        if stats.get("conflict", 0) >= 2:
            risk_level = "Cao"
        elif stats.get("conflict", 0) == 1 or stats.get("update", 0) > 2:
            risk_level = "Trung bình"

        risk_reason = "Không phát hiện rủi ro đáng kể."
        sorted_findings = sorted(filtered_findings, key=lambda f: f.risk_score, reverse=True)
        for finding in sorted_findings:
            if finding.summary:
                risk_reason = finding.summary
                break

        linkages: List[DocLinkage] = []
        ref_by_file: Dict[str, Dict[str, object]] = {}
        for finding in filtered_findings:
            if not finding.related_ref_id:
                continue
            ref_doc = state.get("knowledge_references", {}).get(finding.related_ref_id)
            if not ref_doc:
                continue
            doc_code_value = ref_doc.doc_code if ref_doc.doc_code and ref_doc.doc_code != "N/A" else ""
            file_id = doc_code_value or self._normalize_ref_id(ref_doc.file_name)
            bucket = ref_by_file.setdefault(
                file_id,
                {
                    "file_name": ref_doc.file_name,
                    "links": [],
                    "impacted": set(),
                },
            )
            if isinstance(bucket.get("impacted"), set):
                bucket["impacted"].add(finding.draft_chunk_id)
            bucket_links = bucket.get("links") if isinstance(bucket.get("links"), list) else []
            bucket_links.append(
                LinkageItem(
                    finding_id=finding.finding_id,
                    type=finding.type,
                    draft_chunk_id=finding.draft_chunk_id,
                    risk_score=finding.risk_score,
                )
            )
            bucket["links"] = bucket_links
        for file_id, payload in ref_by_file.items():
            impacted = sorted(payload.get("impacted", [])) if isinstance(payload.get("impacted"), set) else []
            links = payload.get("links", []) if isinstance(payload.get("links"), list) else []
            linkages.append(
                DocLinkage(
                    file_id=file_id,
                    file_name=payload.get("file_name", ""),
                    links=links,
                    impacted_draft_chunks=list(dict.fromkeys(impacted)),
                )
            )

        overview = OverviewReport(
            risk_level=risk_level,
            risk_reason=risk_reason,
            stats=stats,
            exec_summary=(
                "AI đã đối chiếu "
                f"{len(state.get('draft_chunks', []))} đoạn trong dự thảo với "
                f"{len(state.get('knowledge_references', {}))} văn bản tham chiếu. "
                f"Phát hiện {stats.get('conflict', 0)} mâu thuẫn, {stats.get('update', 0)} cập nhật, "
                f"{stats.get('duplicate', 0)} trùng khớp"
            ),
            doc_linkages=linkages,
        )

        processing_ms = state.get("processing_time_ms") or 0
        return AnalysisResultResponse(
            request_id=request_id,
            processing_time_ms=processing_ms,
            overview=overview,
            draft_content=state.get("draft_chunks", []),
            knowledge_references=state.get("knowledge_references", {}),
            findings=filtered_findings,
        )
