from __future__ import annotations

import asyncio
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
        self.state_store: Dict[str, AuditState] = {}
        self.workflow = self._build_workflow()

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
        initial_state: AuditState = {
            "request_id": request_id,
            "file_path": str(file_path),
            "kb_folder_id": kb_folder_id,
            "processing_phase": ProcessingPhase.INIT,
            "draft_chunks": [],
            "knowledge_references": {},
            "findings": [],
            "total_chunks": 0,
            "processed_chunks": 0,
        }
        self.state_store[request_id] = initial_state
        asyncio.create_task(self.workflow.ainvoke(initial_state, thread_id=request_id))
        return request_id

    def _persist_state(self, state: AuditState) -> AuditState:
        request_id = state.get("request_id")
        if request_id:
            self.state_store[request_id] = state
        return state

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
        updated_state["processing_phase"] = ProcessingPhase.SEARCH
        return self._persist_state(updated_state)

    async def _retrieve_references(self, state: AuditState) -> AuditState:
        updated_state = dict(state)
        updated_state["processing_phase"] = ProcessingPhase.SEARCH
        knowledge_references: Dict[str, ReferenceDoc] = dict(updated_state.get("knowledge_references", {}))

        for draft_chunk in updated_state.get("draft_chunks", []):
            await asyncio.sleep(self.settings.processing.search_delay)
            try:
                results = await self.kb_client.search(draft_chunk.content, self.settings.kb_api.top_k_retrieval)
            except Exception:
                results = []
            for idx, item in enumerate(results):
                ref_id = item.get("id") or f"ref_{draft_chunk.id}_{idx}"
                knowledge_references[ref_id] = ReferenceDoc(
                    ref_id=str(ref_id),
                    doc_code=item.get("filed", "N/A"),
                    file_name=item.get("file_name", "unknown.pdf"),
                    title=item.get("passage_title", ""),
                    content=md(item.get("passage_content", "")),
                    page_number=item.get("page_id", 1),
                )

        updated_state["knowledge_references"] = knowledge_references
        updated_state["processing_phase"] = ProcessingPhase.COMPARE
        return self._persist_state(updated_state)

    async def _compare_chunks(self, state: AuditState) -> AuditState:
        updated_state = dict(state)
        updated_state["processing_phase"] = ProcessingPhase.COMPARE
        findings: List[Finding] = list(updated_state.get("findings", []))
        references = list(updated_state.get("knowledge_references", {}).values())

        start_time = time.time()
        for draft_chunk in updated_state.get("draft_chunks", []):
            reference_list_formatted = "\n".join(
                [
                    f"- ID: {ref.ref_id}\n  Tiêu đề: {ref.title}\n  Nội dung: {ref.content}"
                    for ref in references
                ]
            )
            try:
                llm_results = await self.llm_client.compare(
                    draft_text=draft_chunk.content,
                    reference_list_formatted=reference_list_formatted,
                    prompt_template=COMPARE_PROMPT,
                )
            except Exception:
                llm_results = []

            if not llm_results:
                findings.append(
                    Finding(
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
                            draft_chunk_id=draft_chunk.id,
                            type=result_type,
                            related_ref_id=result.get("ref_id"),
                            summary=result.get("summary", ""),
                            risk_score=float(result.get("risk_score", 0.0)),
                        )
                    )
            updated_state["processed_chunks"] = updated_state.get("processed_chunks", 0) + 1
            self._persist_state(updated_state)

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
        stats: Dict[str, int] = {
            AnalysisType.CONFLICT.value.lower(): len([f for f in findings if f.type == AnalysisType.CONFLICT]),
            AnalysisType.UPDATE.value.lower(): len([f for f in findings if f.type == AnalysisType.UPDATE]),
            AnalysisType.DUPLICATE.value.lower(): len([f for f in findings if f.type == AnalysisType.DUPLICATE]),
        }
        risk_level = "Thấp"
        if stats.get("conflict", 0) >= 2:
            risk_level = "Cao"
        elif stats.get("conflict", 0) == 1 or stats.get("update", 0) > 2:
            risk_level = "Trung bình"

        linkages: List[DocLinkage] = []
        ref_by_file: Dict[str, List[str]] = {}
        for finding in findings:
            if not finding.related_ref_id:
                continue
            ref_doc = state.get("knowledge_references", {}).get(finding.related_ref_id)
            if not ref_doc:
                continue
            ref_by_file.setdefault(ref_doc.file_name, []).append(finding.draft_chunk_id)
        for file_name, chunk_ids in ref_by_file.items():
            linkages.append(
                DocLinkage(
                    ref_doc_code="N/A",
                    file_name=file_name,
                    linked_count=len(chunk_ids),
                    related_chunk_ids=chunk_ids,
                )
            )

        overview = OverviewReport(
            risk_level=risk_level,
            risk_reason="Tự động tổng hợp từ kết quả LLM",
            stats=stats,
            exec_summary="Kết quả phân tích tự động từ SmartDoc Auditor",
            doc_linkages=linkages,
        )

        processing_ms = state.get("processing_time_ms") or 0
        return AnalysisResultResponse(
            request_id=request_id,
            processing_time_ms=processing_ms,
            overview=overview,
            draft_content=state.get("draft_chunks", []),
            knowledge_references=state.get("knowledge_references", {}),
            findings=findings,
        )
