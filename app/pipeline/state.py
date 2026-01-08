from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from app.models.schemas import DraftChunk, Finding, ProcessingPhase, ReferenceDoc


class AuditState(TypedDict, total=False):
    request_id: str
    file_path: str
    kb_folder_id: str
    processing_phase: ProcessingPhase
    draft_chunks: List[DraftChunk]
    knowledge_references: Dict[str, ReferenceDoc]
    chunk_references: Dict[str, List[str]]
    chunk_primary_sources: Dict[str, str]
    current_compare_target: str
    findings: List[Finding]
    ocr_total_pages: int
    ocr_processed_pages: int
    search_processed: int
    total_chunks: int
    processed_chunks: int
    processing_time_ms: Optional[int]
