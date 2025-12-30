from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    DUPLICATE = "DUPLICATE"
    CONFLICT = "CONFLICT"
    UPDATE = "UPDATE"
    IRRELEVANT = "IRRELEVANT"


class ProcessingPhase(str, Enum):
    INIT = "INIT"
    OCR = "OCR"
    SEARCH = "SEARCH"
    COMPARE = "COMPARE"
    DONE = "DONE"
    ERROR = "ERROR"


class DraftChunk(BaseModel):
    id: str
    content: str
    page_number: int


class ReferenceDoc(BaseModel):
    ref_id: str
    doc_code: str = Field(default="N/A")
    file_name: str
    title: str
    content: str
    page_number: Optional[int] = Field(default=1)


class Finding(BaseModel):
    finding_id: str
    draft_chunk_id: str
    type: AnalysisType
    related_ref_id: Optional[str] = None
    summary: str
    risk_score: float


class LinkageItem(BaseModel):
    finding_id: str
    type: AnalysisType
    draft_chunk_id: str
    risk_score: float


class DocLinkage(BaseModel):
    file_id: str
    file_name: str
    links: List[LinkageItem]
    impacted_draft_chunks: List[str]


class OverviewReport(BaseModel):
    risk_level: str
    risk_reason: str = Field(default="")
    stats: Dict[str, int]
    exec_summary: str = Field(default="")
    doc_linkages: List[DocLinkage]


class AnalysisResultResponse(BaseModel):
    request_id: str
    processing_time_ms: int
    overview: OverviewReport
    draft_content: List[DraftChunk]
    knowledge_references: Dict[str, ReferenceDoc]
    findings: List[Finding]


class ProgressResponse(BaseModel):
    phase: ProcessingPhase
    total_items: int
    reviewed_items: int
    percentage: int


class UpdateKnowledgeRequest(BaseModel):
    request_id: str
    ref_id: str
    new_content: str


class DeleteKnowledgeRequest(BaseModel):
    request_id: str
    ref_id: str
