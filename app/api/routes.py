from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AnalysisResultResponse,
    DeleteKnowledgeRequest,
    ProcessingPhase,
    ProgressResponse,
    UpdateKnowledgeRequest,
)
from app.pipeline.workflow import AuditWorkflow

router = APIRouter(prefix="/api/v1/auditor", tags=["auditor"])


def get_workflow(request: Request) -> AuditWorkflow:
    return request.app.state.workflow


@router.post("/upload")
async def upload_and_process(
    kb_folder_id: str = Form(""),
    file: UploadFile = File(...),
    workflow: AuditWorkflow = Depends(get_workflow),
):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / file.filename
    content = await file.read()
    saved_path.write_bytes(content)

    task_id = await workflow.start(saved_path, kb_folder_id)
    return {"success": True, "task_id": task_id, "message": "File uploaded successfully. Processing started."}


async def _progress_stream(workflow: AuditWorkflow, task_id: str) -> AsyncGenerator[bytes, None]:
    ocr_weight = 25
    search_weight = 5
    compare_weight = 70
    while True:
        state = workflow.get_state(task_id)
        if not state:
            yield b"event: message\n"
            yield b'data: {"phase": "ERROR", "progress": 0, "message": "Task not found"}\n\n'
            break
        total_chunks = state.get("total_chunks", 0)
        search_processed = state.get("search_processed", 0)
        compare_processed = state.get("processed_chunks", 0)
        phase = state.get("processing_phase")
        phase_value = phase.value if hasattr(phase, "value") else phase

        if phase == ProcessingPhase.INIT:
            progress = 0
            message = "Bắt đầu phân tích"
        elif phase == ProcessingPhase.OCR:
            ocr_total = state.get("ocr_total_pages", 0)
            ocr_processed = state.get("ocr_processed_pages", 0)
            if ocr_total:
                progress = int((min(ocr_processed, ocr_total) / ocr_total) * ocr_weight)
                message = f"OCR processing {ocr_processed}/{ocr_total} pages..."
            else:
                ocr_started_at = state.get("ocr_started_at")
                if ocr_started_at:
                    elapsed = max(0.0, time.time() - ocr_started_at)
                    progress = int(min(elapsed / 5.0, 1.0) * ocr_weight)
                else:
                    progress = 0
                message = "Đang OCR văn bản..."
        elif phase == ProcessingPhase.SEARCH:
            if total_chunks:
                search_progress = int((min(search_processed, total_chunks) / total_chunks) * search_weight)
            else:
                search_progress = 0
            progress = ocr_weight + search_progress
            message = "Đang tìm kiếm văn bản liên quan..."
        elif phase == ProcessingPhase.COMPARE:
            base = ocr_weight + search_weight
            if total_chunks:
                chunk_span = compare_weight / total_chunks
                completed = min(compare_processed, total_chunks)
                progress = base + int(completed * chunk_span)
                if completed < total_chunks:
                    last_tick = state.get("compare_last_progress_at") or state.get("compare_started_at")
                    if last_tick:
                        elapsed = max(0.0, time.time() - last_tick)
                        creep = min(chunk_span * 0.7, elapsed * 0.5)
                        progress = min(
                            progress + int(creep),
                            base + int((completed + 1) * chunk_span) - 1,
                        )
            else:
                progress = base
            target_file = state.get("current_compare_target") or "các tài liệu tham chiếu"
            message = f"Đang đối chiếu với: {target_file}"
        elif phase == ProcessingPhase.DONE:
            progress = 100
            message = "Hoàn tất!"
        else:
            progress = int((completed_operations / total_operations) * 100)
            message = f"Đã xử lý {compare_processed}/{total_chunks} khối"

        payload = {
            "phase": phase_value,
            "progress": progress,
            "message": message,
        }
        if phase == ProcessingPhase.DONE:
            payload["result_id"] = task_id
        yield b"event: message\n"
        yield f"data: {payload}\n\n".encode("utf-8")
        if phase == ProcessingPhase.DONE:
            break
        await asyncio.sleep(1)


@router.get("/stream/{task_id}")
async def stream_progress(task_id: str, workflow: AuditWorkflow = Depends(get_workflow)):
    generator = _progress_stream(workflow, task_id)
    return StreamingResponse(generator, media_type="text/event-stream")


@router.get("/progress/{request_id}", response_model=ProgressResponse)
async def get_progress(request_id: str, workflow: AuditWorkflow = Depends(get_workflow)):
    state = workflow.get_state(request_id)
    if not state:
        raise HTTPException(status_code=404, detail="Request not found")
    phase = state.get("processing_phase")
    if phase == ProcessingPhase.OCR:
        total_items = state.get("ocr_total_pages", 0)
        reviewed_items = state.get("ocr_processed_pages", 0)
        percentage = int((min(reviewed_items, total_items) / total_items) * 100) if total_items else 0
        return ProgressResponse(
            phase=phase,
            total_items=total_items,
            reviewed_items=reviewed_items,
            percentage=percentage,
        )
    total_chunks = state.get("total_chunks", 0)
    search_processed = state.get("search_processed", 0)
    compare_processed = state.get("processed_chunks", 0)
    total_operations = max((total_chunks or 0) * 2, 1)
    completed_operations = min(search_processed + compare_processed, total_operations)
    percentage = int((completed_operations / total_operations) * 100)
    return ProgressResponse(
        phase=phase,
        total_items=total_chunks,
        reviewed_items=compare_processed,
        percentage=percentage,
    )


@router.get("/result/{result_id}", response_model=AnalysisResultResponse)
async def get_result(result_id: str, workflow: AuditWorkflow = Depends(get_workflow)):
    result = workflow.build_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.post("/actions/kb/update")
async def update_kb(payload: UpdateKnowledgeRequest, workflow: AuditWorkflow = Depends(get_workflow)):
    state = workflow.get_state(payload.request_id)
    if not state:
        raise HTTPException(status_code=404, detail="Request not found")
    reference = state.get("knowledge_references", {}).get(payload.ref_id)
    if not reference:
        raise HTTPException(status_code=404, detail="Reference not found")
    reference.content = payload.new_content
    return {"success": True, "message": f"Knowledge chunk {payload.ref_id} updated."}


@router.post("/actions/kb/delete")
async def delete_kb(payload: DeleteKnowledgeRequest, workflow: AuditWorkflow = Depends(get_workflow)):
    state = workflow.get_state(payload.request_id)
    if not state:
        raise HTTPException(status_code=404, detail="Request not found")
    if payload.ref_id in state.get("knowledge_references", {}):
        state["knowledge_references"].pop(payload.ref_id)
    return {
        "success": True,
        "message": f"Knowledge chunk {payload.ref_id} removed from active knowledge base.",
    }
