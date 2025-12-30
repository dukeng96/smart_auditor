from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as auditor_router
from app.pipeline.workflow import AuditWorkflow
from settings import get_settings

settings = get_settings()
app = FastAPI(title="SmartDoc Auditor", version="0.1.0")
app.state.workflow = AuditWorkflow(settings)
app.include_router(auditor_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
