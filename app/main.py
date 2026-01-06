from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router as auditor_router
from app.pipeline.workflow import AuditWorkflow
from settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)
app = FastAPI(title="SmartDoc Auditor", version="0.1.0")
app.state.workflow = AuditWorkflow(settings)
app.include_router(auditor_router)

logger.info("SmartDoc Auditor initialized", extra={"log_level": settings.logging.level})


@app.get("/health")
async def health_check():
    return {"status": "ok"}
