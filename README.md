# SmartDoc Auditor Backend

FastAPI backend implementing the SmartDoc Auditor workflow with LangGraph orchestration, PDF ingestion, knowledge base retrieval, and LLM-powered comparison.

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
3. Upload a document and start the audit via `POST /api/v1/auditor/upload`.

Configuration defaults live in `config.yaml` and can be overridden with environment variables using the `SMARTAUDITOR_` prefix (e.g., `SMARTAUDITOR_LLM_API__API_KEY`).
