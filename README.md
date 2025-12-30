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
Logging controls live under the `logging` section (level + `log_external_io` to print KB/LLM request & response bodies).

## API reference & sample requests

### Upload & trigger processing
- **POST** `/api/v1/auditor/upload`
- **Form fields**:
  - `kb_folder_id`: scope retrieval to a KB folder path (string)
  - `file`: PDF/DOCX binary
- **Sample (curl)**
  ```bash
  curl -X POST "http://localhost:8000/api/v1/auditor/upload" \
       -F "kb_folder_id=/Chính sách gói cước" \
       -F "file=@./sample.pdf"
  ```
- **Response**
  ```json
  {
    "success": true,
    "task_id": "thread_abc123_xyz",
    "message": "File uploaded successfully. Processing started."
  }
  ```

### Stream progress (SSE)
- **GET** `/api/v1/auditor/stream/{task_id}`
- **Headers**: `Accept: text/event-stream`
- **Sample (curl)**
  ```bash
  curl -N "http://localhost:8000/api/v1/auditor/stream/thread_abc123_xyz"
  ```
- **Events** (examples)
  ```text
  event: message
  data: {"phase": "OCR", "progress": 10, "message": "Đang trích xuất nội dung văn bản..."}

  event: message
  data: {"phase": "SEARCH", "progress": 20, "message": "Đang tìm kiếm văn bản liên quan..."}

  event: message
  data: {"phase": "COMPARE", "progress": 30, "message": "Đang đối chiếu với: file_a.pdf"}

  event: message
  data: {"phase": "DONE", "progress": 100, "message": "Hoàn tất!", "result_id": "thread_abc123_xyz"}
  ```

### Poll progress
- **GET** `/api/v1/auditor/progress/{request_id}`
- **Sample**
  ```bash
  curl "http://localhost:8000/api/v1/auditor/progress/thread_abc123_xyz"
  ```
- **Response**
  ```json
  {
    "phase": "COMPARE",
    "total_items": 12,
    "reviewed_items": 6,
    "percentage": 50
  }
  ```

### Get full analysis result
- **GET** `/api/v1/auditor/result/{result_id}`
- **Sample**
  ```bash
  curl "http://localhost:8000/api/v1/auditor/result/thread_abc123_xyz"
  ```
- **Response** (truncated shape)
  ```json
  {
    "request_id": "thread_abc123_xyz",
    "processing_time_ms": 1250,
    "overview": {
      "risk_level": "Cao",
      "risk_reason": "Phát hiện 2 mâu thuẫn nghiêm trọng về chính sách tài chính.",
      "stats": {"conflict": 2, "update": 2, "duplicate": 2, "irrelevant": 1},
      "exec_summary": "AI đã đối chiếu 7 đoạn trong dự thảo với 5 văn bản tham chiếu. Phát hiện 2 mâu thuẫn, 2 cập nhật, 2 trùng khớp và 1 nhiễu",
      "doc_linkages": [
        {
          "file_id": "QĐ-KD-2024",
          "file_name": "Quy_dinh_cu_v1.pdf",
          "links": [
            {"finding_id": "f_001", "type": "UPDATE", "draft_chunk_id": "draft_chunk_01", "risk_score": 0.85},
            {"finding_id": "f_002", "type": "DUPLICATE", "draft_chunk_id": "draft_chunk_02", "risk_score": 0.0}
          ],
          "impacted_draft_chunks": ["draft_chunk_01", "draft_chunk_02"]
        }
      ]
    },
    "draft_content": [
      {"id": "draft_chunk_01", "page_number": 1, "content": "..."}
    ],
    "knowledge_references": {
      "ref_db_1023": {"doc_code": "QĐ-KD-2024", "file_name": "Quy_dinh_cu_v1.pdf", "title": "Điều 1", "content": "..."}
    },
    "findings": [
      {
        "finding_id": "f_001",
        "draft_chunk_id": "draft_chunk_01",
        "type": "UPDATE",
        "related_ref_id": "ref_db_1023",
        "summary": "Thay đổi giá (99k -> 109k)",
        "risk_score": 0.3
      }
    ]
  }
  ```

### Knowledge base actions
- **Update chunk** – `POST /api/v1/auditor/actions/kb/update`
  ```bash
  curl -X POST "http://localhost:8000/api/v1/auditor/actions/kb/update" \
       -H "Content-Type: application/json" \
       -d '{
         "request_id": "thread_abc123_xyz",
         "ref_id": "ref_db_5521",
         "new_content": "Nội dung quy định cũ đã được hiệu chỉnh..."
       }'
  ```

- **Delete chunk** – `POST /api/v1/auditor/actions/kb/delete`
  ```bash
  curl -X POST "http://localhost:8000/api/v1/auditor/actions/kb/delete" \
       -H "Content-Type: application/json" \
       -d '{
         "request_id": "thread_abc123_xyz",
         "ref_id": "ref_db_5521"
       }'
  ```
