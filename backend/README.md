FastAPI backend for the On-Device AI Framework

Run locally (in a virtualenv):

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Endpoints:
- `GET /health` — health check
- `POST /index` — JSON payload `{ "docs": { "id1": "text", ... } }`
- `POST /query` — JSON payload `{ "query": "text", "k": 5 }`

The backend dynamically loads the framework code from `Framework/src` (no packaging required for the demo).
