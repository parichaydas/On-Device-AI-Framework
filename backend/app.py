import os
import sys
import importlib.util
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Dynamically load the framework app from Framework/src to avoid name conflicts
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Framework', 'src'))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

spec = importlib.util.spec_from_file_location("framework_app", os.path.join(SRC, "app.py"))
framework_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(framework_app)
FrameworkApp = framework_app.FrameworkApp

app = FastAPI(title="On-Device AI Framework API")
framework = FrameworkApp()


class IndexPayload(BaseModel):
    docs: Dict[str, str]


class QueryPayload(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/index")
async def index_documents(payload: IndexPayload):
    try:
        framework.index_documents(payload.docs)
        return {"indexed": framework.index.index.count() if hasattr(framework.index, 'count') else len(payload.docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(payload: QueryPayload):
    try:
        results = framework.query(payload.query, k=payload.k)
        # serialize results
        return [{"id": r[0], "score": r[1], "meta": r[2]} for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
