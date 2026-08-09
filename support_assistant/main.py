"""
main.py — Task 5: FastAPI wrapper around the LangGraph pipeline.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

POST /ask  {"query": "..."}  ->  {"answer": "...", "sources": [...], "confidence": ...}

MOCK_LLM is left at its default (unset / "1") unless you explicitly export
MOCK_LLM=0 — the default (mock) path is what's graded and requires no API key.
"""

from fastapi import FastAPI

from src.schema import AskRequest, AskResponse
from src.ingest import ensure_index_built
from src.graph import ask

app = FastAPI(title="Zepto Support Assistant")


@app.on_event("startup")
def startup_event():
    # Idempotent: builds the ChromaDB index only if it isn't already populated,
    # so `uvicorn main:app` just works with no separate manual step.
    ensure_index_built()


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    return ask(request.query)


@app.get("/health")
def health():
    return {"status": "ok"}
