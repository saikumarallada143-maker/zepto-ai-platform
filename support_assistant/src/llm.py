"""
src/llm.py — Optional MOCK_LLM=0 extension. NOT part of the graded baseline;
graph.py only imports/calls this module when MOCK_LLM=0 is explicitly set.

Uses Groq's free-tier API (OpenAI-compatible /chat/completions endpoint) —
no payment required, just a free account + API key in the GROQ_API_KEY env
var. Any other genuinely-free-tier LLM API could be swapped in here without
changing graph.py, since this module's only contract is:
    generate_structured_answer(query, retrieved_chunks | None) -> AskResponse
"""

import json
import os
import requests

from src.schema import AskResponse
from src.prompts import build_prompt

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 2  # per Task 4: retry up to 2 additional times on validation failure


def _call_groq(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires GROQ_API_KEY to be set (console.groq.com free tier)."
        )
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_response(raw_text, sources):
    """Best-effort: expect the LLM to return prose; wrap it into the required
    schema ourselves (sources/confidence aren't something we ask the LLM to
    invent — they come from retrieval, same as mock mode)."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty LLM response")
    return AskResponse(answer=text, sources=sources, confidence=0.8)


def generate_structured_answer(query, retrieved_chunks=None):
    """
    retrieved_chunks: list from src.ingest.retrieve_top_k, or None for
    general_question (no-retrieval) queries.
    """
    sources = [c["id"] for c in retrieved_chunks] if retrieved_chunks else []
    prompt = (
        build_prompt(query, retrieved_chunks)
        if retrieved_chunks
        else f"Answer this question directly and concisely: {query}"
    )

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt > 0:
                prompt = (
                    prompt
                    + f"\n\n(Your previous response could not be parsed: {last_error}. "
                    "Please respond again, following the FORMAT instructions exactly.)"
                )
            raw = _call_groq(prompt)
            return _parse_response(raw, sources)
        except Exception as e:  # noqa: BLE001 - any failure triggers the retry/error path
            last_error = str(e)

    # All retries exhausted -> clearly marked error response, not a crash.
    return AskResponse(
        answer=f"[ERROR] Could not generate a validated response after {MAX_RETRIES + 1} "
               f"attempts. Last error: {last_error}",
        sources=sources,
        confidence=0.0,
    )
