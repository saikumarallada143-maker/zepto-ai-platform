# `/support_assistant` — Zepto Data & AI Platform

A small RAG (retrieval-augmented generation) service that answers customer questions
grounded in Zepto's own policy documents, orchestrated with LangGraph and served via
FastAPI. Fully gradable offline: the default `MOCK_LLM` path makes no LLM API calls
at all — no signup, no key, no network access to any LLM provider required.

## How to run

```bash
cd support_assistant
pip install -r requirements.txt   # includes sentence-transformers -> pulls torch;
                                   # this is a bigger install than the other modules,
                                   # give it a few minutes

uvicorn main:app --host 0.0.0.0 --port 7860
```

First startup builds the ChromaDB index (embeds the 8 docs with `all-MiniLM-L6-v2`,
which needs internet the first time to download the model — cached locally after
that). Subsequent restarts reuse the existing index and need no network at all.

Then, in another terminal:
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d "{\"query\": \"What is your delivery fee?\"}"
```

Or with Docker:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

`MOCK_LLM` defaults to mock mode (`1`) in both paths — that's what's graded and needs
no API key. To try the optional real-LLM extension: get a free key from
console.groq.com, then `export MOCK_LLM=0` and `export GROQ_API_KEY=...` before
running (or `-e MOCK_LLM=0 -e GROQ_API_KEY=...` with `docker run`).

## Architecture (Task 7)

```
docs/doc_01.txt ... doc_08.txt
        |
        v
[INGESTION]  src/ingest.py: load_documents()
        |      one chunk per document (8 chunks total - each doc is already
        |      a single short policy paragraph, see src/ingest.py docstring)
        v
[EMBEDDING]  src/ingest.py: get_embedding_model() -> SentenceTransformer("all-MiniLM-L6-v2")
        |      runs entirely locally, no API key
        v
[STORAGE]    ChromaDB persistent collection "zepto_policies" (./chroma_db),
        |      explicitly configured for cosine similarity (hnsw:space=cosine)
        v
   --- at query time ---
        |
[GRAPH]      src/graph.py: LangGraph StateGraph (GraphState TypedDict)
        |
        +-- node: classify_intent
        |     mock (default): keyword heuristic over the query
        |     MOCK_LLM=0:      LLM classifies instead
        |     -> conditional edge routes to one of:
        |
        +-- node: retrieve_and_answer  (policy_question path)
        |     [RETRIEVAL] src/ingest.py: retrieve_top_k() - embeds the query,
        |         queries ChromaDB for top-3 chunks by cosine similarity.
        |         Runs for REAL in both modes (no API key needed).
        |     [GENERATION] branches on MOCK_LLM:
        |         mock (default): canned f"Based on the retrieved context: {snippet}"
        |         MOCK_LLM=0: src/llm.py calls Groq, prompted with the Task-2
        |                     template from src/prompts.py, grounded only in
        |                     the retrieved chunks
        |
        +-- node: direct_answer  (general_question path, no retrieval)
              [GENERATION] branches on MOCK_LLM:
                  mock (default): fixed canned string, no LLM call
                  MOCK_LLM=0: src/llm.py prompts the LLM directly, no retrieval
        |
        v
[SCHEMA]     src/schema.py: AskResponse (Pydantic) - answer/sources/confidence,
        |      populated deterministically in mock mode; in the MOCK_LLM=0 path,
        |      src/llm.py retries up to 2 more times with a corrective
        |      instruction if the LLM's output fails to parse/validate.
        v
[API]        main.py: FastAPI POST /ask -> validated AskResponse JSON
```

**What changes between mock (default) and MOCK_LLM=0:** only the *generation* step
inside `retrieve_and_answer` and `direct_answer` (and, for the extension,
`classify_intent`'s classification step). Ingestion, embedding, storage, retrieval,
the graph's routing logic, and the response schema are identical in both modes.

## Example calls (MOCK_LLM at default)

> These two transcripts were captured with a **TF-IDF stand-in** for the embedding
> model, not the real `all-MiniLM-L6-v2` — huggingface.co isn't reachable from the
> sandbox this was built in (confirmed via a direct request: `403 host_not_allowed`).
> The pipeline mechanics (routing, ChromaDB storage/retrieval, schema validation,
> the FastAPI response shape) are verified correct by these calls. **Before
> submitting, replace this section with the real transcripts from your own machine**
> — the API's *shape* won't change, but retrieval quality should improve with the
> real semantic embedding model. See the "How this was tested" note below.

**Call 1 — policy question (triggers retrieval):**
```
POST /ask
{"query": "What is your delivery fee for small orders?"}
```
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_05", "doc_08"],
  "confidence": 1.0
}
```

**Call 2 — general question (no retrieval):**
```
POST /ask
{"query": "What's the capital of France?"}
```
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

`classify_intent`'s keyword heuristic was also checked directly against 4 queries
(2 policy-style, 2 general) and routed all 4 correctly, with no LLM call in either
mode.

## Design decisions

**Chunking.** One chunk per document (8 chunks total) — each policy document is
already a single short, self-contained paragraph; splitting further would fragment
one policy statement across chunks for no benefit. Explicitly allowed by the module
spec given the documents' length.

**Cosine similarity.** ChromaDB's default HNSW index metric is L2 (Euclidean), not
cosine — the collection is created with `metadata={"hnsw:space": "cosine"}` to match
the module's explicit "via cosine similarity" requirement.

**Idempotent index build on startup.** `main.py`'s startup event calls
`ensure_index_built()`, which checks `collection.count() >= 8` before re-embedding
anything — so `uvicorn main:app` just works on every restart without a separate
manual indexing step, and doesn't redundantly re-embed on every restart either.

**`sources` field in mock mode.** Populated with the IDs of all 3 retrieved chunks
(not just the one quoted in the canned answer) — this is a closer match to "the ids
of the chunks retrieved for policy_question" than only the top-1, and is more useful
to a caller deciding whether to trust/verify the answer.

## A note on how this was tested

This module was built with AI assistance (permitted per the program's guidelines,
provided the implementation is understood and can be explained). LangGraph, ChromaDB,
FastAPI, and Pydantic schema validation were all installed and exercised for real in
the build environment. The one piece that could not be verified end-to-end there is
the real embedding model itself: `huggingface.co` is not reachable from that sandbox
(confirmed via a direct request returning `403 host_not_allowed`), so
`all-MiniLM-L6-v2`'s weights could not be downloaded there. Every other part of the
pipeline — chunking, ChromaDB storage/query mechanics, the LangGraph routing logic,
the canned mock-mode answer templates, Pydantic schema validation (including
rejecting an out-of-range confidence value), and the FastAPI request/response cycle —
was verified with a local TF-IDF vectorizer standing in for the embedding call only.
**The real model must be run and its retrieval quality re-verified locally** (see
the "Example calls" section above) before submission.

## Git workflow

The overall repository (not just this module) includes a feature branch created,
committed to at least twice, and merged into `main` — see root README /
`git log --graph --all`.
