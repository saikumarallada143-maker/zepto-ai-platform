"""
src/ingest.py — Task 1: load the 8 corpus docs, chunk, embed, store in ChromaDB.

Chunking strategy: one chunk per document. Each of the 8 policy documents is
a single short paragraph (see docs/), well under any reasonable chunk-size
threshold — splitting further would only fragment a single self-contained
policy statement across chunks with no benefit, so a simple per-document
chunk (explicitly sanctioned by the module spec given their length) is used.
Chunk IDs are the document IDs themselves (doc_01 ... doc_08).

Embeddings: sentence-transformers' all-MiniLM-L6-v2, run entirely locally
(no API key, no network call after the model is first downloaded/cached).
Storage: a persistent ChromaDB collection on disk at ./chroma_db, so the
index survives across FastAPI restarts without re-embedding every time.
"""

import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"

_model = None  # lazy-loaded singleton, avoids reloading the model repeatedly


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_documents(docs_dir=DOCS_DIR):
    """Returns [(doc_id, text), ...] for every docs/doc_*.txt file, in order."""
    paths = sorted(glob.glob(os.path.join(docs_dir, "doc_*.txt")))
    docs = []
    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append((doc_id, text))
    return docs


def get_collection(chroma_dir=CHROMA_DIR):
    client = chromadb.PersistentClient(path=chroma_dir)
    # Explicitly cosine similarity (Chroma's HNSW default is L2, not cosine) —
    # the module spec requires retrieval "via cosine similarity".
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def ensure_index_built(docs_dir=DOCS_DIR, chroma_dir=CHROMA_DIR, force=False):
    """
    Idempotent: if the collection already has all 8 chunks, does nothing
    (safe to call on every FastAPI startup). Pass force=True to rebuild.
    """
    collection = get_collection(chroma_dir)

    if not force and collection.count() >= 8:
        return collection

    docs = load_documents(docs_dir)
    model = get_embedding_model()

    ids = [doc_id for doc_id, _ in docs]
    texts = [text for _, text in docs]
    embeddings = model.encode(texts).tolist()

    # Fully rebuild on force, to avoid duplicate/stale entries
    if force:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts)
    return collection


def retrieve_top_k(query, k=3, chroma_dir=CHROMA_DIR):
    """Embed `query` and return the top-k most similar chunks (cosine
    similarity, ChromaDB's default space) as [{"id":, "text":, "distance":}, ...],
    ordered most-similar first."""
    collection = get_collection(chroma_dir)
    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=k)
    hits = []
    for doc_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        hits.append({"id": doc_id, "text": text, "distance": distance})
    return hits


if __name__ == "__main__":
    collection = ensure_index_built(force=True)
    print(f"Indexed {collection.count()} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
