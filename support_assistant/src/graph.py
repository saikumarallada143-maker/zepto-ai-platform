"""
src/graph.py — Task 3: LangGraph StateGraph.

TypedDict state, 3 nodes (classify_intent, retrieve_and_answer, direct_answer),
and a conditional edge from classify_intent routing to one of the other two
based on the classification. Every node's *generation* step branches on the
MOCK_LLM env var (mock = required graded baseline; real-LLM = optional
extension) — the routing logic itself does not depend on MOCK_LLM.
"""

import os
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.ingest import retrieve_top_k
from src.schema import AskResponse

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]


def is_mock_mode():
    # Left unset, or explicitly "1" -> mock (the graded baseline).
    # Only "0" turns mock mode off.
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict):
    query: str
    intent: Optional[str]          # "policy_question" | "general_question"
    retrieved_chunks: Optional[List[dict]]
    response: Optional[AskResponse]


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Required graded baseline: keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: ask the LLM to classify instead.
        from src.llm import _call_groq  # local import: only needed in this branch
        raw = _call_groq(
            "Classify this customer query as exactly one word, either "
            "'policy_question' (about Zepto delivery/returns/membership/tracking/"
            "cancellation/gift cards/support hours) or 'general_question' "
            f"(anything else). Query: {query!r}\nAnswer with one word only:"
        )
        intent = "policy_question" if "policy_question" in raw.lower() else "general_question"

    return {**state, "intent": intent}


def route_after_classify(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question path)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real, in both modes - no API key/network needed.
    chunks = retrieve_top_k(query, k=3)

    if is_mock_mode():
        # Required graded baseline: canned templated answer, no LLM call.
        top_snippet = chunks[0]["text"][:200] if chunks else ""
        answer_text = f"Based on the retrieved context: {top_snippet}"
        response = AskResponse(
            answer=answer_text,
            sources=[c["id"] for c in chunks],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: real LLM grounded in retrieved chunks.
        from src.llm import generate_structured_answer
        response = generate_structured_answer(query, retrieved_chunks=chunks)

    return {**state, "retrieved_chunks": chunks, "response": response}


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general_question path)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Required graded baseline: fixed canned string, no LLM call.
        response = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        from src.llm import generate_structured_answer
        response = generate_structured_answer(query, retrieved_chunks=None)

    return {**state, "retrieved_chunks": None, "response": response}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(query: str) -> AskResponse:
    graph = get_graph()
    result = graph.invoke({"query": query, "intent": None, "retrieved_chunks": None, "response": None})
    return result["response"]
