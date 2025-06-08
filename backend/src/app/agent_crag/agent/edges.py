"""Decide routing after classifier and evaluator stages."""

from .state import State


def decide_next_step(state: State) -> str:
    """
    After classification, choose the next node:
    - If question's category is classified as greeting or irrelevant → generator
    - otherwise, we always go to the document FIRST for source of truth → retriever
    """
    if state.get("category") in ("greeting", "irrelevant"):
        return "generator"

    else:
        return "retriever"


def should_search_web(state: State) -> str:
    """
    After grading, decide whether to do a web search:
      1. If classifier flagged needs_web_search → query_rewriter
      2. If no filtered_documents → query_rewriter
      3. If doc_relevance < 0.3 → query_rewriter (most of the documents are not relevant, might need to search the web)
      4. Otherwise → generator
    """
    # Condition 1: Real-time request from classifier
    if state.get("needs_web_search", False):
        return "query_rewriter"

    # Condition 2: No relevant FAQ docs
    if not state.get("filtered_documents"):
        return "query_rewriter"

    # Condition 3: Low doc_relevance, most of the documents are not relevant, might need to search the web
    if state.get("doc_relevance", 0.0) < 0.3:
        return "query_rewriter"

    # Condition 4: Otherwise, go to generator
    return "generator"
