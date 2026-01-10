from __future__ import annotations
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import AgentState


def planner_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Produce a short analysis plan (2-4 steps).
    Phase 1: keep it simple and stable.
    """
    question = state["question"]

    system = (
        "You are a data analysis planner. "
        "Return a short step-by-step plan (2-4 steps) to answer the user's question using pandas. "
        "Keep steps concise and actionable."
    )
    user = f"Question: {question}"

    msg = llm.invoke([("system", system), ("user", user)]).content

    # Minimal parsing: split lines into bullet-like items
    plan: List[str] = [line.strip("-• ").strip() for line in msg.splitlines() if line.strip()]
    plan = plan[:4] if plan else ["Load the dataset", "Compute relevant summary stats", "Answer the question"]

    return {"plan": plan}
