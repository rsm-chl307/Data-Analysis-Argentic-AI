from __future__ import annotations
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import AgentState


def interpreter_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Turn tool_result into a stakeholder-friendly answer.
    """
    system = (
        "You are a business analyst. Use the tool_result as ground truth. "
        "Write a clear, concise answer in Traditional Chinese. "
        "If a ranked table exists, summarize the top item and key comparisons."
    )
    user = (
        f"Question: {state['question']}\n"
        f"Plan: {state.get('plan')}\n"
        f"Tool result: {state.get('tool_result')}\n"
    )

    final = llm.invoke([("system", system), ("user", user)]).content
    return {"final_answer": final}
