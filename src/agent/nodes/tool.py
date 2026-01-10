from __future__ import annotations
from typing import Dict, Any

from ..state import AgentState
from ...tools.pandas_tool import run_basic_analysis


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute a deterministic pandas analysis.
    Phase 1: This tool is intentionally simple and partially hard-coded.
    """
    csv_path = state["csv_path"]
    question = state["question"]

    result = run_basic_analysis(csv_path=csv_path, question=question)
    return {"tool_result": result}
