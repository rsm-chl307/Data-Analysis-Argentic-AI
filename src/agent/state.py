from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd


class AgentState(TypedDict, total=False):
    # Inputs
    question: str
    csv_path: str

    # Runtime only
    df: pd.DataFrame

    # Planner output
    plan: List[str]
    target: Optional[str]
    target_selection: Dict[str, Any]
    target_rerank: Dict[str, Any]

    # Tool output
    tool_result: Dict[str, Any]

    # Final output
    final_answer: str

