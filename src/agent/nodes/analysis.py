from __future__ import annotations
from typing import Dict, Any

import pandas as pd

from ..state import AgentState
from ...tools.correlation import compute_pearson_correlation
from ...tools.public_view import make_public_tool_result


def _merge_tool_result(state: AgentState, updates: Dict[str, Any]) -> Dict[str, Any]:
    base = state.get("tool_result", {}) or {}
    merged = dict(base)
    merged.update(updates)
    return merged


def analysis_node(state: AgentState) -> Dict[str, Any]:
    df = state.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        merged = _merge_tool_result(
            state,
            {"error": {"message": "analysis_node: missing df in state", "payload": {}}},
        )
        merged["public_tool_result"] = make_public_tool_result(merged)
        return {"tool_result": merged}

    target = state.get("target")
    if not target:
        ts = state.get("target_selection", {}) or {}
        target = ts.get("selected_target")

    if not target:
        merged = _merge_tool_result(
            state,
            {"error": {"message": "analysis_node: missing selected target", "payload": {}}},
        )
        merged["public_tool_result"] = make_public_tool_result(merged)
        return {"tool_result": merged}

    tool_result = state.get("tool_result", {}) or {}
    task_type = tool_result.get("task_type")

    corr_payload = compute_pearson_correlation(df, target=target, top_k=8)

    updates: Dict[str, Any] = {
        "correlation": corr_payload,
        "analysis_executed": {"correlation": True, "task_type_seen": task_type},
    }

    merged = _merge_tool_result(state, updates)

    # store public view inside tool_result (stable across graph merges)
    merged["public_tool_result"] = make_public_tool_result(merged)

    return {"tool_result": merged}
