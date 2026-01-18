from __future__ import annotations
from typing import Dict, Any, List

import pandas as pd

from ..state import AgentState
from ...tools.correlation import compute_pearson_correlation
from ...tools.public_view import make_public_tool_result


def _merge_tool_result(state: AgentState, updates: Dict[str, Any]) -> Dict[str, Any]:
    base = state.get("tool_result", {}) or {}
    merged = dict(base)
    merged.update(updates)
    return merged


def _plan_tools_from_state(state: AgentState) -> List[str]:
    """
    Return a list of machine-readable tool tags requested by the planner.

    Priority:
    1) If planner already provided `plan_tools` (list[str]), use it.
    2) Otherwise, try to read from `plan` lines and extract tags like [TOOL:correlation].
       (This fallback keeps backward compatibility if planner wasn't updated.)
    """
    # prefer explicit plan_tools
    plan_tools = state.get("plan_tools")
    if isinstance(plan_tools, list) and plan_tools:
        return [str(t).lower() for t in plan_tools]

    # fallback: try to parse from plan lines (simple heuristic)
    plan = state.get("plan") or []
    tags = []
    if isinstance(plan, list):
        for line in plan:
            line_low = (line or "").lower()
            # look for common tokens; a robust parser could be used but keep minimal
            if "[tool:correlation]" in line_low:
                tags.append("correlation")
            if "[tool:baseline_model]" in line_low:
                tags.append("baseline_model")
            if "[tool:plot]" in line_low:
                tags.append("plot")
    return tags


def analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Analysis orchestration node (gating-enabled):
    - Use planner's machine-readable `plan_tools` for deterministic gating.
    - Fallback to simple rule-based decisions if plan_tools is absent.
    - Run selected analysis tools (first version: correlation).
    - Merge results back into tool_result and store a stable public view.
    """
    df = state.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        merged = _merge_tool_result(
            state,
            {"error": {"message": "analysis_node: missing df in state", "payload": {}}},
        )
        merged["public_tool_result"] = make_public_tool_result(merged)
        return {"tool_result": merged}

    # determine selected target (planner should set this)
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
    schema_summary = tool_result.get("schema_summary", {}) or {}

    # -------------------------
    # GATING: use planner tags first, fallback to rules
    # -------------------------
    plan_tools = _plan_tools_from_state(state)
    # rule-based fallback signals
    numeric_columns = schema_summary.get("numeric_columns") or []
    n_numeric = len(numeric_columns)

    # Decide whether to run correlation:
    # - If planner explicitly requested correlation via tags => run
    # - Else fallback: run correlation when we have a target and >=2 numeric features
    should_run_correlation = False
    gating_reasons: Dict[str, Any] = {}
    if "correlation" in plan_tools:
        should_run_correlation = True
        gating_reasons["why"] = "planner_requested"
    else:
        # fallback rule: correlation is cheap and generally useful when numeric features exist
        if target and n_numeric >= 2:
            should_run_correlation = True
            gating_reasons["why"] = "fallback_rule_numeric_enough"
        else:
            gating_reasons["why"] = "no_numeric_or_no_target"

    # record gating decisions for debug / transparency
    gating_summary = {
        "plan_tools": plan_tools,
        "n_numeric": n_numeric,
        "should_run_correlation": should_run_correlation,
        "gating_reasons": gating_reasons,
    }

    updates: Dict[str, Any] = {}
    # -------------------------
    # Execute selected analyses
    # -------------------------
    if should_run_correlation:
        corr_payload = compute_pearson_correlation(df, target=target, top_k=8)
        updates["correlation"] = corr_payload
        updates["analysis_executed"] = {"correlation": True, "task_type_seen": task_type}
    else:
        updates["analysis_executed"] = {"correlation": False, "task_type_seen": task_type}

    # store gating summary for observability
    updates["analysis_gating"] = gating_summary

    # Merge and store public view inside tool_result (stable across graph merges)
    merged = _merge_tool_result(state, updates)
    merged["public_tool_result"] = make_public_tool_result(merged)

    return {"tool_result": merged}
