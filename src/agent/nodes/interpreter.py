from __future__ import annotations
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import AgentState


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _compact_tool_result(state: AgentState) -> Dict[str, Any]:
    """
    Fallback compact summary of internal tool_result.
    Used only when public_tool_result is not available.
    """
    tool_result = state.get("tool_result", {}) or {}

    dataset_meta = tool_result.get("dataset_meta") or {}
    schema_summary = tool_result.get("schema_summary") or {}
    target_candidates = tool_result.get("target_candidates") or {}

    compact: Dict[str, Any] = {}

    compact["dataset_overview"] = {
        "path": dataset_meta.get("path"),
        "n_rows": dataset_meta.get("n_rows"),
        "n_cols": dataset_meta.get("n_cols"),
        "columns": dataset_meta.get("columns"),
    }

    compact["target"] = state.get("target") or _safe_get(state, ["target_selection", "selected_target"])
    compact["task_type"] = tool_result.get("task_type")
    compact["target_selection"] = state.get("target_selection")

    compact["schema_brief"] = {
        "numeric_columns": schema_summary.get("numeric_columns"),
        "categorical_candidates": schema_summary.get("categorical_candidates"),
        "id_like_columns": schema_summary.get("id_like_columns"),
        "n_rows": schema_summary.get("n_rows"),
        "n_cols": schema_summary.get("n_cols"),
    }

    cands = target_candidates.get("candidates") or []
    compact["target_candidates_top"] = [
        {"column": c.get("column"), "score": c.get("score"), "reasons": c.get("reasons")}
        for c in cands[:3]
    ]
    compact["target_candidate_heuristic_top"] = target_candidates.get("top_candidate")

    # Include analysis outputs if present (still keep them compact)
    corr = tool_result.get("correlation") or {}
    if isinstance(corr, dict) and corr:
        compact["correlation"] = {
            "method": corr.get("method"),
            "target": corr.get("target"),
            "top_abs": corr.get("top_abs"),
            "top_positive": corr.get("top_positive"),
            "top_negative": corr.get("top_negative"),
        }

    if "baseline_metrics" in tool_result:
        compact["baseline_metrics"] = tool_result.get("baseline_metrics")
    if "top_features" in tool_result:
        compact["top_features"] = tool_result.get("top_features")
    if "error" in tool_result:
        compact["error"] = tool_result.get("error")

    return compact


def interpreter_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Turn analysis results into a stakeholder-friendly answer.

    Key behavior:
    - Prefer public_tool_result as ground truth (deterministic, curated, stable).
    - Fallback to a compact view of internal tool_result if public_tool_result is missing.
    """
    public_view = (state.get("tool_result", {}) or {}).get("public_tool_result")
    summary = public_view if isinstance(public_view, dict) and public_view else _compact_tool_result(state)

    system = (
        "You are a business analyst. Use the provided summary as ground truth. "
        "Write a clear, concise answer in English.\n\n"
        "Rules:\n"
        "- Do NOT repeat the raw JSON/dict.\n"
        "- Focus on answering the user's question.\n"
        "- If the needed analysis results are missing, state what is missing in ONE sentence "
        "and suggest the next computation to run.\n"
        "- Keep the answer short (3-6 sentences)."
    )

    user = (
        f"Question: {state['question']}\n"
        f"Plan: {state.get('plan')}\n"
        f"Summary: {summary}\n"
    )

    final = llm.invoke([("system", system), ("user", user)]).content
    return {"final_answer": final}
