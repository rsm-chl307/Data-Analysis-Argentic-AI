from __future__ import annotations
from typing import Dict, Any, Optional, List

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
    Create a compact, LLM-friendly summary of tool_result.

    Goal:
    - keep critical context for answering
    - avoid dumping huge schema_summary/legacy_analysis blobs into the prompt
    - keep full tool_result in state (unchanged), only compact for prompting
    """
    tool_result = state.get("tool_result", {}) or {}

    dataset_meta = tool_result.get("dataset_meta") or {}
    schema_summary = tool_result.get("schema_summary") or {}
    target_candidates = tool_result.get("target_candidates") or {}

    compact: Dict[str, Any] = {}

    # 1) Dataset overview (small)
    compact["dataset_overview"] = {
        "path": dataset_meta.get("path"),
        "n_rows": dataset_meta.get("n_rows"),
        "n_cols": dataset_meta.get("n_cols"),
        "columns": dataset_meta.get("columns"),
    }

    # 2) Selected target + task_type (most important for Phase 2)
    compact["target"] = state.get("target") or _safe_get(state, ["target_selection", "selected_target"])
    compact["task_type"] = tool_result.get("task_type")
    compact["target_selection"] = state.get("target_selection")

    # 3) Schema summary (only the smallest useful bits)
    compact["schema_brief"] = {
        "numeric_columns": schema_summary.get("numeric_columns"),
        "categorical_candidates": schema_summary.get("categorical_candidates"),
        "id_like_columns": schema_summary.get("id_like_columns"),
        "n_rows": schema_summary.get("n_rows"),
        "n_cols": schema_summary.get("n_cols"),
    }

    # 4) Target candidates (keep only top 3 candidates + scores)
    cands = target_candidates.get("candidates") or []
    compact["target_candidates_top"] = [
        {"column": c.get("column"), "score": c.get("score"), "reasons": c.get("reasons")}
        for c in cands[:3]
    ]
    compact["target_candidate_heuristic_top"] = target_candidates.get("top_candidate")

    # 5) Keep correlation/baseline results IF they exist (future-proof)
    # (These keys don't exist yet in your pipeline, but adding this now keeps interpreter stable later.)
    if "correlation" in tool_result:
        compact["correlation"] = tool_result.get("correlation")
    if "baseline_metrics" in tool_result:
        compact["baseline_metrics"] = tool_result.get("baseline_metrics")
    if "top_features" in tool_result:
        compact["top_features"] = tool_result.get("top_features")

    # Do NOT include legacy_analysis / full schema columns list etc.
    return compact


def interpreter_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Turn tool_result into a stakeholder-friendly answer.

    Key change:
    - Instead of dumping full tool_result, we pass a compact summary to reduce verbosity/cost.
    """
    compact = _compact_tool_result(state)

    system = (
        "You are a business analyst. Use the provided summary as ground truth. "
        "Write a clear, concise answer in English.\n\n"
        "Rules:\n"
        "- Do NOT repeat the raw JSON/dict.\n"
        "- Focus on answering the user's question.\n"
        "- If the needed analysis results are missing (e.g., correlation table not present), "
        "state what is missing in ONE sentence, and suggest the next computation to run.\n"
        "- Keep the answer short (3-6 sentences)."
    )

    user = (
        f"Question: {state['question']}\n"
        f"Plan: {state.get('plan')}\n"
        f"Summary: {compact}\n"
    )

    final = llm.invoke([("system", system), ("user", user)]).content
    return {"final_answer": final}
