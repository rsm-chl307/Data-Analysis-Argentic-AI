from __future__ import annotations
from typing import Dict, Any, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import AgentState
from ...tools.target_rerank_llm import rerank_target_candidates_with_llm
from ...tools.task_type_inference import infer_task_type  # <<< ADD (2.2.3)


def _should_rerank_with_llm(question: str, candidates: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Deterministic gating: decide whether to spend LLM budget on re-ranking.
    Returns (should_rerank, reasons).
    """
    reasons: List[str] = []
    if not candidates or len(candidates) < 2:
        return False, ["insufficient_candidates"]

    # defensive sort by heuristic score
    c_sorted = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    top1, top2 = c_sorted[0], c_sorted[1]

    s1 = float(top1.get("score", 0.0))
    s2 = float(top2.get("score", 0.0))
    gap = s1 - s2

    top1_reasons = " ".join([str(r) for r in top1.get("reasons", [])]).lower()
    q = (question or "").lower()

    # 1) small score gap => uncertainty
    if gap < 0.15:
        reasons.append(f"small_score_gap({gap:.2f})")

    # 2) no name-based signal on top1
    if ("strong_name_token" not in top1_reasons) and ("domain_name_token" not in top1_reasons):
        reasons.append("top1_missing_name_signal")

    # 3) question strongly implies outcome but top1 score is not confident
    outcome_keywords = [
        "predict", "prediction", "classify", "classification", "regression",
        "label", "target", "outcome", "churn", "conversion"
    ]
    if any(k in q for k in outcome_keywords) and s1 < 0.6:
        reasons.append("question_outcome_intent_but_low_top1_score")

    return (len(reasons) > 0), reasons


def planner_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Produce an analysis plan and select a target variable.

    Phase 2.2.2 (in planner): optionally re-rank heuristic target candidates using LLM,
    gated by deterministic uncertainty rules. If rerank is not triggered or fails,
    fall back to the heuristic top candidate.

    Phase 2.2.3 (added): infer supervised task type from selected target (rules-only).
    Writes back into tool_result: task_type + task_type_inference payload.
    """
    question = state["question"]
    tool_result = state.get("tool_result", {}) or {}

    schema_summary = tool_result.get("schema_summary")
    target_info = tool_result.get("target_candidates") or {}
    candidates = target_info.get("candidates", [])
    heuristic_top = target_info.get("top_candidate")

    selected_target = heuristic_top
    rerank_payload = None

    should_rerank, gate_reasons = _should_rerank_with_llm(question, candidates)

    if should_rerank and schema_summary is not None and candidates:
        rerank_payload = rerank_target_candidates_with_llm(
            llm=llm,
            question=question,
            heuristic_result=target_info,
            schema_summary=schema_summary,
            data_dictionary=state.get("data_dictionary"),  # optional
        )
        selected_target = rerank_payload.get("final_target") or heuristic_top

    # -------------------------
    # Phase 2.2.3: task type inference (rules-only, cheap)
    # -------------------------
    df = state.get("df")  # tool_node sets this (runtime only)
    task_type_payload = None
    if df is not None:
        task_type_payload = infer_task_type(df, selected_target)

    # Merge back into existing tool_result (do NOT overwrite other keys)
    merged_tool_result = dict(tool_result)
    if task_type_payload is not None:
        merged_tool_result["task_type"] = task_type_payload.get("task_type")
        merged_tool_result["task_type_inference"] = task_type_payload

    # -------------------------
    # Plan generation (keep your original behavior)
    # -------------------------
    system = (
        "You are a data analysis planner. "
        "Return a short step-by-step plan (2-4 steps) to answer the user's question using pandas. "
        "Keep steps concise and actionable."
    )
    user = (
        f"Question: {question}\n"
        f"Selected target column: {selected_target}\n"
        f"Note: If target is None, propose how to identify it from the dataset."
    )

    msg = llm.invoke([("system", system), ("user", user)]).content
    plan: List[str] = [line.strip("-• ").strip() for line in msg.splitlines() if line.strip()]
    plan = plan[:4] if plan else ["Load the dataset", "Compute relevant summary stats", "Answer the question"]

    out: Dict[str, Any] = {
        "plan": plan,
        "target": selected_target,
        "target_selection": {
            "selected_target": selected_target,
            "source": "llm_rerank" if rerank_payload and not rerank_payload.get("fallback_used") else "heuristic",
            "gate_reasons": gate_reasons,
            "heuristic_top": heuristic_top,
        },
        # IMPORTANT: keep tool_result updated with task_type info
        "tool_result": merged_tool_result,
    }
    if rerank_payload is not None:
        out["target_rerank"] = rerank_payload

    return out

