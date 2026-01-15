"""
LLM-based semantic re-ranking for target variable selection.

This module refines a deterministically generated list of target candidates
using a language model. It does not discover new target columns; instead,
it re-ranks and validates existing candidates based on semantic alignment
with the user question and optional data dictionary context.

Key design principles:
- Operates strictly on a closed candidate set produced by heuristics.
- Uses LLMs only for semantic judgment, not structural inference.
- Provides safe fallback to heuristic results if LLM output is invalid.
- Returns JSON-serializable outputs suitable for agent state and logging.

This module is intended as an optional enhancement layer on top of
deterministic target heuristics in an agentic data analysis pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


class LLMRerankError(Exception):
    """Tool-friendly error for LLM rerank failures."""
    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.payload = payload or {}


def _build_rerank_prompt(
    *,
    question: str,
    candidates: List[Dict[str, Any]],
    schema_summary: Dict[str, Any],
    data_dictionary: Optional[Dict[str, Any]] = None,
) -> str:
    # Keep prompt compact: only include top candidates + minimal schema context
    cols_brief = []
    for c in candidates:
        col = c.get("column")
        score = c.get("score")
        reasons = c.get("reasons", [])
        signals = c.get("signals", {})
        cols_brief.append(
            {
                "column": col,
                "heuristic_score": score,
                "heuristic_reasons": reasons,
                "signals": {
                    "n_unique": signals.get("n_unique"),
                    "missing_rate": signals.get("missing_rate"),
                    "is_numeric_candidate": signals.get("is_numeric_candidate"),
                    "is_categorical_candidate": signals.get("is_categorical_candidate"),
                    "is_id_like": signals.get("is_id_like"),
                    "is_last_column": signals.get("is_last_column"),
                },
            }
        )

    payload = {
        "question": question,
        "candidates": cols_brief,
        "dataset_context": {
            "n_rows": schema_summary.get("n_rows"),
            "n_cols": schema_summary.get("n_cols"),
        },
        "data_dictionary": data_dictionary,  # may be None
        "constraints": {
            "only_choose_from_candidates": True,
            "no_new_columns": True,
            "output_format": "json_only",
        },
        "expected_output_schema": {
            "final_target": "string (must be one of candidates.column)",
            "ranking": [
                {
                    "column": "string (must be one of candidates.column)",
                    "rank": "int (1=best)",
                    "reason": "string (short)",
                }
            ],
            "confidence": "high|medium|low",
        },
    }

    return (
        "You are a data analyst assistant.\n"
        "Your task is to re-rank potential target variables based on the user question.\n"
        "You MUST choose the final target from the provided candidate columns only.\n"
        "Do NOT invent new columns.\n"
        "Return JSON ONLY that matches the expected_output_schema.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _safe_json_loads(text: str) -> Dict[str, Any]:
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting the first JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    raise ValueError("Failed to parse JSON from LLM output.")


def rerank_target_candidates_with_llm(
    *,
    llm: Any,
    question: str,
    heuristic_result: Dict[str, Any],
    schema_summary: Dict[str, Any],
    data_dictionary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LLM-based semantic re-ranking on top of deterministic target candidates.

    - Only re-ranks among heuristic candidates.
    - Fallback to heuristic top_candidate if LLM output is invalid.

    Returns JSON-friendly dict:
      - final_target
      - reranked_candidates (with llm_rank and llm_reason)
      - fallback_used
      - raw_llm_output
    """
    candidates = (heuristic_result or {}).get("candidates", [])
    heuristic_top = (heuristic_result or {}).get("top_candidate")

    if not candidates or not heuristic_top:
        return {
            "final_target": heuristic_top,
            "reranked_candidates": candidates or [],
            "llm_notes": {"warning": "No candidates provided; rerank skipped."},
            "fallback_used": True,
            "raw_llm_output": None,
        }

    allowed = {c.get("column") for c in candidates if c.get("column")}
    prompt = _build_rerank_prompt(
        question=question,
        candidates=candidates,
        schema_summary=schema_summary,
        data_dictionary=data_dictionary,
    )

    try:
        resp = llm.invoke(prompt)
        raw_text = getattr(resp, "content", resp)  # AIMessage.content or plain string
        parsed = _safe_json_loads(str(raw_text))

        final_target = parsed.get("final_target")
        ranking = parsed.get("ranking", [])
        confidence = parsed.get("confidence", "low")

        # Validate final_target
        if final_target not in allowed:
            raise LLMRerankError(
                "LLM returned a final_target not in candidate set.",
                payload={"final_target": final_target, "allowed": sorted(list(allowed))},
            )

        # Build rank map
        rank_map: Dict[str, Tuple[int, str]] = {}
        for item in ranking:
            col = item.get("column")
            r = item.get("rank")
            reason = item.get("reason", "")
            if col in allowed and isinstance(r, int):
                rank_map[col] = (r, str(reason))

        # Ensure every candidate has a rank; missing ones go after
        reranked = []
        for c in candidates:
            col = c.get("column")
            if col in rank_map:
                llm_rank, llm_reason = rank_map[col]
            else:
                llm_rank, llm_reason = (999, "Not ranked by LLM; kept for completeness.")
            reranked.append(
                {
                    "column": col,
                    "score": c.get("score"),
                    "heuristic_reasons": c.get("reasons", []),
                    "llm_rank": llm_rank,
                    "llm_reason": llm_reason,
                }
            )

        reranked_sorted = sorted(reranked, key=lambda x: (x["llm_rank"], -(x.get("score") or 0.0)))

        return {
            "final_target": final_target,
            "reranked_candidates": reranked_sorted,
            "llm_notes": {
                "model_confidence": confidence,
                "policy": "only_choose_from_candidates",
            },
            "fallback_used": False,
            "raw_llm_output": str(raw_text),
        }

    except Exception as e:
        # Fallback to heuristic
        return {
            "final_target": heuristic_top,
            "reranked_candidates": [
                {
                    "column": c.get("column"),
                    "score": c.get("score"),
                    "heuristic_reasons": c.get("reasons", []),
                    "llm_rank": None,
                    "llm_reason": None,
                }
                for c in candidates
            ],
            "llm_notes": {
                "error": str(e),
                "policy": "fallback_to_heuristic_top_candidate",
            },
            "fallback_used": True,
            "raw_llm_output": None,
        }
