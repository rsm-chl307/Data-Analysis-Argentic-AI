"""
Deterministic target variable inference for agentic data analysis.

This module implements a rule-based (heuristic) approach to infer and rank
potential target variables from a dataset schema summary. It uses only
structural signals—such as column names, cardinality, data type flags, and
ID-like indicators—without relying on dataset-specific assumptions or
language models.

The output is a ranked list of target candidates with explicit scores and
human-readable reasons, designed to:
1) Provide a stable prior for downstream planning and analysis,
2) Reduce reliance on LLM guessing or hard-coded targets, and
3) Serve as a foundation for optional LLM-based re-ranking in later phases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# -------------------------
# Scoring config (Phase 2.2.1)
# -------------------------

_STRONG_NAME_TOKENS = {"target", "label", "class"}
_DOMAIN_NAME_TOKENS = {"quality", "score", "rating", "outcome"}

# Weak "feature-like" tokens (very light penalty)
_FEATURE_LIKE_TOKENS = {"age", "price", "amount", "count", "num", "qty", "quantity", "total", "sum"}

# Split name into tokens by common delimiters and camel-case-ish boundaries
_SPLIT_RE = re.compile(r"[^\w]+|(?<=[a-z])(?=[A-Z])")


@dataclass
class TargetCandidate:
    column: str
    score: float
    reasons: List[str]
    signals: Dict[str, Any]


def _tokenize(col_name: str) -> List[str]:
    raw = str(col_name).strip()
    parts = [p for p in _SPLIT_RE.split(raw) if p]
    return [p.lower() for p in parts]


def _score_column(
    col: Dict[str, Any],
    *,
    n_rows: int,
    is_last: bool,
) -> TargetCandidate:
    name = str(col.get("name", ""))
    tokens = set(_tokenize(name))

    n_unique = int(col.get("n_unique", 0))
    missing_rate = float(col.get("missing_rate", 0.0))
    is_id_like = bool(col.get("is_id_like", False))
    is_numeric = bool(col.get("is_numeric_candidate", False))
    is_categorical = bool(col.get("is_categorical_candidate", False))

    score = 0.0
    reasons: List[str] = []
    signals: Dict[str, Any] = {
        "n_unique": n_unique,
        "missing_rate": missing_rate,
        "is_id_like": is_id_like,
        "is_numeric_candidate": is_numeric,
        "is_categorical_candidate": is_categorical,
        "is_last_column": is_last,
        "name_tokens": sorted(tokens),
    }

    # -------------------------
    # Negative signals (strong)
    # -------------------------
    if is_id_like:
        score -= 1.0
        reasons.append("penalty:id_like(-1.0)")

    # Too many uniques (near row count) is usually not a good supervised target
    if n_rows > 0 and n_unique >= int(0.8 * n_rows):
        score -= 0.6
        reasons.append("penalty:too_many_uniques(-0.6)")

    # -------------------------
    # Name-based signals
    # -------------------------
    if tokens & _STRONG_NAME_TOKENS:
        score += 0.6
        reasons.append("bonus:strong_name_token(+0.6)")

    if tokens & _DOMAIN_NAME_TOKENS:
        score += 0.4
        reasons.append("bonus:domain_name_token(+0.4)")

    # -------------------------
    # Cardinality/type signals
    # -------------------------
    # (A) Categorical target candidates: low-ish unique count
    if is_categorical and 2 <= n_unique <= 20:
        score += 0.4
        reasons.append("bonus:low_cardinality_categorical(+0.4)")

    # (B) Numeric targets
    if is_numeric:
        if 3 <= n_unique <= 10:
            score += 0.3
            reasons.append("bonus:ordinal_numeric(+0.3)")
        elif 10 < n_unique <= 100:
            score += 0.2
            reasons.append("bonus:moderate_numeric_cardinality(+0.2)")

        # Binary numeric is often a feature flag (light penalty)
        if n_unique == 2:
            score -= 0.2
            reasons.append("penalty:binary_numeric(-0.2)")

    # -------------------------
    # Weak heuristics
    # -------------------------
    if is_last:
        score += 0.1
        reasons.append("bonus:last_column_bias(+0.1)")

    # Very light "feature-like" name penalty (weak)
    if tokens & _FEATURE_LIKE_TOKENS:
        score -= 0.1
        reasons.append("penalty:feature_like_name(-0.1)")

    # Optional: missingness penalty (keep it weak)
    if missing_rate >= 0.3:
        score -= 0.1
        reasons.append("penalty:high_missing_rate(-0.1)")

    return TargetCandidate(column=name, score=float(score), reasons=reasons, signals=signals)


def infer_target_candidates(
    schema_summary: Dict[str, Any],
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Infer a ranked list of potential target columns using deterministic heuristics.

    Input: schema_summary from profile_schema()
    Output (JSON-friendly):
      - candidates: ranked list [{column, score, reasons, signals}, ...]
      - top_candidate: best guess (or None)
      - notes: heuristic version + guidance
    """
    if not isinstance(schema_summary, dict):
        raise ValueError("schema_summary must be a dict")

    n_rows = int(schema_summary.get("n_rows", 0))
    cols = schema_summary.get("columns", [])
    if not isinstance(cols, list) or len(cols) == 0:
        return {
            "candidates": [],
            "top_candidate": None,
            "notes": {"warning": "No columns found in schema_summary."},
        }

    candidates: List[TargetCandidate] = []
    last_idx = len(cols) - 1

    for i, col in enumerate(cols):
        if not isinstance(col, dict) or "name" not in col:
            continue
        candidates.append(
            _score_column(col, n_rows=n_rows, is_last=(i == last_idx))
        )

    # Rank by score desc, tie-breaker: lower missing_rate, then lower n_unique
    def _sort_key(c: TargetCandidate):
        mr = float(c.signals.get("missing_rate", 0.0))
        nu = int(c.signals.get("n_unique", 0))
        return (-c.score, mr, nu, c.column.lower())

    candidates_sorted = sorted(candidates, key=_sort_key)

    # Ensure at least top_k returned (if available)
    k = min(max(1, top_k), len(candidates_sorted))
    top_list = candidates_sorted[:k]

    top_candidate = top_list[0].column if top_list else None

    return {
        "candidates": [asdict(c) for c in top_list],
        "top_candidate": top_candidate,
        "notes": {
            "heuristic_version": "phase2.2.1.v1",
            "guidance": "This is a ranked suggestion. You may override target explicitly or refine with LLM (phase 2.2.2).",
        },
    }
