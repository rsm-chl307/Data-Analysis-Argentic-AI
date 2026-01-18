"""
public_view.py

This module converts the internal `tool_result` (used by planner and analysis)
into a compact, display-safe `public_tool_result`.

The goal is to:
- Expose only allowlisted, user-facing analysis results
- Trim large internal payloads (e.g., full tables) to concise summaries
- Keep outputs stable and predictable as new analysis tools are added

`tool_result` is internal and may be verbose.
`public_tool_result` is the curated view used by the interpreter and UI.
"""

from __future__ import annotations
from typing import Dict, Any, List


DEFAULT_ALLOWLIST = [
    "dataset_meta",
    "task_type",
    "target_candidates",
    "correlation",
    "baseline_metrics",
    "top_features",
    "plots",
    "analysis_executed",
    "error",
]


def make_public_tool_result(tool_result: Dict[str, Any], allowlist: List[str] | None = None) -> Dict[str, Any]:
    """
    Create a public, display-safe view of tool_result.
    - Keeps only allowlisted keys
    - Trims oversized payloads (e.g., keep top_k only)
    """
    allow = set(allowlist or DEFAULT_ALLOWLIST)
    tr = tool_result or {}

    out: Dict[str, Any] = {}

    for k in allow:
        if k in tr:
            out[k] = tr[k]

    # Trim dataset_meta to essentials
    dm = out.get("dataset_meta")
    if isinstance(dm, dict):
        out["dataset_meta"] = {
            "path": dm.get("path"),
            "n_rows": dm.get("n_rows"),
            "n_cols": dm.get("n_cols"),
            "columns": dm.get("columns"),
        }

    # Trim target_candidates to top3
    tc = out.get("target_candidates")
    if isinstance(tc, dict):
        cands = tc.get("candidates") or []
        out["target_candidates"] = {
            "top_candidate": tc.get("top_candidate"),
            "candidates": cands[:3],
        }

    # Trim correlation payload if it contains big tables
    corr = out.get("correlation")
    if isinstance(corr, dict):
        # keep only these keys if present
        out["correlation"] = {
            "method": corr.get("method"),
            "target": corr.get("target"),
            "top_abs": corr.get("top_abs"),
            "top_positive": corr.get("top_positive"),
            "top_negative": corr.get("top_negative"),
        }

    return out
