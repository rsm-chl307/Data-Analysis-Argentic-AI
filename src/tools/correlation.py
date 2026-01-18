from __future__ import annotations
from typing import Dict, Any, List

import pandas as pd


def compute_pearson_correlation(df: pd.DataFrame, target: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Compute Pearson correlation between numeric features and target.
    Returns JSON-friendly payload.
    """
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if target not in numeric_df.columns:
        return {
            "error": {
                "message": f"target '{target}' is not numeric; Pearson correlation requires numeric target",
                "payload": {"target": target},
            }
        }

    sub = numeric_df.dropna(subset=[target])
    if sub.empty:
        return {
            "error": {
                "message": "no non-missing rows for target; cannot compute correlation",
                "payload": {"target": target},
            }
        }

    corr_series = sub.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").dropna()
    if corr_series.empty:
        return {
            "error": {
                "message": "no valid numeric features to correlate with target",
                "payload": {"target": target},
            }
        }

    corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

    rows: List[Dict[str, Any]] = [
        {"feature": idx, "pearson_r": float(val), "direction": "positive" if val > 0 else "negative"}
        for idx, val in corr_sorted.items()
    ]

    pos = [r for r in rows if r["pearson_r"] > 0][:top_k]
    neg = [r for r in rows if r["pearson_r"] < 0][:top_k]

    return {
        "method": "pearson",
        "target": target,
        "n_features_used": int(len(rows)),
        "top_abs": rows[:top_k],
        "top_positive": pos,
        "top_negative": neg,
    }
