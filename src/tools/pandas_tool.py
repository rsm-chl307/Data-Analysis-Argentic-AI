from __future__ import annotations
from typing import Dict, Any
import pandas as pd


def run_basic_analysis(csv_path: str, question: str) -> Dict[str, Any]:
    """
    Phase 1: minimal deterministic analysis.
    - Loads CSV
    - If question implies "highest average revenue by channel", compute it
    - Otherwise returns basic describe
    """
    df = pd.read_csv(csv_path)

    q = question.lower()

    # Minimal heuristic for demo
    if "highest" in q and ("average" in q or "avg" in q) and "revenue" in q and "channel" in df.columns:
        by = df.groupby("channel")["revenue"].mean().sort_values(ascending=False)
        return {
            "analysis": "avg_revenue_by_channel",
            "top_channel": str(by.index[0]),
            "top_value": float(by.iloc[0]),
            "table": by.reset_index().to_dict(orient="records"),
        }

    return {
        "analysis": "basic_describe",
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "describe": df.describe(include="all").to_string(),
    }
