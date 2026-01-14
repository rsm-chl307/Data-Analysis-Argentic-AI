"""
Schema profiling utilities for agentic data analysis.

This module inspects a pandas DataFrame and produces a JSON-serializable
schema summary, including column data types, missing rates, cardinality,
and heuristic flags (numeric, categorical, ID-like). The output is designed
for planner and LLM consumption rather than direct modeling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import os
import inspect

def _numeric_parseable_rate(s: pd.Series) -> float:
    """
    For object-like columns, estimate how many non-null values can be parsed as numeric.
    """
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return 1.0

    non_null = s.dropna()
    if len(non_null) == 0:
        return 0.0

    # Try numeric conversion
    parsed = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return float(parsed.notna().mean())


def _is_id_like(col_name: str, s: pd.Series, n_rows: int) -> bool:
    """
    Heuristic to flag ID-like columns.
    """
    name = col_name.lower()

    # Strong name-based signals
    if any(k in name for k in ["id", "uuid", "guid", "hash"]):
        return True

    # Numeric continuous variables should NOT be treated as IDs
    if pd.api.types.is_numeric_dtype(s):
        return False

    non_null = s.dropna()
    if len(non_null) == 0:
        return False

    nunique = int(non_null.nunique())
    unique_ratio = nunique / max(1, len(non_null))

    # High uniqueness for non-numeric columns likely indicates identifiers or free text
    if unique_ratio >= 0.98 and len(non_null) >= 50:
        return True

    return False



def _is_categorical_candidate(s: pd.Series, n_rows: int) -> bool:
    """
    Categorical candidate if:
    - dtype is object/category/bool, or
    - numeric but low cardinality (e.g., 0/1, 1-5 ratings)
    """
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True

    nunique = int(s.nunique(dropna=True))

    if pd.api.types.is_numeric_dtype(s):
        return nunique <= 10

    # object-like: low cardinality relative to rows
    return nunique <= max(20, int(0.05 * n_rows))


def profile_schema(    
    df: pd.DataFrame,
    *,
    sample_values_n: int = 5,
    max_columns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Produce a JSON-serializable schema summary for planner/LLM/tool routing.
    """
    print("[DEBUG] USING profiling.py:", os.path.abspath(__file__))
    print("[DEBUG] _is_id_like starts at line:",
            inspect.getsourcelines(_is_id_like)[1])
    n_rows, n_cols = df.shape
    cols = list(df.columns)

    if max_columns is not None:
        cols = cols[:max_columns]

    columns_summary: List[Dict[str, Any]] = []
    numeric_columns: List[str] = []
    categorical_candidates: List[str] = []
    id_like_columns: List[str] = []

    for c in cols:
        s = df[c]
        missing_rate = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)

        is_numeric = bool(pd.api.types.is_numeric_dtype(s))
        parseable_rate = _numeric_parseable_rate(s) if (not is_numeric) else 1.0
        numeric_parseable = bool((not is_numeric) and (parseable_rate >= 0.95))

        is_cat = _is_categorical_candidate(s, n_rows)
        is_id = _is_id_like(c, s, n_rows)

        # sample values (stringified for JSON safety)
        sample_vals_raw = s.dropna().unique().tolist()[:sample_values_n]
        sample_vals = [str(v) for v in sample_vals_raw]

        col_payload = {
            "name": str(c),
            "dtype": dtype,
            "missing_rate": missing_rate,
            "n_unique": nunique,
            "sample_values": sample_vals,
            "is_numeric_candidate": is_numeric or numeric_parseable,
            "is_categorical_candidate": bool(is_cat),
            "is_id_like": bool(is_id),
            "numeric_parseable_rate": float(parseable_rate),
        }
        columns_summary.append(col_payload)

        if col_payload["is_numeric_candidate"]:
            numeric_columns.append(str(c))
        if col_payload["is_categorical_candidate"]:
            categorical_candidates.append(str(c))
        if col_payload["is_id_like"]:
            id_like_columns.append(str(c))

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": columns_summary,
        "numeric_columns": numeric_columns,
        "categorical_candidates": categorical_candidates,
        "id_like_columns": id_like_columns,
        "notes": {
            "numeric_parseable_rule": "object-like columns with >=95% values parseable as numeric are treated as numeric candidates",
            "categorical_rule": "object/category/bool or low-cardinality numeric (<=10 unique) are treated as categorical candidates",
            "id_like_rule": "name contains id/uuid/hash or uniqueness ratio >= 0.98 with >=50 non-null rows",
        },
    }
