

# src/tools/task_type_inference.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class TaskTypeInferenceResult:
    task_type: str  # regression | binary_classification | multiclass_classification | eda_only
    task_type_source: str  # rules | llm (future)
    task_type_reasons: List[str]
    target_profile: Dict[str, Any]


def _is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True

    # Only try parse on small sample for cost + stability
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(30)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        return parsed.notna().mean() > 0.8

    return False


def _is_id_like(n_nonnull: int, n_unique: int) -> bool:
    if n_nonnull <= 0:
        return False
    unique_ratio = n_unique / n_nonnull
    # high uniqueness + many distinct values -> likely ID
    return unique_ratio > 0.9 and n_unique > 50


def infer_task_type(
    df: pd.DataFrame,
    target: str,
    *,
    min_samples: int = 30,
    max_missing_rate: float = 0.95,
    small_discrete_unique_threshold: int = 10,
) -> TaskTypeInferenceResult:
    """
    Rules-only task type inference.

    Output is designed for planner/tool routing:
      - regression
      - binary_classification
      - multiclass_classification
      - eda_only  (invalid / unsuitable target)
    """
    if target not in df.columns:
        return TaskTypeInferenceResult(
            task_type="eda_only",
            task_type_source="rules",
            task_type_reasons=[f"target column not found: {target}"],
            target_profile={"target": target},
        )

    y = df[target]
    n_rows = int(len(df))
    n_nonnull = int(y.notna().sum())
    missing_rate = 1.0 - (n_nonnull / n_rows) if n_rows > 0 else 1.0

    y_nonnull = y.dropna()
    n_unique = int(y_nonnull.nunique()) if n_nonnull > 0 else 0
    unique_ratio = (n_unique / n_nonnull) if n_nonnull > 0 else None
    dtype = str(y.dtype)

    profile: Dict[str, Any] = {
        "target": target,
        "dtype": dtype,
        "n_rows": n_rows,
        "n_nonnull": n_nonnull,
        "missing_rate": missing_rate,
        "n_unique": n_unique,
        "unique_ratio": unique_ratio,
    }
    reasons: List[str] = []

    # Hard gating
    if n_rows == 0:
        return TaskTypeInferenceResult("eda_only", "rules", ["empty dataframe"], profile)

    if n_nonnull == 0:
        return TaskTypeInferenceResult("eda_only", "rules", ["target is all missing"], profile)

    if missing_rate > max_missing_rate:
        return TaskTypeInferenceResult(
            "eda_only",
            "rules",
            [f"missing_rate too high: {missing_rate:.2f} > {max_missing_rate}"],
            profile,
        )

    if n_nonnull < min_samples:
        return TaskTypeInferenceResult(
            "eda_only",
            "rules",
            [f"too few non-missing samples: {n_nonnull} < {min_samples}"],
            profile,
        )

    if n_unique <= 1:
        return TaskTypeInferenceResult("eda_only", "rules", ["target is constant"], profile)

    # datetime-like
    if _is_datetime_like(y):
        profile["is_datetime_like"] = True
        return TaskTypeInferenceResult(
            "eda_only",
            "rules",
            ["target is datetime-like; supervised formulation needs definition (timestamp/duration)"],
            profile,
        )
    profile["is_datetime_like"] = False

    # ID-like
    if _is_id_like(n_nonnull, n_unique):
        profile["id_like"] = True
        return TaskTypeInferenceResult(
            "eda_only",
            "rules",
            ["target looks ID-like (unique_ratio > 0.9 and many uniques)"],
            profile,
        )
    profile["id_like"] = False

    # Classification vs regression
    if pd.api.types.is_bool_dtype(y):
        reasons.append("target dtype is boolean")
        return TaskTypeInferenceResult("binary_classification", "rules", reasons, profile)

    if (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
    ):
        reasons.append("target is categorical (non-numeric)")
        if n_unique == 2:
            reasons.append("n_unique == 2")
            return TaskTypeInferenceResult("binary_classification", "rules", reasons, profile)
        reasons.append(f"n_unique == {n_unique} (>2)")
        return TaskTypeInferenceResult("multiclass_classification", "rules", reasons, profile)

    if pd.api.types.is_numeric_dtype(y):
        # binary-like numeric {0,1}
        # only check small sample to avoid memory blow-ups
        uniq_vals = set(pd.Series(y_nonnull.unique()).head(50).tolist())
        if n_unique == 2 and uniq_vals.issubset({0, 1, 0.0, 1.0}):
            reasons.append("numeric but values look like {0,1}")
            return TaskTypeInferenceResult("binary_classification", "rules", reasons, profile)

        # small discrete numeric -> treat as (multi)classification by default
        if n_unique <= small_discrete_unique_threshold:
            reasons.append(
                f"numeric with small n_unique ({n_unique} <= {small_discrete_unique_threshold})"
            )
            return TaskTypeInferenceResult("multiclass_classification", "rules", reasons, profile)

        reasons.append("numeric with many unique values")
        return TaskTypeInferenceResult("regression", "rules", reasons, profile)

    # Fallback
    return TaskTypeInferenceResult(
        "eda_only",
        "rules",
        ["unable to infer supervised task type from dtype; default to eda_only"],
        profile,
    )


def infer_task_type_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool-friendly wrapper.

    Expected payload (adjust to your pipeline):
      {
        "df": <pandas.DataFrame>,
        "target": <str>
      }

    Returns:
      {
        "task_type": ...,
        "task_type_source": ...,
        "task_type_reasons": [...],
        "target_profile": {...}
      }
    """
    df = payload.get("df")
    target = payload.get("target")

    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "error": {
                "message": "infer_task_type_tool: payload['df'] must be a pandas DataFrame",
                "payload_keys": list(payload.keys()),
            }
        }
    if not target:
        return {
            "error": {
                "message": "infer_task_type_tool: payload['target'] is required",
                "payload_keys": list(payload.keys()),
            }
        }

    res = infer_task_type(df, str(target))
    return asdict(res)
