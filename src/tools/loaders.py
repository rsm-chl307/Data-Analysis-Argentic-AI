"""
Utility functions for loading tabular datasets in an agentic data analysis workflow.

This module provides a unified interface for reading CSV (required) and Parquet
(optional) files, handling common ingestion issues such as encoding and file
validation. It returns both a pandas DataFrame for downstream analysis and a
JSON-serializable metadata summary for planning and reasoning agents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# --------- Errors (tool-friendly) ---------

class DataLoadError(Exception):
    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.payload = payload or {}


# --------- Metadata ---------

@dataclass
class DatasetMeta:
    path: str
    file_type: str
    n_rows: int
    n_cols: int
    columns: list[str]

    # CSV-related 
    encoding: Optional[str] = None
    sep: Optional[str] = None
    sampled: bool = False
    sample_rows: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------- Loader ---------

def load_dataset(
    path: str,
    *,
    sample_rows: Optional[int] = None,
    csv_sep: Optional[str] = None,
    csv_encoding: Optional[str] = None,
) -> Tuple[pd.DataFrame, DatasetMeta]:
    """
    Load a tabular dataset from path.

    - Supports CSV (required), Parquet (optional)
    - sample_rows: if provided, only reads first N rows (useful for profiling)
    - csv_sep/csv_encoding: optional overrides for edge cases

    Returns (df, meta)
    """
    if not path or not isinstance(path, str):
        raise DataLoadError("Invalid path input.", payload={"path": path})

    if not os.path.exists(path):
        raise DataLoadError("File not found.", payload={"path": path})

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df, meta = _load_csv(
            path,
            sample_rows=sample_rows,
            sep=csv_sep,
            encoding=csv_encoding,
        )
        return df, meta

    if ext in {".parquet", ".pq"}:
        df, meta = _load_parquet(path, sample_rows=sample_rows)
        return df, meta

    raise DataLoadError(
        "Unsupported file type. Only .csv and .parquet are supported.",
        payload={"path": path, "ext": ext},
    )


def _load_csv(
    path: str,
    *,
    sample_rows: Optional[int],
    sep: Optional[str],
    encoding: Optional[str],
) -> Tuple[pd.DataFrame, DatasetMeta]:
    # Strategy:
    # 1) Try a small set of common encodings if encoding not specified
    # 2) Use python engine if separator not specified (more flexible)
    encodings_to_try = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin-1"]

    last_err: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            read_kwargs = dict(
                encoding=enc,
                nrows=sample_rows,
            )
            if sep:
                read_kwargs["sep"] = sep
                read_kwargs["engine"] = "python"  # safe for weird separators
            else:
                # Let pandas infer; python engine is more permissive
                read_kwargs["engine"] = "python"

            df = pd.read_csv(path, **read_kwargs)

            meta = DatasetMeta(
                path=path,
                file_type="csv",
                n_rows=int(df.shape[0]),
                n_cols=int(df.shape[1]),
                columns=[str(c) for c in df.columns],
                encoding=enc,
                sep=sep,
                sampled=sample_rows is not None,
                sample_rows=sample_rows,
            )
            return df, meta

        except Exception as e:
            last_err = e
            continue

    raise DataLoadError(
        "Failed to read CSV with tried encodings.",
        payload={"path": path, "tried_encodings": encodings_to_try, "error": repr(last_err)},
    )


def _load_parquet(
    path: str,
    *,
    sample_rows: Optional[int],
) -> Tuple[pd.DataFrame, DatasetMeta]:
    # Parquet reading depends on pyarrow/fastparquet
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise DataLoadError(
            "Failed to read Parquet. Ensure 'pyarrow' or 'fastparquet' is installed.",
            payload={"path": path, "error": repr(e)},
        )

    if sample_rows is not None and sample_rows > 0:
        df = df.head(sample_rows)

    meta = DatasetMeta(
        path=path,
        file_type="parquet",
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns],
        sampled=sample_rows is not None,
        sample_rows=sample_rows,
    )
    return df, meta
