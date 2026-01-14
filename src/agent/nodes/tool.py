from __future__ import annotations
from typing import Dict, Any

from ..state import AgentState
from ...tools.loaders import load_dataset, DataLoadError
from ...tools.pandas_tool import run_basic_analysis


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute analysis tools.
    Phase 2.1.1: Load dataset and expose dataset metadata.
    """
    try:
        csv_path = state["csv_path"]
        question = state["question"]

        # --- Phase 2.1.1: load dataset ---
        df, meta = load_dataset(csv_path)

        # keep dataframe in runtime state (not JSON-serializable)
        state["df"] = df

        # initialize tool_result
        tool_result: Dict[str, Any] = {}
        tool_result["dataset_meta"] = meta.to_dict()

        # --- Phase 1 legacy logic (temporary) ---
        # still allow existing analysis to run
        legacy_result = run_basic_analysis(
            csv_path=csv_path,
            question=question,
        )
        tool_result["legacy_analysis"] = legacy_result

        return {"tool_result": tool_result}

    except DataLoadError as e:
        return {
            "tool_result": {
                "error": {
                    "message": str(e),
                    "payload": e.payload,
                }
            }
        }

