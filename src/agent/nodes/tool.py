from __future__ import annotations
from typing import Dict, Any

from ..state import AgentState
from ...tools.loaders import load_dataset, DataLoadError
from ...tools.pandas_tool import run_basic_analysis
from ...tools.profiling import profile_schema


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute analysis tools.
    Phase 2.1.1: Load dataset and expose dataset metadata.
    Phase 2.1.2: Profile schema summary (JSON-friendly).
    """
    try:
        csv_path = state["csv_path"]
        question = state["question"]

        df, meta = load_dataset(csv_path)
        state["df"] = df  # runtime only

        tool_result: Dict[str, Any] = {}
        tool_result["dataset_meta"] = meta.to_dict()

        # Phase 2.1.2
        tool_result["schema_summary"] = profile_schema(df)

        # Phase 1 legacy (temporary)
        tool_result["legacy_analysis"] = run_basic_analysis(
            csv_path=csv_path,
            question=question,
        )

        return {"tool_result": tool_result}

    except DataLoadError as e:
        return {
            "tool_result": {
                "error": {"message": str(e), "payload": e.payload}
            }
        }
