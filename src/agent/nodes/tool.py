from __future__ import annotations
from typing import Dict, Any

from ..state import AgentState
from ...tools.loaders import load_dataset, DataLoadError
from ...tools.pandas_tool import run_basic_analysis
from ...tools.profiling import profile_schema
from ...tools.target_heuristic import infer_target_candidates


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute analysis tools.
    Phase 2.1.1: Load dataset and expose dataset metadata.
    Phase 2.1.2: Profile schema summary (JSON-friendly).
    Phase 2.2.1: Infer target candidates (deterministic heuristic).
    """
    try:
        csv_path = state["csv_path"]
        question = state["question"]

        df, meta = load_dataset(csv_path)

        tool_result: Dict[str, Any] = {}
        tool_result["dataset_meta"] = meta.to_dict()

        # Phase 2.1.2
        schema_summary = profile_schema(df)
        tool_result["schema_summary"] = schema_summary

        # Phase 2.2.1
        tool_result["target_candidates"] = infer_target_candidates(schema_summary, top_k=3)

        # Phase 1 legacy (temporary)
        tool_result["legacy_analysis"] = run_basic_analysis(
            csv_path=csv_path,
            question=question,
        )

        # IMPORTANT: return df so planner_node can access it from state
        return {"tool_result": tool_result, "df": df}

    except DataLoadError as e:
        return {
            "tool_result": {
                "error": {"message": str(e), "payload": e.payload}
            }
        }
