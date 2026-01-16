from __future__ import annotations

import os
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import get_gemini_api_key, get_gemini_model_name
from .agent.graph import build_graph


def _brief_tool_result(tool_result: dict) -> dict:
    dataset_meta = tool_result.get("dataset_meta", {}) or {}
    target_candidates = tool_result.get("target_candidates", {}) or {}

    brief = {
        "dataset_meta": {
            "path": dataset_meta.get("path"),
            "n_rows": dataset_meta.get("n_rows"),
            "n_cols": dataset_meta.get("n_cols"),
        },
        "top_target_candidate": target_candidates.get("top_candidate"),
        "task_type": tool_result.get("task_type"),
    }

    # Optional: show why task_type was chosen (short)
    tti = tool_result.get("task_type_inference") or {}
    if isinstance(tti, dict) and tti:
        brief["task_type_reasons"] = tti.get("task_type_reasons")
        profile = tti.get("target_profile") or {}
        brief["target_profile"] = {
            "target": profile.get("target"),
            "dtype": profile.get("dtype"),
            "n_unique": profile.get("n_unique"),
            "missing_rate": profile.get("missing_rate"),
        }

    return brief


def main():
    # Ensure env is loaded
    api_key = get_gemini_api_key()
    model_name = get_gemini_model_name()

    # Gemini client (LangChain wrapper)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )

    graph = build_graph(llm)

    # Phase 1: fixed demo inputs
    csv_path = "data/samples/red wine/winequality-red.csv"
    question = (
        "Which chemical properties have the strongest impact on red wine quality, "
        "and in which direction do they influence it?"
    )

    result = graph.invoke({"csv_path": csv_path, "question": question})

    print("\n=== PLAN ===")
    for i, step in enumerate(result.get("plan", []), 1):
        print(f"{i}. {step}")

    tool_result = result.get("tool_result", {}) or {}

    # Toggle verbose debug output via env var:
    #   DEBUG_TOOL_RESULT=1 python -m src.app
    debug = os.getenv("DEBUG_TOOL_RESULT", "0").strip() in {"1", "true", "True", "YES", "yes"}

    if debug:
        print("\n=== TOOL_RESULT (full) ===")
        print(tool_result)
    else:
        print("\n=== TOOL_RESULT ===")
        print(_brief_tool_result(tool_result))

    print("\n=== FINAL ANSWER ===")
    print(result.get("final_answer", ""))


if __name__ == "__main__":
    main()
