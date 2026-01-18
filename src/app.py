from __future__ import annotations

import os
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import get_gemini_api_key, get_gemini_model_name
from .agent.graph import build_graph


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

    # Demo inputs
    csv_path = "data/samples/red wine/winequality-red.csv"
    question = (
        "Which chemical properties have the strongest impact on red wine quality, "
        "and in which direction do they influence it?"
    )

    result = graph.invoke({"csv_path": csv_path, "question": question})

    print("\n=== PLAN ===")
    for i, step in enumerate(result.get("plan", []), 1):
        print(f"{i}. {step}")

    # Toggle verbose debug output via env var:
    #   DEBUG_TOOL_RESULT=1 python -m src.app
    debug = os.getenv("DEBUG_TOOL_RESULT", "0").strip() in {"1", "true", "True", "YES", "yes"}

    if debug:
        print("\n=== TOOL_RESULT (full, internal) ===")
        print(result.get("tool_result", {}) or {})
    else:
        print("\n=== TOOL_RESULT (public) ===")
        # public_tool_result is produced by analysis_node via make_public_tool_result(...)
        # Fallback: if for some reason it doesn't exist, show a minimal safe dict
        public_view = (result.get("tool_result", {}) or {}).get("public_tool_result")
        if isinstance(public_view, dict) and public_view:
            print(public_view)
        else:
            tool_result = result.get("tool_result", {}) or {}
            print({
                "dataset_meta": (tool_result.get("dataset_meta") or {}),
                "task_type": tool_result.get("task_type"),
                "error": tool_result.get("error"),
            })

    print("\n=== FINAL ANSWER ===")
    print(result.get("final_answer", ""))


if __name__ == "__main__":
    main()
