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

    # Phase 1: fixed demo inputs
    csv_path = "data/samples/demo.csv"
    question = "Which channel has the highest average revenue and why?"

    result = graph.invoke({"csv_path": csv_path, "question": question})

    print("\n=== PLAN ===")
    for i, step in enumerate(result.get("plan", []), 1):
        print(f"{i}. {step}")

    print("\n=== TOOL_RESULT (summary) ===")
    tool_result = result.get("tool_result", {})
    print(tool_result)

    print("\n=== FINAL ANSWER ===")
    print(result.get("final_answer", ""))


if __name__ == "__main__":
    main()
