from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState
from .nodes.planner import planner_node
from .nodes.tool import tool_node
from .nodes.interpreter import interpreter_node


def build_graph(llm: ChatGoogleGenerativeAI):
    """
    Phase 2 graph (updated):
      START -> tool -> planner -> interpreter -> END

    Rationale:
    - tool runs first to load data + produce schema_summary + target_candidates
    - planner can then decide whether to use optional LLM re-rank (based on tool_result)
    - interpreter produces the final user-facing answer
    """
    g = StateGraph(AgentState)

    # Wrap nodes that need llm
    g.add_node("tool", tool_node)
    g.add_node("planner", lambda state: planner_node(state, llm))
    g.add_node("interpreter", lambda state: interpreter_node(state, llm))

    # -----------------------------
    # CHANGED: graph execution order
    # (Phase 1 was START -> planner -> tool -> interpreter)
    # Now: START -> tool -> planner -> interpreter -> END
    # -----------------------------
    g.add_edge(START, "tool")          # <-- CHANGED (was "planner")
    g.add_edge("tool", "planner")      # <-- CHANGED (was "tool" after planner)
    g.add_edge("planner", "interpreter")  # <-- CHANGED (planner no longer goes to tool)
    g.add_edge("interpreter", END)

    return g.compile()
