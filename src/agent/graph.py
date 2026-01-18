from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState
from .nodes.planner import planner_node
from .nodes.tool import tool_node
from .nodes.analysis import analysis_node
from .nodes.interpreter import interpreter_node


def build_graph(llm: ChatGoogleGenerativeAI):
    """
    Phase 2 graph:
      START -> tool -> planner -> analysis -> interpreter -> END
    """
    g = StateGraph(AgentState)

    g.add_node("tool", tool_node)
    g.add_node("planner", lambda state: planner_node(state, llm))
    g.add_node("analysis", analysis_node)
    g.add_node("interpreter", lambda state: interpreter_node(state, llm))

    g.add_edge(START, "tool")
    g.add_edge("tool", "planner")
    g.add_edge("planner", "analysis")
    g.add_edge("analysis", "interpreter")
    g.add_edge("interpreter", END)

    return g.compile()
