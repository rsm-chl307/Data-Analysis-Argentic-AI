"""
File: src/agent/graph.py
Mission: Assembles the LangGraph state machine. It transitions the agent from a 
linear script to a cyclic, autonomous workflow by incorporating conditional edges 
that allow for error recovery and self-correction.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState
from .nodes.planner import planner_node
from .nodes.tool import tool_node
from .nodes.analysis import analysis_node
from .nodes.interpreter import interpreter_node

# ==========================================
# ROUTERS IMPORT (NEW: Import our traffic controller functions)
# ==========================================
from .routers import route_after_tool, route_after_analysis


def build_graph(llm: ChatGoogleGenerativeAI):
    """
    Phase 3 graph (Cyclic/Agentic):
      START -> tool
      tool -> [route_after_tool] -> planner OR interpreter
      planner -> analysis
      analysis -> [route_after_analysis] -> interpreter OR planner (retry loop)
      interpreter -> END
    """
    g = StateGraph(AgentState)

    g.add_node("tool", tool_node)
    g.add_node("planner", lambda state: planner_node(state, llm))
    g.add_node("analysis", analysis_node)
    g.add_node("interpreter", lambda state: interpreter_node(state, llm))

    # 1. Start execution
    g.add_edge(START, "tool")
    
    # 2. Tool routing (NEW: Check if data loading was successful)
    # Replaced: g.add_edge("tool", "planner")
    g.add_conditional_edges(
        "tool",
        route_after_tool,
        {
            "planner": "planner",
            "interpreter": "interpreter"
        }
    )

    # 3. Planner strictly goes to analysis
    g.add_edge("planner", "analysis")
    
    # 4. Analysis routing (NEW: Core self-correction loop)
    # Replaced: g.add_edge("analysis", "interpreter")
    g.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "planner": "planner",          # Error detected -> Retry planning
            "interpreter": "interpreter"   # Success or max retries -> Finalize
        }
    )

    # 5. End execution
    g.add_edge("interpreter", END)

    return g.compile()