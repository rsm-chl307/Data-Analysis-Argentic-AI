"""
File: src/agent/routers.py
Task: Defines the routing logic (conditional edges) for the LangGraph state machine.
These functions evaluate the current AgentState and determine the next node to execute,
enabling error recovery and cyclic workflows.
"""

from typing import Literal
from .state import AgentState

def route_after_tool(state: AgentState) -> Literal["planner", "interpreter"]:
    """
    Determines the next step after the initial tool_node execution (data loading & profiling).
    If the data fails to load, it routes directly to the interpreter to report the error.
    Otherwise, it proceeds to the planner.
    """
    tool_result = state.get("tool_result", {})
    
    # Check if there is an error in the initial tool execution (e.g., file not found)
    if "error" in tool_result:
        return "interpreter"
        
    return "planner"


def route_after_analysis(state: AgentState) -> Literal["planner", "interpreter"]:
    """
    Determines the next step after the analysis_node execution.
    This is the core of the error recovery cycle (Cyclic Logic).
    If an error occurs and the retry limit is not reached, it routes back to the planner.
    If successful or the retry limit is exceeded, it routes to the interpreter.
    """
    tool_result = state.get("tool_result", {})
    retry_count = state.get("retry_count", 0)
    
    # Check if the analysis step encountered an error
    if "error" in tool_result:
        # Prevent infinite loops by checking the retry counter
        if retry_count < 3:
            return "planner"
        else:
            # If we retried 3 times and still failed, give up and report to user
            return "interpreter"
            
    # If no error, proceed to generate the final answer
    return "interpreter"