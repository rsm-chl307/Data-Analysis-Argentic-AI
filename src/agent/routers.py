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
    Checks for both top-level errors and nested tool-specific errors 
    to trigger the self-correction loop.
    """
    tool_result = state.get("tool_result", {})
    retry_count = state.get("retry_count", 0)
    
    # 1. Check for top-level errors (e.g., missing dataframe or target)
    has_error = "error" in tool_result
    
    # 2. Check for nested errors inside specific tool executions (Silent Failures)
    # Example: tool_result = {"correlation": {"error": {"message": "..."}}}
    if not has_error:
        for key, value in tool_result.items():
            if isinstance(value, dict) and "error" in value:
                has_error = True
                break
                
    # 3. Route based on the error state and retry limits
    if has_error:
        if retry_count < 3:
            return "planner"      # Trigger self-correction loop
        else:
            return "interpreter"  # Max retries reached, proceed to final report
            
    return "interpreter"          # Success, proceed to final report