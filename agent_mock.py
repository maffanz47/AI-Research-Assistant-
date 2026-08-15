"""
agent_mock.py
=============
Alias wrapper for agent_engine.py to maintain backward compatibility.
"""

from agent_engine import (
    ResearchGapAgent,
    AgentResult,
    ReActStep,
    fetch_graph,
    get_branch_summary,
    retrieve_text_chunks,
    calculate_abandonment_ratio,
    calculate_cluster_abandonment_ratios,
)

__all__ = [
    "ResearchGapAgent",
    "AgentResult",
    "ReActStep",
    "fetch_graph",
    "get_branch_summary",
    "retrieve_text_chunks",
    "calculate_abandonment_ratio",
    "calculate_cluster_abandonment_ratios",
]
