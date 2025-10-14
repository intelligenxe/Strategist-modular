

# ============================================================================
# File: rag_analysis/workflows/__init__.py
# Location: rag_analysis/workflows/__init__.py
# ============================================================================
"""
Workflows Module
Orchestrates complete analysis workflows
"""

from .analyzer import (
    AnalysisWorkflow,
    quick_analysis,
    compare_topics,
    deep_dive_analysis,
    analyze_with_existing_kb
)

__all__ = [
    'AnalysisWorkflow',
    'quick_analysis',
    'compare_topics',
    'deep_dive_analysis',
    'analyze_with_existing_kb',
]


# ============================================================================
