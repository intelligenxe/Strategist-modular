
# ============================================================================
# File: rag_analysis/__init__.py
# Location: rag_analysis/__init__.py
# ============================================================================
"""
RAG Analysis Package
A modular system for document analysis using RAG and CrewAI agents
"""

__version__ = "0.1.0"

from .config import Config, get_config, set_config
from .knowledge_base.builder import KnowledgeBaseBuilder, create_knowledge_base
from .agents.crew_builder import CrewBuilder, create_analysis_crew
from .workflows.analyzer import (
    AnalysisWorkflow,
    quick_analysis,
    compare_topics,
    deep_dive_analysis,
    analyze_with_existing_kb
)

__all__ = [
    # Config
    'Config',
    'get_config',
    'set_config',
    
    # Knowledge Base
    'KnowledgeBaseBuilder',
    'create_knowledge_base',
    
    # Agents and Crews
    'CrewBuilder',
    'create_analysis_crew',
    
    # Workflows
    'AnalysisWorkflow',
    'quick_analysis',
    'compare_topics',
    'deep_dive_analysis',
    'analyze_with_existing_kb',
]

