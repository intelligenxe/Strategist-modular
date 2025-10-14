

# ============================================================================
# File: rag_analysis/agents/__init__.py
# Location: rag_analysis/agents/__init__.py
# ============================================================================
"""
Agents Module
Defines CrewAI agents and crews for analysis
"""

from .crew_builder import CrewBuilder, create_analysis_crew
from .agents import (
    create_research_agent,
    create_data_analyst_agent,
    create_report_writer_agent,
    create_critic_agent,
    create_summarizer_agent,
    create_domain_expert_agent,
    create_fact_checker_agent,
    create_strategic_advisor_agent,
    create_agent_from_template,
    AGENT_TEMPLATES
)
from .tools import (
    search_knowledge_base,
    search_with_filters,
    get_document_sources,
    extract_statistics,
    compare_topics as compare_topics_tool,
    RAG_TOOLS,
    BASIC_TOOLS,
    ADVANCED_TOOLS,
    ANALYSIS_TOOLS,
    set_query_engine,
    get_query_engine
)

__all__ = [
    # Crew Builder
    'CrewBuilder',
    'create_analysis_crew',
    
    # Agent Creators
    'create_research_agent',
    'create_data_analyst_agent',
    'create_report_writer_agent',
    'create_critic_agent',
    'create_summarizer_agent',
    'create_domain_expert_agent',
    'create_fact_checker_agent',
    'create_strategic_advisor_agent',
    'create_agent_from_template',
    'AGENT_TEMPLATES',
    
    # Tools
    'search_knowledge_base',
    'search_with_filters',
    'get_document_sources',
    'extract_statistics',
    'compare_topics_tool',
    'RAG_TOOLS',
    'BASIC_TOOLS',
    'ADVANCED_TOOLS',
    'ANALYSIS_TOOLS',
    'set_query_engine',
    'get_query_engine',
]


# ============================================================================
