"""
RAG tools for CrewAI agents
Provides the bridge between LlamaIndex query engine and CrewAI tools
"""

from typing import Optional
from crewai_tools import tool


# Global query engine reference (set by workflow)
_query_engine = None


def set_query_engine(query_engine):
    """
    Set the global query engine for RAG tools
    
    Args:
        query_engine: LlamaIndex query engine instance
    """
    global _query_engine
    _query_engine = query_engine


def get_query_engine():
    """Get the global query engine"""
    return _query_engine


@tool("Search Knowledge Base")
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    Use this tool whenever you need to retrieve context from the document collection.
    
    This tool performs semantic search across all indexed documents and returns
    the most relevant information based on the query.
    
    Args:
        query: The search query or question. Be specific and clear.
               Examples:
               - "What are the main findings about climate change?"
               - "Show me statistics about revenue growth"
               - "Explain the methodology used in the study"
    
    Returns:
        Relevant information from the knowledge base, including source citations
    """
    if _query_engine is None:
        return "Error: Knowledge base not initialized. Please build the knowledge base first."
    
    try:
        response = _query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool("Search Knowledge Base with Filters")
def search_with_filters(query: str, source_type: Optional[str] = None) -> str:
    """
    Search the knowledge base with metadata filters.
    Use this when you need to search specific types of documents.
    
    Args:
        query: The search query or question
        source_type: Optional filter for document type ('pdf', 'docx', 'txt', 'web')
    
    Returns:
        Relevant information filtered by the specified criteria
    """
    if _query_engine is None:
        return "Error: Knowledge base not initialized. Please build the knowledge base first."
    
    try:
        # Build metadata filters if provided
        filters = None
        if source_type:
            from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="type", value=source_type)]
            )
        
        response = _query_engine.query(query, filters=filters)
        return str(response)
    except Exception as e:
        return f"Error searching with filters: {str(e)}"


@tool("Get Document Sources")
def get_document_sources(query: str) -> str:
    """
    Search the knowledge base and return the sources of the retrieved documents.
    Use this when you need to know which specific documents contain relevant information.
    
    Args:
        query: The search query
    
    Returns:
        List of source documents that contain relevant information
    """
    if _query_engine is None:
        return "Error: Knowledge base not initialized."
    
    try:
        response = _query_engine.query(query)
        
        # Extract source nodes
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source = node.metadata.get('source', 'Unknown')
                score = node.score if hasattr(node, 'score') else 'N/A'
                sources.append(f"- {source} (relevance: {score})")
        
        if sources:
            return "Sources:\n" + "\n".join(sources)
        else:
            return "No source information available"
            
    except Exception as e:
        return f"Error retrieving sources: {str(e)}"


# Additional tool for statistical queries
@tool("Extract Statistics")
def extract_statistics(topic: str) -> str:
    """
    Extract numerical data and statistics about a topic from the knowledge base.
    Use this when you need specific numbers, percentages, or quantitative data.
    
    Args:
        topic: The topic to extract statistics about
    
    Returns:
        Statistical information and numerical data related to the topic
    """
    if _query_engine is None:
        return "Error: Knowledge base not initialized."
    
    query = f"What are the key statistics, numbers, percentages, and quantitative data about {topic}?"
    
    try:
        response = _query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error extracting statistics: {str(e)}"


@tool("Compare Topics")
def compare_topics(topic1: str, topic2: str) -> str:
    """
    Compare two topics based on information in the knowledge base.
    Use this when you need to analyze similarities and differences between topics.
    
    Args:
        topic1: First topic to compare
        topic2: Second topic to compare
    
    Returns:
        Comparison of the two topics based on available information
    """
    if _query_engine is None:
        return "Error: Knowledge base not initialized."
    
    query = f"Compare and contrast {topic1} and {topic2}. What are the key similarities and differences?"
    
    try:
        response = _query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error comparing topics: {str(e)}"


# Export all tools as a list for easy import
RAG_TOOLS = [
    search_knowledge_base,
    search_with_filters,
    get_document_sources,
    extract_statistics,
    compare_topics
]


# Tool categories for organization
BASIC_TOOLS = [search_knowledge_base]
ADVANCED_TOOLS = [search_with_filters, get_document_sources]
ANALYSIS_TOOLS = [extract_statistics, compare_topics]
