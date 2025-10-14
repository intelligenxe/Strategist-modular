"""
Individual agent definitions for CrewAI
Each function creates a specialized agent with specific role and capabilities
"""

from typing import List
from crewai import Agent


def create_research_agent(llm, tools: List) -> Agent:
    """
    Create a research analyst agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Research Analyst",
        goal="Extract and synthesize relevant information from the knowledge base",
        backstory="""You are an expert research analyst with 15 years of experience 
        in information retrieval and synthesis. You have a PhD in Information Science 
        and excel at finding relevant information across large document collections. 
        You are thorough, methodical, and always cite your sources. You know how to 
        ask the right questions to get the most relevant information.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_data_analyst_agent(llm, tools: List) -> Agent:
    """
    Create a data analyst agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Data Analyst",
        goal="Analyze information to identify patterns, trends, and actionable insights",
        backstory="""You are a skilled data analyst with expertise in quantitative 
        and qualitative analysis. You have a Master's degree in Data Science and 
        10 years of experience analyzing complex datasets and documents. You excel 
        at identifying patterns, extracting insights, and presenting findings clearly. 
        You always support your analysis with evidence and consider multiple perspectives.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_report_writer_agent(llm, tools: List) -> Agent:
    """
    Create a report writing agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Report Writer",
        goal="Create clear, comprehensive, and well-structured reports",
        backstory="""You are an expert technical writer with 12 years of experience 
        creating professional reports, white papers, and documentation. You have a 
        talent for synthesizing complex information into clear, compelling narratives. 
        Your reports are known for their clarity, structure, and actionable recommendations. 
        You always ensure proper citations and maintain a professional tone.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_critic_agent(llm, tools: List) -> Agent:
    """
    Create a critical reviewer agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Critical Reviewer",
        goal="Identify gaps, validate findings, and ensure quality",
        backstory="""You are a meticulous quality assurance specialist with 20 years 
        of experience in research validation and peer review. You have published over 
        50 peer-reviewed papers and served on editorial boards of major journals. 
        You excel at identifying logical flaws, missing evidence, and unsupported claims. 
        You are constructive in your criticism and always suggest improvements.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_summarizer_agent(llm, tools: List) -> Agent:
    """
    Create a summarization agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Information Summarizer",
        goal="Create concise, accurate summaries that capture essential information",
        backstory="""You are an expert at distilling complex information into clear, 
        concise summaries. With 8 years of experience as an executive briefing specialist, 
        you know how to identify the most important points and present them efficiently. 
        Your summaries are prized for their clarity and completeness despite their brevity. 
        You never sacrifice accuracy for conciseness.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_domain_expert_agent(
    llm, 
    tools: List,
    domain: str,
    expertise_description: str
) -> Agent:
    """
    Create a domain-specific expert agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        domain: The domain of expertise (e.g., "Healthcare", "Finance")
        expertise_description: Detailed description of the agent's expertise
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role=f"{domain} Expert",
        goal=f"Provide expert analysis and insights in {domain}",
        backstory=f"""You are a recognized expert in {domain}. {expertise_description}
        You combine deep domain knowledge with analytical skills to provide valuable 
        insights. You always consider industry-specific context and best practices 
        in your analysis.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_fact_checker_agent(llm, tools: List) -> Agent:
    """
    Create a fact-checking agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Fact Checker",
        goal="Verify claims and ensure all statements are supported by evidence",
        backstory="""You are a professional fact-checker with 10 years of experience 
        in journalism and research verification. You are trained to identify unsupported 
        claims, check sources, and verify the accuracy of statements. You are thorough, 
        objective, and never make assumptions. You always trace claims back to primary 
        sources when possible.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_strategic_advisor_agent(llm, tools: List) -> Agent:
    """
    Create a strategic advisor agent
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Strategic Advisor",
        goal="Provide strategic recommendations based on analysis",
        backstory="""You are a senior strategic consultant with 18 years of experience 
        advising Fortune 500 companies. You have an MBA from a top business school and 
        have led numerous successful strategic initiatives. You excel at translating 
        analysis into actionable recommendations. You consider risks, opportunities, 
        and implementation feasibility in your advice.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# Agent templates for common use cases
AGENT_TEMPLATES = {
    "researcher": create_research_agent,
    "analyst": create_data_analyst_agent,
    "writer": create_report_writer_agent,
    "critic": create_critic_agent,
    "summarizer": create_summarizer_agent,
    "fact_checker": create_fact_checker_agent,
    "strategic_advisor": create_strategic_advisor_agent
}


def create_agent_from_template(
    template_name: str,
    llm,
    tools: List
) -> Agent:
    """
    Create an agent from a template
    
    Args:
        template_name: Name of the template
        llm: Language model instance
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
        
    Example:
        agent = create_agent_from_template("researcher", llm, tools)
    """
    if template_name not in AGENT_TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available templates: {list(AGENT_TEMPLATES.keys())}"
        )
    
    return AGENT_TEMPLATES[template_name](llm, tools)
