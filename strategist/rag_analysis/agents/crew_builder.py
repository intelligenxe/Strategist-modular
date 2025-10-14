"""
Part 2: CrewAI Agent and Crew Builder
Defines agents and creates crews for different analysis workflows
"""

from typing import List, Optional
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

from ..config import Config, get_config
from .tools import RAG_TOOLS, BASIC_TOOLS, ADVANCED_TOOLS, ANALYSIS_TOOLS
from .agents import (
    create_research_agent,
    create_data_analyst_agent,
    create_report_writer_agent,
    create_critic_agent,
    create_summarizer_agent
)


class CrewBuilder:
    """
    Builds CrewAI crews for different analysis workflows
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize crew builder
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.llm = ChatGroq(
            model=self.config.llm.model_name,
            api_key=self.config.groq_api_key,
            temperature=self.config.llm.temperature
        )
    
    def create_basic_analysis_crew(
        self, 
        research_topic: str,
        tools: Optional[List] = None
    ) -> Crew:
        """
        Create a basic analysis crew: Research -> Analyze -> Report
        
        Args:
            research_topic: Topic to research and analyze
            tools: Tools to provide to agents. If None, uses all RAG tools
            
        Returns:
            Configured Crew instance
        """
        tools = tools or RAG_TOOLS
        
        # Create agents
        researcher = create_research_agent(self.llm, tools)
        analyst = create_data_analyst_agent(self.llm, tools)
        writer = create_report_writer_agent(self.llm, tools)
        
        # Create tasks
        research_task = Task(
            description=f"""Research the following topic using the knowledge base:
            
            Topic: {research_topic}
            
            Find all relevant information, key facts, important details, and context.
            Be thorough and comprehensive in your research.""",
            agent=researcher,
            expected_output="Comprehensive research findings with key information and context"
        )
        
        analysis_task = Task(
            description="""Analyze the research findings to identify:
            - Key patterns and trends
            - Important insights and implications
            - Critical findings that require attention
            - Relationships between different concepts
            
            Support your analysis with evidence from the knowledge base.""",
            agent=analyst,
            expected_output="Detailed analysis with insights, patterns, and evidence-based conclusions",
            context=[research_task]
        )
        
        report_task = Task(
            description="""Create a comprehensive report that includes:
            
            1. Executive Summary
            2. Research Findings
            3. Detailed Analysis
            4. Key Insights
            5. Conclusions and Recommendations
            
            The report should be well-structured, professional, and actionable.""",
            agent=writer,
            expected_output="Professional report with all required sections",
            context=[research_task, analysis_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, report_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        return crew
    
    def create_comparative_analysis_crew(
        self,
        topic1: str,
        topic2: str,
        tools: Optional[List] = None
    ) -> Crew:
        """
        Create a crew for comparing two topics
        
        Args:
            topic1: First topic to compare
            topic2: Second topic to compare
            tools: Tools to provide to agents
            
        Returns:
            Configured Crew instance
        """
        tools = tools or RAG_TOOLS
        
        researcher = create_research_agent(self.llm, tools)
        analyst = create_data_analyst_agent(self.llm, tools)
        writer = create_report_writer_agent(self.llm, tools)
        
        # Research both topics
        research_task_1 = Task(
            description=f"Research everything about: {topic1}",
            agent=researcher,
            expected_output=f"Comprehensive information about {topic1}"
        )
        
        research_task_2 = Task(
            description=f"Research everything about: {topic2}",
            agent=researcher,
            expected_output=f"Comprehensive information about {topic2}"
        )
        
        comparison_task = Task(
            description=f"""Compare {topic1} and {topic2} by analyzing:
            - Similarities and differences
            - Strengths and weaknesses of each
            - Use cases and applications
            - Performance metrics (if available)
            - Trade-offs and considerations
            
            Provide an objective, balanced comparison.""",
            agent=analyst,
            expected_output="Detailed comparative analysis",
            context=[research_task_1, research_task_2]
        )
        
        report_task = Task(
            description=f"""Create a comparative analysis report for {topic1} vs {topic2}:
            
            1. Executive Summary
            2. Overview of Each Topic
            3. Comparative Analysis
            4. Pros and Cons
            5. Recommendations
            
            Include tables or structured comparisons where appropriate.""",
            agent=writer,
            expected_output="Professional comparative analysis report",
            context=[research_task_1, research_task_2, comparison_task]
        )
        
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task_1, research_task_2, comparison_task, report_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        return crew
    
    def create_deep_dive_crew(
        self,
        topic: str,
        specific_questions: List[str],
        tools: Optional[List] = None
    ) -> Crew:
        """
        Create a crew for deep-dive analysis with specific questions
        
        Args:
            topic: Main topic to analyze
            specific_questions: List of specific questions to answer
            tools: Tools to provide to agents
            
        Returns:
            Configured Crew instance
        """
        tools = tools or RAG_TOOLS
        
        researcher = create_research_agent(self.llm, tools)
        analyst = create_data_analyst_agent(self.llm, tools)
        critic = create_critic_agent(self.llm, tools)
        writer = create_report_writer_agent(self.llm, tools)
        
        # Initial research
        research_task = Task(
            description=f"""Conduct comprehensive research on: {topic}
            
            Focus on gathering information relevant to these questions:
            {chr(10).join([f'- {q}' for q in specific_questions])}""",
            agent=researcher,
            expected_output="Comprehensive research findings"
        )
        
        # Answer each question
        question_tasks = []
        for i, question in enumerate(specific_questions):
            task = Task(
                description=f"""Answer this specific question using the knowledge base:
                
                Question: {question}
                
                Provide a detailed, evidence-based answer with citations.""",
                agent=analyst,
                expected_output=f"Detailed answer to: {question}",
                context=[research_task]
            )
            question_tasks.append(task)
        
        # Critical review
        review_task = Task(
            description="""Review all the answers provided and:
            - Identify any gaps or inconsistencies
            - Check if answers are well-supported by evidence
            - Suggest areas that need more investigation
            - Validate the conclusions""",
            agent=critic,
            expected_output="Critical review with identified gaps and validation",
            context=[research_task] + question_tasks
        )
        
        # Final report
        report_task = Task(
            description=f"""Create a comprehensive deep-dive report on {topic}:
            
            1. Executive Summary
            2. Background and Context
            3. Detailed Answers to Each Question
            4. Analysis and Insights
            5. Limitations and Gaps
            6. Conclusions
            
            Ensure all claims are supported by evidence from the knowledge base.""",
            agent=writer,
            expected_output="Comprehensive deep-dive analysis report",
            context=[research_task] + question_tasks + [review_task]
        )
        
        crew = Crew(
            agents=[researcher, analyst, critic, writer],
            tasks=[research_task] + question_tasks + [review_task, report_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        return crew
    
    def create_summary_crew(
        self,
        topic: str,
        max_length: str = "2 pages",
        tools: Optional[List] = None
    ) -> Crew:
        """
        Create a crew for generating concise summaries
        
        Args:
            topic: Topic to summarize
            max_length: Maximum length for summary
            tools: Tools to provide to agents
            
        Returns:
            Configured Crew instance
        """
        tools = tools or BASIC_TOOLS
        
        researcher = create_research_agent(self.llm, tools)
        summarizer = create_summarizer_agent(self.llm, tools)
        
        research_task = Task(
            description=f"Research all information about: {topic}",
            agent=researcher,
            expected_output="Complete information about the topic"
        )
        
        summary_task = Task(
            description=f"""Create a concise summary (max {max_length}) covering:
            - Most important points
            - Key takeaways
            - Critical insights
            
            Be concise but comprehensive.""",
            agent=summarizer,
            expected_output=f"Concise summary (max {max_length})",
            context=[research_task]
        )
        
        crew = Crew(
            agents=[researcher, summarizer],
            tasks=[research_task, summary_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        return crew
    
    def create_custom_crew(
        self,
        agents: List[Agent],
        tasks: List[Task],
        process: Process = Process.sequential
    ) -> Crew:
        """
        Create a custom crew with specified agents and tasks
        
        Args:
            agents: List of Agent instances
            tasks: List of Task instances
            process: Process type (sequential or hierarchical)
            
        Returns:
            Configured Crew instance
        """
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=process,
            verbose=self.config.verbose
        )
        
        return crew


def create_analysis_crew(
    research_topic: str,
    config: Optional[Config] = None,
    crew_type: str = "basic"
) -> Crew:
    """
    Convenience function to create an analysis crew
    
    Args:
        research_topic: Topic to analyze
        config: Configuration object
        crew_type: Type of crew ('basic', 'summary', etc.)
        
    Returns:
        Configured Crew instance
        
    Example:
        crew = create_analysis_crew(
            "Impact of AI on healthcare",
            crew_type="basic"
        )
    """
    builder = CrewBuilder(config)
    
    if crew_type == "basic":
        return builder.create_basic_analysis_crew(research_topic)
    elif crew_type == "summary":
        return builder.create_summary_crew(research_topic)
    else:
        raise ValueError(f"Unknown crew type: {crew_type}")
