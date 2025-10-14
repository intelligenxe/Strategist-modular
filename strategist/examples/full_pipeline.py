"""
Example: Complete End-to-End Pipeline
This script demonstrates the full pipeline from document loading to analysis
"""

import os
from rag_analysis import (
    Config,
    AnalysisWorkflow,
    quick_analysis,
    compare_topics,
    deep_dive_analysis
)

def example_1_quick_start():
    """Simplest way to run an analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Quick Start")
    print("=" * 70)
    
    # One-line analysis
    result = quick_analysis(
        topic="Impact of AI on software development",
        document_paths={
            'pdfs': ['./data/ai_report.pdf'],
            'txt': ['./data/notes.txt']
        }
    )
    
    print(result)


def example_2_comparative_analysis():
    """Compare two topics"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Comparative Analysis")
    print("=" * 70)
    
    result = compare_topics(
        topic1="Traditional Software Development",
        topic2="AI-Assisted Development",
        document_paths={
            'pdfs': ['./data/dev_practices.pdf'],
            'docx': ['./data/comparison.docx']
        }
    )
    
    print(result)


def example_3_deep_dive():
    """Deep dive with specific questions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Deep Dive Analysis")
    print("=" * 70)
    
    result = deep_dive_analysis(
        topic="Cloud Computing Adoption",
        questions=[
            "What are the main benefits of cloud adoption?",
            "What are the security concerns?",
            "What is the typical migration timeline?",
            "What are the cost implications?"
        ],
        document_paths={
            'pdfs': ['./data/cloud_strategy.pdf'],
            'urls': ['https://example.com/cloud-guide']
        }
    )
    
    print(result)


def example_4_full_workflow():
    """Complete workflow with full control"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Full Workflow with Custom Configuration")
    print("=" * 70)
    
    # Step 1: Custom configuration
    config = Config(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        verbose=True
    )
    
    # Customize settings
    config.index.chunk_size = 1024
    config.index.similarity_top_k = 7
    config.llm.temperature = 0.5
    
    # Step 2: Initialize workflow
    workflow = AnalysisWorkflow(config)
    
    # Step 3: Build knowledge base
    workflow.setup_knowledge_base(
        document_paths={
            'pdfs': [
                './data/report1.pdf',
                './data/report2.pdf'
            ],
            'docx': [
                './data/analysis.docx'
            ],
            'txt': [
                './data/notes.txt'
            ],
            'urls': [
                'https://example.com/article1',
                'https://example.com/article2'
            ]
        }
    )
    
    # Step 4: Setup crew
    workflow.setup_crew()
    
    # Step 5: Run multiple analyses
    
    # Basic analysis
    result1 = workflow.run_analysis(
        analysis_type='basic',
        parameters={'topic': 'Digital Transformation Strategies'}
    )
    
    # Summary
    result2 = workflow.run_analysis(
        analysis_type='summary',
        parameters={
            'topic': 'Key takeaways from all documents',
            'max_length': '1 page'
        }
    )
    
    print("\n--- Basic Analysis ---")
    print(result1)
    
    print("\n--- Summary ---")
    print(result2)


def example_5_reuse_knowledge_base():
    """Reuse an existing knowledge base for multiple analyses"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Reusing Knowledge Base")
    print("=" * 70)
    
    # First time: Build knowledge base
    print("\nBuilding knowledge base (do this once)...")
    workflow = AnalysisWorkflow()
    workflow.setup_knowledge_base(
        document_paths={
            'pdfs': ['./data/dataset.pdf']
        }
    )
    
    # Later: Reuse for different analyses
    print("\nRunning multiple analyses on same knowledge base...")
    
    from rag_analysis import analyze_with_existing_kb
    
    # Analysis 1
    result1 = analyze_with_existing_kb(
        analysis_type='basic',
        parameters={'topic': 'Market trends'}
    )
    
    # Analysis 2
    result2 = analyze_with_existing_kb(
        analysis_type='summary',
        parameters={'topic': 'Executive summary'}
    )
    
    print("Analysis 1:", result1)
    print("\nAnalysis 2:", result2)


def example_6_custom_crew():
    """Create a custom crew with specific agents"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Custom Crew Configuration")
    print("=" * 70)
    
    from rag_analysis import Config, AnalysisWorkflow
    from rag_analysis.agents import (
        create_research_agent,
        create_fact_checker_agent,
        create_strategic_advisor_agent
    )
    from rag_analysis.agents.tools import RAG_TOOLS
    from crewai import Task, Crew, Process
    from langchain_groq import ChatGroq
    
    config = Config.from_env()
    workflow = AnalysisWorkflow(config)
    
    # Setup knowledge base
    workflow.setup_knowledge_base(
        document_paths={'pdfs': ['./data/business_plan.pdf']}
    )
    
    # Create custom agents
    llm = ChatGroq(
        model=config.llm.model_name,
        api_key=config.groq_api_key
    )
    
    researcher = create_research_agent(llm, RAG_TOOLS)
    fact_checker = create_fact_checker_agent(llm, RAG_TOOLS)
    advisor = create_strategic_advisor_agent(llm, RAG_TOOLS)
    
    # Create custom tasks
    research_task = Task(
        description="Research our business strategy and market position",
        agent=researcher,
        expected_output="Detailed research findings"
    )
    
    verify_task = Task(
        description="Verify all claims made in the research",
        agent=fact_checker,
        expected_output="Fact-checked research with verified claims",
        context=[research_task]
    )
    
    strategy_task = Task(
        description="Provide strategic recommendations based on findings",
        agent=advisor,
        expected_output="Strategic recommendations",
        context=[research_task, verify_task]
    )
    
    # Create custom crew
    crew = Crew(
        agents=[researcher, fact_checker, advisor],
        tasks=[research_task, verify_task, strategy_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Run custom crew
    result = crew.kickoff()
    print(result)


def main():
    """Run all examples"""
    
    print("=" * 70)
    print("RAG ANALYSIS - COMPLETE PIPELINE EXAMPLES")
    print("=" * 70)
    
    # Choose which example to run
    # Uncomment the one you want to test
    
    # example_1_quick_start()
    # example_2_comparative_analysis()
    # example_3_deep_dive()
    # example_4_full_workflow()
    # example_5_reuse_knowledge_base()
    # example_6_custom_crew()
    
    # Or run a simple test
    print("\nRunning simple test...")
    print("Make sure to:")
    print("1. Set GROQ_API_KEY environment variable")
    print("2. Place documents in ./data/ directory")
    print("3. Uncomment the example you want to run above")
    
    # Simple test with minimal setup
    if os.getenv("GROQ_API_KEY"):
        print("\n✓ GROQ_API_KEY is set")
        print("✓ Ready to run examples")
        print("\nUncomment one of the examples above and run again!")
    else:
        print("\n⚠️  GROQ_API_KEY not set in environment")
        print("   Set it with: export GROQ_API_KEY='your_key_here'")


if __name__ == "__main__":
    main()
