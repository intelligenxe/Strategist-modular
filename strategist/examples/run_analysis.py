"""
Example: Running Analysis with Existing Knowledge Base
This script shows how to run various types of analysis using a pre-built knowledge base
"""

import os
from rag_analysis import AnalysisWorkflow, Config

def main():
    """Run analysis using existing knowledge base"""
    
    # Step 1: Configure
    config = Config(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        verbose=True
    )
    
    print("Running Analysis")
    print("=" * 70)
    
    # Step 2: Initialize workflow
    workflow = AnalysisWorkflow(config)
    
    # Step 3: Load existing knowledge base
    print("\nLoading knowledge base from disk...")
    workflow.setup_knowledge_base(load_from_disk=True)
    
    # Step 4: Setup analysis crew
    workflow.setup_crew()
    
    # Step 5: Run different types of analysis
    
    # ========================================================================
    # Example 1: Basic Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Basic Analysis")
    print("=" * 70)
    
    result = workflow.run_analysis(
        analysis_type='basic',
        parameters={
            'topic': 'Key trends and patterns in artificial intelligence'
        },
        save_report=True
    )
    
    print("\nResult:")
    print(result)
    
    # ========================================================================
    # Example 2: Comparative Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Comparative Analysis")
    print("=" * 70)
    
    result = workflow.run_analysis(
        analysis_type='comparative',
        parameters={
            'topic1': 'Machine Learning',
            'topic2': 'Deep Learning'
        },
        save_report=True
    )
    
    print("\nResult:")
    print(result)
    
    # ========================================================================
    # Example 3: Deep Dive Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Deep Dive Analysis")
    print("=" * 70)
    
    result = workflow.run_analysis(
        analysis_type='deep_dive',
        parameters={
            'topic': 'AI in Healthcare',
            'questions': [
                'What are the main applications of AI in healthcare?',
                'What are the challenges and limitations?',
                'What does the future hold?',
                'What are the ethical considerations?'
            ]
        },
        save_report=True
    )
    
    print("\nResult:")
    print(result)
    
    # ========================================================================
    # Example 4: Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 4: Quick Summary")
    print("=" * 70)
    
    result = workflow.run_analysis(
        analysis_type='summary',
        parameters={
            'topic': 'Overall findings from all documents',
            'max_length': '1 page'
        },
        save_report=True
    )
    
    print("\nResult:")
    print(result)
    
    print("\n" + "=" * 70)
    print("✓ All analyses complete!")
    print(f"Reports saved to: {config.storage.reports_dir}")


if __name__ == "__main__":
    main()
