"""
Part 3: Analysis Workflow Runner
Orchestrates the complete analysis workflow using knowledge base and agents
"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from ..config import Config, get_config
from ..knowledge_base.builder import KnowledgeBaseBuilder
from ..agents.crew_builder import CrewBuilder
from ..agents.tools import set_query_engine


class AnalysisWorkflow:
    """
    Orchestrates the complete analysis workflow:
    1. Load/build knowledge base
    2. Initialize agents and crews
    3. Run analysis
    4. Save results
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize analysis workflow
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.kb_builder: Optional[KnowledgeBaseBuilder] = None
        self.crew_builder: Optional[CrewBuilder] = None
        self.query_engine = None
    
    def setup_knowledge_base(
        self,
        document_paths: Optional[Dict[str, list]] = None,
        load_from_disk: bool = False,
        persist_dir: Optional[str] = None
    ):
        """
        Setup the knowledge base
        
        Args:
            document_paths: Dictionary with document paths
                          {'pdfs': [...], 'docx': [...], 'txt': [...], 'urls': [...]}
            load_from_disk: If True, load existing index instead of building
            persist_dir: Directory to load from (if load_from_disk=True)
        """
        if self.config.verbose:
            print("\n" + "="*70)
            print("STEP 1: Setting up Knowledge Base")
            print("="*70)
        
        self.kb_builder = KnowledgeBaseBuilder(self.config)
        
        if load_from_disk:
            # Load existing index
            self.kb_builder.load_index(persist_dir)
        else:
            # Build new index
            if document_paths:
                self.kb_builder.load_documents(
                    pdf_paths=document_paths.get('pdfs'),
                    docx_paths=document_paths.get('docx'),
                    txt_paths=document_paths.get('txt'),
                    urls=document_paths.get('urls')
                )
            else:
                raise ValueError(
                    "Either provide document_paths or set load_from_disk=True"
                )
            
            self.kb_builder.build_index(persist=True)
        
        # Get query engine and set it globally for tools
        self.query_engine = self.kb_builder.get_query_engine()
        set_query_engine(self.query_engine)
        
        if self.config.verbose:
            stats = self.kb_builder.get_stats()
            print(f"\n✓ Knowledge base ready")
            print(f"  Documents: {stats['total_documents']}")
            print(f"  Index built: {stats['index_built']}")
    
    def setup_crew(self):
        """Initialize the crew builder"""
        if self.config.verbose:
            print("\n" + "="*70)
            print("STEP 2: Setting up Analysis Crew")
            print("="*70)
        
        self.crew_builder = CrewBuilder(self.config)
        
        if self.config.verbose:
            print("✓ Crew builder initialized")
    
    def run_analysis(
        self,
        analysis_type: str,
        parameters: Dict[str, Any],
        save_report: bool = True
    ) -> str:
        """
        Run analysis using the specified crew type
        
        Args:
            analysis_type: Type of analysis ('basic', 'comparative', 'deep_dive', 'summary')
            parameters: Parameters for the analysis (varies by type)
            save_report: If True, save the report to disk
            
        Returns:
            Analysis result as string
        """
        if self.kb_builder is None or self.crew_builder is None:
            raise ValueError(
                "Knowledge base and crew not initialized. "
                "Call setup_knowledge_base() and setup_crew() first."
            )
        
        if self.config.verbose:
            print("\n" + "="*70)
            print(f"STEP 3: Running {analysis_type.upper()} Analysis")
            print("="*70)
        
        # Create appropriate crew based on analysis type
        if analysis_type == "basic":
            crew = self.crew_builder.create_basic_analysis_crew(
                research_topic=parameters['topic']
            )
        elif analysis_type == "comparative":
            crew = self.crew_builder.create_comparative_analysis_crew(
                topic1=parameters['topic1'],
                topic2=parameters['topic2']
            )
        elif analysis_type == "deep_dive":
            crew = self.crew_builder.create_deep_dive_crew(
                topic=parameters['topic'],
                specific_questions=parameters['questions']
            )
        elif analysis_type == "summary":
            crew = self.crew_builder.create_summary_crew(
                topic=parameters['topic'],
                max_length=parameters.get('max_length', '2 pages')
            )
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Run the crew
        if self.config.verbose:
            print(f"\n🚀 Starting analysis...")
        
        result = crew.kickoff()
        
        if self.config.verbose:
            print("\n✓ Analysis complete!")
        
        # Save report if requested
        if save_report:
            self._save_report(result, analysis_type, parameters)
        
        return str(result)
    
    def _save_report(
        self,
        result: Any,
        analysis_type: str,
        parameters: Dict[str, Any]
    ):
        """
        Save analysis report to disk
        
        Args:
            result: Analysis result
            analysis_type: Type of analysis
            parameters: Analysis parameters
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{analysis_type}_analysis_{timestamp}.md"
        filepath = self.config.storage.reports_dir / filename
        
        # Create report content
        report_content = f"""# {analysis_type.upper()} Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Parameters
{self._format_parameters(parameters)}

## Results

{str(result)}
"""
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if self.config.verbose:
            print(f"\n💾 Report saved to: {filepath}")
    
    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for report"""
        lines = []
        for key, value in parameters.items():
            if isinstance(value, list):
                lines.append(f"- **{key}**: {len(value)} items")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)
    
    def run_complete_workflow(
        self,
        document_paths: Dict[str, list],
        analysis_type: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Run the complete workflow in one call
        
        Args:
            document_paths: Dictionary with document paths
            analysis_type: Type of analysis to run
            parameters: Analysis parameters
            
        Returns:
            Analysis result as string
            
        Example:
            result = workflow.run_complete_workflow(
                document_paths={'pdfs': ['doc1.pdf', 'doc2.pdf']},
                analysis_type='basic',
                parameters={'topic': 'AI in healthcare'}
            )
        """
        self.setup_knowledge_base(document_paths=document_paths)
        self.setup_crew()
        result = self.run_analysis(analysis_type, parameters)
        return result


# Convenience functions for common workflows

def quick_analysis(
    topic: str,
    document_paths: Dict[str, list],
    config: Optional[Config] = None
) -> str:
    """
    Run a quick basic analysis
    
    Args:
        topic: Topic to analyze
        document_paths: Dictionary with document paths
        config: Configuration object
        
    Returns:
        Analysis result
    """
    workflow = AnalysisWorkflow(config)
    return workflow.run_complete_workflow(
        document_paths=document_paths,
        analysis_type='basic',
        parameters={'topic': topic}
    )


def compare_topics(
    topic1: str,
    topic2: str,
    document_paths: Dict[str, list],
    config: Optional[Config] = None
) -> str:
    """
    Run a comparative analysis
    
    Args:
        topic1: First topic
        topic2: Second topic
        document_paths: Dictionary with document paths
        config: Configuration object
        
    Returns:
        Analysis result
    """
    workflow = AnalysisWorkflow(config)
    return workflow.run_complete_workflow(
        document_paths=document_paths,
        analysis_type='comparative',
        parameters={'topic1': topic1, 'topic2': topic2}
    )


def deep_dive_analysis(
    topic: str,
    questions: list,
    document_paths: Dict[str, list],
    config: Optional[Config] = None
) -> str:
    """
    Run a deep-dive analysis with specific questions
    
    Args:
        topic: Main topic
        questions: List of specific questions
        document_paths: Dictionary with document paths
        config: Configuration object
        
    Returns:
        Analysis result
    """
    workflow = AnalysisWorkflow(config)
    return workflow.run_complete_workflow(
        document_paths=document_paths,
        analysis_type='deep_dive',
        parameters={'topic': topic, 'questions': questions}
    )


def analyze_with_existing_kb(
    analysis_type: str,
    parameters: Dict[str, Any],
    index_dir: Optional[str] = None,
    config: Optional[Config] = None
) -> str:
    """
    Run analysis using an existing knowledge base
    
    Args:
        analysis_type: Type of analysis
        parameters: Analysis parameters
        index_dir: Directory with existing index
        config: Configuration object
        
    Returns:
        Analysis result
    """
    workflow = AnalysisWorkflow(config)
    workflow.setup_knowledge_base(load_from_disk=True, persist_dir=index_dir)
    workflow.setup_crew()
    return workflow.run_analysis(analysis_type, parameters)
