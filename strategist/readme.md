# RAG Analysis System

A modular Python package for document analysis using RAG (Retrieval-Augmented Generation) and CrewAI agents. Built with LlamaIndex for superior document parsing and CrewAI for multi-agent orchestration.

## 🎯 Features

- **📚 Part 1: Advanced Document Parsing** (LlamaIndex)
  - Support for PDFs, DOCX, TXT, and web content
  - Complex document structures (tables, images, metadata)
  - 100+ data source connectors
  - Persistent vector index storage

- **🤖 Part 2: Multi-Agent Analysis** (CrewAI)
  - Pre-built agent templates (researcher, analyst, writer, critic, etc.)
  - Multiple analysis workflows (basic, comparative, deep-dive, summary)
  - Custom crew builder for specialized workflows
  - Seamless RAG integration via tools

- **⚡ Part 3: Complete Workflows**
  - End-to-end pipeline orchestration
  - Reusable knowledge bases
  - Automatic report generation
  - Fast inference with Groq

## 🏗️ Architecture

```
rag_analysis/
├── config.py                 # Configuration management
├── knowledge_base/          # Part 1: Document parsing & indexing
│   ├── builder.py
│   └── loaders.py
├── agents/                  # Part 2: CrewAI agents & crews
│   ├── crew_builder.py
│   ├── agents.py
│   └── tools.py
└── workflows/               # Part 3: Analysis orchestration
    └── analyzer.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag_analysis.git
cd rag_analysis

# Install the package
pip install -e .

# Set up environment
export GROQ_API_KEY='your_groq_api_key_here'
```

### Simple Usage

```python
from rag_analysis import quick_analysis

# One-line analysis
result = quick_analysis(
    topic="Impact of AI on healthcare",
    document_paths={
        'pdfs': ['./data/report1.pdf', './data/report2.pdf'],
        'txt': ['./data/notes.txt']
    }
)

print(result)
```

### Full Workflow

```python
from rag_analysis import AnalysisWorkflow, Config

# Configure
config = Config.from_env()

# Initialize workflow
workflow = AnalysisWorkflow(config)

# Build knowledge base (do once)
workflow.setup_knowledge_base(
    document_paths={
        'pdfs': ['./data/report1.pdf'],
        'docx': ['./data/analysis.docx'],
        'urls': ['https://example.com/article']
    }
)

# Setup agents
workflow.setup_crew()

# Run analysis
result = workflow.run_analysis(
    analysis_type='basic',
    parameters={'topic': 'Key findings and trends'}
)

print(result)
```

## 📖 Usage Examples

### 1. Build Knowledge Base Independently

```python
from rag_analysis import KnowledgeBaseBuilder

# Create and build
kb = KnowledgeBaseBuilder()
kb.load_from_directory('./data', recursive=True)
kb.build_index(persist=True)

# Later: Load existing index
kb.load_index('./storage/index')
query_engine = kb.get_query_engine()
```

### 2. Comparative Analysis

```python
from rag_analysis import compare_topics

result = compare_topics(
    topic1="Machine Learning",
    topic2="Deep Learning",
    document_paths={'pdfs': ['./data/ml_guide.pdf']}
)
```

### 3. Deep Dive with Specific Questions

```python
from rag_analysis import deep_dive_analysis

result = deep_dive_analysis(
    topic="Cloud Migration Strategy",
    questions=[
        "What are the main benefits?",
        "What are the risks?",
        "What is the typical timeline?",
        "What are the cost implications?"
    ],
    document_paths={'pdfs': ['./data/cloud_strategy.pdf']}
)
```

### 4. Custom Crew

```python
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
workflow.setup_knowledge_base(document_paths={'pdfs': ['./data/doc.pdf']})

# Create custom agents
llm = ChatGroq(model=config.llm.model_name, api_key=config.groq_api_key)
researcher = create_research_agent(llm, RAG_TOOLS)
advisor = create_strategic_advisor_agent(llm, RAG_TOOLS)

# Create custom tasks
research_task = Task(
    description="Research market trends",
    agent=researcher,
    expected_output="Market research findings"
)

strategy_task = Task(
    description="Provide strategic recommendations",
    agent=advisor,
    expected_output="Strategic plan",
    context=[research_task]
)

# Run custom crew
crew = Crew(
    agents=[researcher, advisor],
    tasks=[research_task, strategy_task],
    process=Process.sequential
)

result = crew.kickoff()
```

## 🔧 Configuration

```python
from rag_analysis import Config

config = Config(
    groq_api_key="your_key",
    verbose=True
)

# Customize embedding model
config.embedding.model_name = "BAAI/bge-large-en-v1.5"

# Customize LLM
config.llm.model_name = "llama-3.3-70b-versatile"
config.llm.temperature = 0.5

# Customize indexing
config.index.chunk_size = 1024
config.index.chunk_overlap = 100
config.index.similarity_top_k = 7

# Customize storage paths
config.storage.index_dir = Path("./custom_storage/index")
config.storage.reports_dir = Path("./custom_reports")
```

## 📊 Analysis Types

| Type | Description | Use Case |
|------|-------------|----------|
| **basic** | Research → Analyze → Report | General topic analysis |
| **comparative** | Compare two topics side-by-side | Feature comparison, A vs B |
| **deep_dive** | Answer specific questions in depth | Detailed investigation |
| **summary** | Concise overview | Executive summaries |
| **custom** | Build your own crew | Specialized workflows |

## 🛠️ Available Agents

- `create_research_agent` - Information retrieval specialist
- `create_data_analyst_agent` - Pattern and trend analysis
- `create_report_writer_agent` - Professional report creation
- `create_critic_agent` - Quality assurance and validation
- `create_summarizer_agent` - Concise summaries
- `create_fact_checker_agent` - Verify claims and sources
- `create_strategic_advisor_agent` - Strategic recommendations
- `create_domain_expert_agent` - Custom domain expertise

## 🔌 RAG Tools

Agents have access to these tools for querying the knowledge base:

- `search_knowledge_base` - Basic semantic search
- `search_with_filters` - Search with metadata filters
- `get_document_sources` - Retrieve source documents
- `extract_statistics` - Extract numerical data
- `compare_topics` - Compare two topics

## 📁 Project Structure

```
your_project/
├── rag_analysis/              # Main package
│   ├── __init__.py
│   ├── config.py
│   ├── knowledge_base/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── loaders.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── crew_builder.py
│   │   ├── agents.py
│   │   └── tools.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   └── analyzer.py
│   └── utils/
│       └── __init__.py
├── examples/
│   ├── build_knowledge_base.py
│   ├── run_analysis.py
│   └── full_pipeline.py
├── data/                      # Your documents
├── storage/                   # Persisted indices
├── reports/                   # Generated reports
├── setup.py
├── requirements.txt
├── README.md
└── .env                       # Environment variables
```

## 🔐 Environment Variables

Create a `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
VERBOSE=true
```

## 🎓 Advanced Usage

### Reusing Knowledge Base

Build once, analyze many times:

```python
from rag_analysis import analyze_with_existing_kb

# First time: build knowledge base
# (See build_knowledge_base.py example)

# Multiple analyses on same KB
result1 = analyze_with_existing_kb(
    analysis_type='basic',
    parameters={'topic': 'Topic 1'}
)

result2 = analyze_with_existing_kb(
    analysis_type='summary',
    parameters={'topic': 'Topic 2'}
)
```

### Custom Document Loader

```python
from rag_analysis.knowledge_base.loaders import CustomDocumentLoader
from llama_index.core import Document

class JSONDocumentLoader(CustomDocumentLoader):
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.json')
    
    def load(self, file_path: str) -> list:
        import json
        with open(file_path) as f:
            data = json.load(f)
        
        doc = Document(
            text=json.dumps(data, indent=2),
            metadata={"source": file_path, "type": "json"}
        )
        return [doc]

# Use custom loader
from rag_analysis import KnowledgeBaseBuilder
kb = KnowledgeBaseBuilder()
loader = JSONDocumentLoader(kb.config)
docs = loader.load('./data/config.json')
kb.add_documents(docs)
```

### Batch Processing

```python
from rag_analysis import AnalysisWorkflow
import os

workflow = AnalysisWorkflow()
workflow.setup_knowledge_base(load_from_disk=True)
workflow.setup_crew()

topics = [
    "AI in Finance",
    "AI in Healthcare", 
    "AI in Education"
]

for topic in topics:
    print(f"\nAnalyzing: {topic}")
    result = workflow.run_analysis(
        analysis_type='basic',
        parameters={'topic': topic},
        save_report=True
    )
```

## 🚀 Deployment on VPS

### Installation

```bash
# On your Ubuntu VPS
sudo apt update && sudo apt install python3 python3-pip python3-venv -y

# Clone and setup
git clone https://github.com/yourusername/rag_analysis.git
cd rag_analysis
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Set environment
export GROQ_API_KEY='your_key_here'
```

### Create Systemd Service

```bash
sudo nano /etc/systemd/system/rag-analysis.service
```

```ini
[Unit]
Description=RAG Analysis Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/rag_analysis
Environment="PATH=/home/your_username/rag_analysis/venv/bin"
Environment="GROQ_API_KEY=your_groq_key"
ExecStart=/home/your_username/rag_analysis/venv/bin/python examples/run_analysis.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-analysis
sudo systemctl start rag-analysis
sudo systemctl status rag-analysis
```

## 📊 Performance Tips

1. **Index Persistence**: Build index once, reuse many times
2. **Chunk Size**: Larger chunks (1024) for context, smaller (256) for precision
3. **Top-K**: Start with 3-5, increase for comprehensive analysis
4. **Caching**: LlamaIndex automatically caches embeddings
5. **Groq Speed**: ~100 tokens/sec, much faster than alternatives

## 🐛 Troubleshooting

### "Knowledge base not initialized"
```python
# Make sure to set query engine
from rag_analysis.agents.tools import set_query_engine
set_query_engine(your_query_engine)
```

### "GROQ_API_KEY not found"
```bash
export GROQ_API_KEY='your_key_here'
# Or add to .env file
```

### "No module named 'rag_analysis'"
```bash
# Install in development mode
pip install -e .
```

### Out of Memory
```python
# Reduce chunk size and top-k
config.index.chunk_size = 256
config.index.similarity_top_k = 3
```

## 🧪 Testing

```bash
# Run examples
python examples/build_knowledge_base.py
python examples/run_analysis.py
python examples/full_pipeline.py

# Run with custom config
GROQ_API_KEY=your_key python examples/full_pipeline.py
```

## 📝 Best Practices

1. **Build Once**: Create knowledge base once, reuse for multiple analyses
2. **Modular Design**: Use individual modules independently
3. **Custom Agents**: Create domain-specific agents for specialized tasks
4. **Metadata**: Add rich metadata to documents for better filtering
5. **Validation**: Use critic agents to validate analysis quality
6. **Reports**: Always save reports for audit trail

## 🔄 Workflow Patterns

### Pattern 1: Daily Analysis
```python
# Build KB once per day
kb_builder.load_from_directory('./incoming_docs')
kb_builder.build_index()

# Run multiple analyses
for topic in daily_topics:
    analyze_with_existing_kb('basic', {'topic': topic})
```

### Pattern 2: Comparative Studies
```python
# Compare multiple items
items = ['Product A', 'Product B', 'Product C']
for i, item1 in enumerate(items):
    for item2 in items[i+1:]:
        compare_topics(item1, item2, document_paths)
```

### Pattern 3: Incremental Updates
```python
# Load existing KB
kb = KnowledgeBaseBuilder()
kb.load_index()

# Add new documents
kb.load_documents(pdf_paths=['./new_doc.pdf'])
kb.build_index()  # Rebuilds with new docs
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional document loaders (Excel, CSV, JSON)
- New agent templates
- More analysis workflows
- Performance optimizations
- Better error handling
- Unit tests

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) - Document parsing and RAG
- [CrewAI](https://www.crewai.io/) - Multi-agent framework
- [Groq](https://groq.com/) - Fast LLM inference
- [LangChain](https://www.langchain.com/) - LLM orchestration

## 📞 Support

- Documentation: [https://github.com/yourusername/rag_analysis/wiki](https://github.com/yourusername/rag_analysis/wiki)
- Issues: [https://github.com/yourusername/rag_analysis/issues](https://github.com/yourusername/rag_analysis/issues)
- Discussions: [https://github.com/yourusername/rag_analysis/discussions](https://github.com/yourusername/rag_analysis/discussions)

## 🗺️ Roadmap

- [ ] Add support for more LLM providers (OpenAI, Anthropic, Local models)
- [ ] Implement caching layer for faster repeated queries
- [ ] Add web UI with Gradio/Streamlit
- [ ] Support for multilingual documents
- [ ] Real-time document monitoring and auto-indexing
- [ ] Enhanced metadata extraction
- [ ] Integration with vector databases (Pinecone, Weaviate, Chroma)
- [ ] Batch processing utilities
- [ ] Cost tracking and optimization

---

Made with ❤️ using LlamaIndex + CrewAI + Groq
