# Quick Start Guide

Get up and running with RAG Analysis in 5 minutes!

## 🚀 Installation

### 1. Prerequisites

```bash
# Python 3.9 or higher
python3 --version

# Git
git --version
```

### 2. Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/rag_analysis.git
cd rag_analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### 3. Get Groq API Key

1. Go to https://console.groq.com/
2. Sign up (free)
3. Create an API key
4. Copy the key

### 4. Configure Environment

```bash
# Create .env file
cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
EOF

# Or export directly
export GROQ_API_KEY='your_groq_api_key_here'
```

## 📚 Prepare Your Documents

```bash
# Create data directory
mkdir -p data

# Add your documents
cp /path/to/your/documents/*.pdf data/
cp /path/to/your/documents/*.docx data/
cp /path/to/your/documents/*.txt data/
```

## 🎯 Run Your First Analysis

### Option 1: One-Line Quick Analysis

Create `my_analysis.py`:

```python
from rag_analysis import quick_analysis

result = quick_analysis(
    topic="What are the main findings?",
    document_paths={
        'pdfs': ['./data/report.pdf'],
        'txt': ['./data/notes.txt']
    }
)

print(result)
```

Run it:

```bash
python my_analysis.py
```

### Option 2: Use Example Scripts

```bash
# Build knowledge base
python examples/build_knowledge_base.py

# Run analysis
python examples/run_analysis.py

# See full pipeline
python examples/full_pipeline.py
```

## 📖 Common Use Cases

### Use Case 1: Analyze Research Papers

```python
from rag_analysis import deep_dive_analysis

result = deep_dive_analysis(
    topic="Machine Learning in Healthcare",
    questions=[
        "What are the main applications?",
        "What are the challenges?",
        "What does the future hold?"
    ],
    document_paths={
        'pdfs': ['./data/research_paper.pdf']
    }
)

print(result)
```

### Use Case 2: Compare Two Options

```python
from rag_analysis import compare_topics

result = compare_topics(
    topic1="Cloud Provider A",
    topic2="Cloud Provider B",
    document_paths={
        'pdfs': [
            './data/provider_a_specs.pdf',
            './data/provider_b_specs.pdf'
        ]
    }
)

print(result)
```

### Use Case 3: Generate Executive Summary

```python
from rag_analysis import AnalysisWorkflow

workflow = AnalysisWorkflow()
workflow.setup_knowledge_base(
    document_paths={'pdfs': ['./data/quarterly_report.pdf']}
)
workflow.setup_crew()

result = workflow.run_analysis(
    analysis_type='summary',
    parameters={
        'topic': 'Q4 2024 Performance',
        'max_length': '1 page'
    }
)

print(result)
```

## 🔄 Complete Workflow

```python
from rag_analysis import AnalysisWorkflow, Config
import os

# 1. Configure
config = Config(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    verbose=True
)

# 2. Initialize
workflow = AnalysisWorkflow(config)

# 3. Build Knowledge Base (do once)
workflow.setup_knowledge_base(
    document_paths={
        'pdfs': ['./data/doc1.pdf', './data/doc2.pdf'],
        'docx': ['./data/analysis.docx'],
        'txt': ['./data/notes.txt']
    }
)

# 4. Setup Crew
workflow.setup_crew()

# 5. Run Analysis
result = workflow.run_analysis(
    analysis_type='basic',
    parameters={'topic': 'Key insights and recommendations'}
)

print(result)

# Find report in ./reports/ directory
```

## 📁 Directory Structure After Setup

```
rag_analysis/
├── data/                    # Your documents
│   ├── report.pdf
│   ├── analysis.docx
│   └── notes.txt
├── storage/                 # Auto-created
│   └── index/              # Persisted vector index
├── reports/                 # Auto-created
│   └── basic_analysis_20241011_143052.md
├── examples/
├── rag_analysis/
└── .env
```

## 🎨 Customization

### Change LLM Model

```python
config = Config.from_env()
config.llm.model_name = "mixtral-8x7b-32768"
config.llm.temperature = 0.5
```

### Change Embedding Model

```python
config.embedding.model_name = "BAAI/bge-large-en-v1.5"
```

### Adjust Retrieval

```python
config.index.chunk_size = 1024
config.index.chunk_overlap = 100
config.index.similarity_top_k = 7
```

## 🐛 Troubleshooting

### Error: "GROQ_API_KEY not found"

```bash
# Check if set
echo $GROQ_API_KEY

# Set it
export GROQ_API_KEY='your_key_here'

# Or use .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### Error: "No documents loaded"

```bash
# Check data directory
ls -la data/

# Verify file formats (PDF, DOCX, TXT)
file data/*
```

### Error: "Module not found"

```bash
# Reinstall package
pip install -e .

# Check installation
pip list | grep rag-analysis
```

### Slow Performance

```python
# Reduce chunk size and top-k
config.index.chunk_size = 256
config.index.similarity_top_k = 3
```

## 📊 Verify Installation

```python
# test_installation.py
import sys

try:
    from rag_analysis import Config, KnowledgeBaseBuilder
    print("✓ Package imported successfully")
    
    config = Config.from_env()
    print("✓ Config loaded successfully")
    
    kb = KnowledgeBaseBuilder(config)
    print("✓ Knowledge base builder initialized")
    
    print("\n✅ Installation verified!")
    print(f"   Using: {config.llm.model_name}")
    print(f"   Embedding: {config.embedding.model_name}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
```

```bash
python test_installation.py
```

## 🎓 Next Steps

1. **Read the README**: Full documentation in `README.md`
2. **Explore Examples**: Check `examples/` directory
3. **Customize Agents**: See `rag_analysis/agents/agents.py`
4. **Create Custom Workflows**: Extend `rag_analysis/workflows/analyzer.py`

## 💡 Tips

1. **Build Once, Use Many**: Build your knowledge base once, then run multiple analyses
2. **Start Simple**: Use `quick_analysis()` first, then move to full workflows
3. **Check Reports**: All analyses are saved to `./reports/` directory
4. **Monitor Costs**: Groq free tier is generous (30 req/min, 14.4k req/day)
5. **Use Persistence**: Knowledge base is saved to disk automatically

## 📞 Get Help

- Read full docs: `README.md`
- Check examples: `examples/`
- Open issue: [GitHub Issues](https://github.com/yourusername/rag_analysis/issues)

---

**Ready to go!** Start with `quick_analysis()` and explore from there. 🚀
