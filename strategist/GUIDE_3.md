# Quick Reference & Glossary

A condensed reference guide for common commands, code snippets, and terminology.

---

## Quick Reference

### File Locations

```
rag_analysis/
├── config.py                      # ← Configuration (centralized)
├── knowledge_base/
│   ├── __init__.py
│   ├── builder.py                 # ← Part 1: Build KB
│   └── loaders.py                 # ← Part 1: Load documents
├── agents/
│   ├── __init__.py
│   ├── crew_builder.py            # ← Part 2: Create crews
│   ├── agents.py                  # ← Part 2: Agent definitions
│   └── tools.py                   # ← Bridge: RAG tools
├── workflows/
│   ├── __init__.py
│   └── analyzer.py                # ← Part 3: Orchestration
└── utils/
    └── __init__.py
```

### Quick Command Reference

```bash
# Installation
pip install -e .

# Environment setup
export GROQ_API_KEY='your_groq_api_key_here'

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install from requirements
pip install -r requirements.txt

# Run examples
python examples/build_knowledge_base.py
python examples/run_analysis.py
python examples/full_pipeline.py

# Test setup
python test_installation.py

# Deploy to VPS
scp -r rag_analysis/ user@vps:/home/user/
ssh user@vps 'cd ~/rag_analysis && pip install -e .'

# View VPS logs
ssh user@vps 'sudo journalctl -u rag-analysis -f'

# Restart VPS service
ssh user@vps 'sudo systemctl restart rag-analysis'

# Check VPS status
ssh user@vps 'sudo systemctl status rag-analysis'
```

### Quick Code Reference

#### One-Line Analysis
```python
from rag_analysis import quick_analysis

result = quick_analysis(
    "What are the main trends?",
    {'pdfs': ['./data/report.pdf']}
)
print(result)
```

#### Build Knowledge Base
```python
from rag_analysis import KnowledgeBaseBuilder

kb = KnowledgeBaseBuilder()
kb.load_from_directory('./data', recursive=True)
kb.build_index(persist=True)
```

#### Load Existing Knowledge Base
```python
from rag_analysis import KnowledgeBaseBuilder

kb = KnowledgeBaseBuilder()
kb.load_index('./storage/index')
query_engine = kb.get_query_engine()
```

#### Compare Two Topics
```python
from rag_analysis import compare_topics

result = compare_topics(
    "Machine Learning",
    "Deep Learning",
    {'pdfs': ['./data/ml_guide.pdf']}
)
print(result)
```

#### Deep Dive Analysis
```python
from rag_analysis import deep_dive_analysis

result = deep_dive_analysis(
    "Cloud Computing",
    [
        "What are the main benefits?",
        "What are the security concerns?",
        "What is the typical migration timeline?",
        "What are the cost implications?"
    ],
    {'pdfs': ['./data/cloud_strategy.pdf']}
)
print(result)
```

#### Full Workflow with Control
```python
from rag_analysis import AnalysisWorkflow, Config

config = Config.from_env()
workflow = AnalysisWorkflow(config)

# Build KB
workflow.setup_knowledge_base(
    document_paths={
        'pdfs': ['./data/doc1.pdf', './data/doc2.pdf'],
        'docx': ['./data/analysis.docx'],
        'txt': ['./data/notes.txt']
    }
)

# Setup agents
workflow.setup_crew()

# Run analysis
result = workflow.run_analysis(
    analysis_type='basic',
    parameters={'topic': 'Key insights'}
)

print(result)
```

#### Reuse Existing Knowledge Base
```python
from rag_analysis import analyze_with_existing_kb

# Multiple analyses on same KB
result1 = analyze_with_existing_kb(
    'basic',
    {'topic': 'Topic 1'}
)

result2 = analyze_with_existing_kb(
    'summary',
    {'topic': 'Topic 2'}
)
```

#### Create Custom Agent
```python
from rag_analysis.agents import create_domain_expert_agent
from rag_analysis.agents.tools import RAG_TOOLS
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_domain_expert_agent(
    llm,
    RAG_TOOLS,
    "Finance",
    "You are a Wall Street analyst with 25 years experience..."
)
```

#### Configure System
```python
from rag_analysis import Config

config = Config.from_env()

# Customize LLM
config.llm.model_name = "mixtral-8x7b-32768"
config.llm.temperature = 0.5

# Customize embeddings
config.embedding.model_name = "BAAI/bge-large-en-v1.5"

# Customize indexing
config.index.chunk_size = 1024
config.index.chunk_overlap = 100
config.index.similarity_top_k = 7

# Customize storage
from pathlib import Path
config.storage.reports_dir = Path("./my_reports")
```

#### Enable Verbose Mode
```python
from rag_analysis import Config

config = Config.from_env()
config.verbose = True

workflow = AnalysisWorkflow(config)
# Now shows detailed progress
```

#### Check Knowledge Base Stats
```python
from rag_analysis import KnowledgeBaseBuilder

kb = KnowledgeBaseBuilder()
kb.load_index()

stats = kb.get_stats()
print(f"Documents: {stats['total_documents']}")
print(f"Index built: {stats['index_built']}")
```

#### Monitor Query Performance
```python
import time

start = time.time()
result = query_engine.query("What is AI?")
elapsed = time.time() - start

print(f"Query time: {elapsed:.2f}s")
print(f"Result length: {len(str(result))} chars")
```

### Configuration Presets

#### Development Setup
```python
from rag_analysis import Config

config = Config.from_env()
config.embedding.model_name = "BAAI/bge-small-en-v1.5"  # Fast
config.index.chunk_size = 512
config.index.similarity_top_k = 5
config.llm.temperature = 0.7  # More creative
```

#### Production Setup
```python
from rag_analysis import Config

config = Config.from_env()
config.embedding.model_name = "BAAI/bge-large-en-v1.5"  # Accurate
config.index.chunk_size = 1024
config.index.similarity_top_k = 7
config.llm.temperature = 0.3  # More consistent
config.verbose = False
```

#### Fast Analysis Setup
```python
from rag_analysis import Config

config = Config.from_env()
config.index.chunk_size = 256  # Smaller chunks
config.index.similarity_top_k = 3  # Less context
config.llm.temperature = 0.1  # Quick decisions
```

#### Comprehensive Analysis Setup
```python
from rag_analysis import Config

config = Config.from_env()
config.index.chunk_size = 2048  # Large chunks
config.index.similarity_top_k = 10  # More context
config.llm.temperature = 0.5  # Balanced
config.verbose = True  # See all details
```

### Environment Variables

```bash
# Required
export GROQ_API_KEY='sk-...'

# Optional (with defaults)
export EMBEDDING_MODEL='BAAI/bge-small-en-v1.5'
export LLM_MODEL='llama-3.3-70b-versatile'
export LLM_TEMPERATURE='0.7'
export VERBOSE='true'

# Or in .env file
GROQ_API_KEY=sk-...
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
VERBOSE=true
```

---

## Glossary

### Core Concepts

#### RAG (Retrieval-Augmented Generation)
**Definition:** Combine document retrieval with LLM generation
- Search knowledge base first
- Generate response using retrieved context
- More accurate and grounded than pure generation
- **Example:** "Find facts about AI, then write summary using those facts"

#### Vector Index
**Definition:** Database of document embeddings for semantic search
- Stores numerical representations of text meaning
- Enables similarity-based search
- Much better than keyword matching
- **Example:** "AI" and "artificial intelligence" are nearby in vector space

#### Embedding
**Definition:** Vector representation of text meaning
- Converts text into numbers (300-768 dimensions)
- Captures semantic relationships
- Similar meaning = similar vectors
- **Example:** "king" - "man" + "woman" ≈ "queen"

#### Chunking
**Definition:** Breaking documents into smaller pieces
- Improves search precision
- Affects quality and performance
- Typical size: 256-2048 tokens
- **Trade-off:** Smaller = more precise, larger = more context

#### Query Engine
**Definition:** Interface to search vector index
- Takes natural language query
- Finds similar documents
- Returns most relevant passages
- **Process:** Embed query → Search vectors → Return top-k

#### Prompt
**Definition:** Instructions to LLM with context
- "You are an analyst. Analyze this: [context]. Question: [query]"
- Quality of prompt affects quality of response
- Context from retrieval goes into prompt

#### Temperature
**Definition:** Controls randomness of LLM responses
- 0.0 = Deterministic (same answer every time)
- 0.7 = Balanced (some variety)
- 1.0+ = Very creative (unpredictable)
- **Use:** 0.1-0.3 for consistent analysis, 0.7+ for brainstorming

### Agent Concepts

#### Agent
**Definition:** AI entity with role, goal, and tools
- Has personality and expertise
- Can search knowledge base
- Can collaborate with other agents
- **Example:** Research Analyst agent searches KB and writes findings

#### Tool
**Definition:** Function agents can call
- `search_knowledge_base()` searches KB
- `extract_statistics()` extracts numbers
- `compare_topics()` compares items
- Agents use tools to accomplish tasks

#### Crew
**Definition:** Group of agents working together
- Multiple agents with different roles
- Execute tasks in sequence
- Share context and collaborate
- **Example:** Researcher → Analyst → Writer

#### Task
**Definition:** Work unit assigned to agent
- Describes what to do
- Can depend on other tasks
- Specifies expected output
- **Example:** "Research AI trends" → "Analyze findings" → "Write report"

#### Role
**Definition:** Agent's professional identity
- Affects how agent approaches problems
- Influences decision-making
- Shapes interactions with other agents
- **Example:** Research Analyst vs Strategic Advisor

#### Goal
**Definition:** Agent's primary objective
- What the agent is trying to achieve
- Guides task execution
- Shapes tool usage
- **Example:** "Extract and synthesize information"

#### Backstory
**Definition:** Agent's background and experience
- Provides context and expertise
- Influences analysis quality
- Shapes professional approach
- **Example:** "15 years in healthcare research..."

### System Architecture

#### Part 1: Knowledge Base
**Definition:** Document parsing and indexing layer
- Loads documents from multiple sources
- Parses complex structures
- Creates vector index
- Persists to disk
- **Component:** `knowledge_base/` directory

#### Part 2: Agents
**Definition:** Multi-agent analysis layer
- Creates specialized agents
- Defines tasks
- Manages collaboration
- Provides tools for KB access
- **Component:** `agents/` directory

#### Part 3: Workflows
**Definition:** Orchestration and execution layer
- Connects KB to agents
- Manages execution flow
- Saves results
- Provides convenience functions
- **Component:** `workflows/` directory

### Technology Stack

#### LlamaIndex
**What:** Document indexing and retrieval framework
- Specializes in RAG
- Handles complex document parsing
- Provides query engines
- 100+ data source connectors
- **Use:** Build and query knowledge base

#### CrewAI
**What:** Multi-agent orchestration framework
- Handles agent collaboration
- Task management
- Sequential execution
- Built on LangChain
- **Use:** Create and run agent workflows

#### LangChain
**What:** LLM framework
- Provides chat/completion interfaces
- Powers both LlamaIndex and CrewAI
- Manages LLM interactions
- Handles prompt construction
- **Use:** Underlying technology for agents and KB

#### Groq
**What:** Fast LLM inference provider
- Specialized hardware (LPU)
- 100x faster than alternatives
- Free tier: 14,400 requests/month
- Cost-effective
- **Use:** Fast LLM responses

#### FAISS
**What:** Vector search library by Meta
- In-memory vector store
- Fast similarity search
- CPU-based (no GPU needed)
- Perfect for local development
- **Use:** Store and search embeddings

#### Sentence Transformers
**What:** Embedding models library
- Pre-trained embedding models
- Fast and lightweight
- Various model sizes
- Good for semantic search
- **Use:** Convert documents to embeddings

### Performance Metrics

#### Tokens Per Second (TPS)
**Definition:** How many tokens LLM generates per second
- OpenAI: ~20 TPS
- Claude: ~30 TPS
- Groq: ~100 TPS ⚡
- **Impact:** Affects response time

#### Query Latency
**Definition:** Time to find relevant documents
- Embedding query: ~50ms
- Vector search: ~10ms
- Total retrieval: ~100ms
- **Good range:** 100-500ms

#### Index Size
**Definition:** Disk space needed for vector index
- ~1-2 MB per 100 pages
- Depends on embedding model
- Compression possible
- **Example:** 1000 pages = 10-20 MB

#### Building Speed
**Definition:** Time to build index from documents
- ~1-2 documents/second (with embedding)
- Depends on document complexity
- Batch processing faster
- **Example:** 1000 pages = 10-30 minutes

### File Formats Supported

#### PDF
**Support:** ✅ Full support
- Text extraction
- Table parsing
- Image handling
- **Loader:** `PDFReader`

#### DOCX
**Support:** ✅ Full support
- Text extraction
- Formatting preserved
- Table support
- **Loader:** `DocxReader`

#### TXT
**Support:** ✅ Full support
- Plain text files
- Any encoding
- Fast loading
- **Loader:** File operations

#### Web Content
**Support:** ✅ Full support
- URL scraping
- HTML parsing
- Multiple pages
- **Loader:** `SimpleWebPageReader`

#### CSV/Excel
**Support:** ⚠️ Partial (custom loaders)
- Requires custom implementation
- Available as example
- CSV → markdown or text
- **Loader:** `CSVDocumentLoader` (example)

### Chunk Size Guidelines

| Size | Speed | Context | Best For |
|------|-------|---------|----------|
| 256 | ⚡⚡⚡ Fast | ⭐ Low | Precise queries |
| 512 | ⚡⚡ Good | ⭐⭐ Medium | Balanced (default) |
| 1024 | ⚡ Slower | ⭐⭐⭐ High | Complex docs |
| 2048 | 🐢 Slow | ⭐⭐⭐⭐ Very high | Large contexts |

### Model Recommendations

#### Embedding Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| all-MiniLM-L6-v2 | ⚡⚡⚡ | ⭐⭐ | Development |
| bge-small-en-v1.5 | ⚡⚡ | ⭐⭐⭐ | Balanced |
| bge-large-en-v1.5 | ⚡ | ⭐⭐⭐⭐ | Production |

#### LLM Models

| Model | Speed | Cost | Quality | Use |
|-------|-------|------|---------|-----|
| Llama 3.3 70B | ⚡⚡⚡ | $ | ⭐⭐⭐ | Default |
| Mixtral 8x7B | ⚡⚡ | $ | ⭐⭐⭐ | Balanced |
| Groq available | ⚡⚡⚡ | $$ | ⭐⭐⭐ | Production |

### Temperature Guidelines

| Value | Behavior | Best For |
|-------|----------|----------|
| 0.1-0.3 | Focused, consistent | Analysis, reports |
| 0.5-0.7 | Balanced | General use (default) |
| 0.8-1.0 | Creative, varied | Brainstorming |
| 1.0+ | Unpredictable | Creative writing |

---

## Troubleshooting Quick Reference

### Error: "GROQ_API_KEY not found"
```bash
# Solution
export GROQ_API_KEY='your_key'
# or
echo "GROQ_API_KEY=your_key" > .env
```

### Error: "No documents loaded"
```bash
# Check files exist
ls -la ./data/

# Check file types
file ./data/*

# Verify paths in code
python -c "from pathlib import Path; print(list(Path('./data').glob('*')))"
```

### Error: "Module not found"
```bash
# Reinstall
pip install -e .

# Verify
pip list | grep rag-analysis
```

### Slow performance
```python
# Reduce context
config.index.chunk_size = 256
config.index.similarity_top_k = 3

# Use faster model
config.embedding.model_name = "all-MiniLM-L6-v2"
```

### Out of memory
```python
# Process in batches
for batch in batches:
    kb.load_documents(pdf_paths=batch)
    kb.build_index()
```

### Low quality results
```python
# Get more context
config.index.similarity_top_k = 10

# Use better embedding
config.embedding.model_name = "BAAI/bge-large-en-v1.5"

# Lower temperature
config.llm.temperature = 0.3
```

---

## File Size Reference

| Item | Typical Size |
|------|--------------|
| Single PDF (100 pages) | 2-5 MB |
| Vector index (100 pages) | 1-2 MB |
| Generated report | 50-500 KB |
| Batch of 10 PDFs | 20-50 MB |
| Complete KB (1000 pages) | 10-20 MB |

---

## Time Reference

| Operation | Time |
|-----------|------|
| Embed single page | 100ms |
| Build index (1000 pages) | 10-30 min |
| Load index from disk | 1-2 sec |
| Single query | 100-500ms |
| LLM response (1000 tokens) | 10-30 sec |
| Full analysis (Basic) | 1-5 min |

---

## Cost Reference (Groq Free Tier)

| Metric | Value |
|--------|-------|
| Monthly requests | 14,400 |
| Per request cost | $0 |
| Daily limit (approx) | 480 |
| Analyses/month (if 3 requests each) | ~4,800 |

---

## Command Line Shortcuts

```bash
# Build KB
alias build-kb='python examples/build_knowledge_base.py'

# Run analysis
alias run-analysis='python examples/run_analysis.py'

# Full pipeline
alias full-pipeline='python examples/full_pipeline.py'

# Check logs
alias logs='sudo journalctl -u rag-analysis -f'

# Restart service
alias restart='sudo systemctl restart rag-analysis'

# Status
alias status='sudo systemctl status rag-analysis'
```

---

## Common Workflows

### Workflow 1: Quick Analysis
```bash
python examples/full_pipeline.py
# Handles everything automatically
```

### Workflow 2: Development
```bash
python examples/build_knowledge_base.py  # Build KB
python examples/run_analysis.py          # Run analyses
```

### Workflow 3: Production
```bash
sudo systemctl start rag-analysis        # Start service
# Service handles analyses automatically
```

### Workflow 4: Batch Processing
```python
# Process list of topics
topics = [...]
for topic in topics:
    analyze_with_existing_kb('basic', {'topic': topic})
```

---

**End of Quick Reference & Glossary**