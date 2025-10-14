"""
Setup script for rag_analysis package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="rag_analysis",
    version="0.1.0",
    description="Modular RAG system with CrewAI agents for document analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/rag_analysis",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core RAG (LlamaIndex)
        "llama-index>=0.9.0",
        "llama-index-llms-groq>=0.1.0",
        "llama-index-embeddings-huggingface>=0.1.0",
        
        # Document parsers
        "llama-index-readers-file>=0.1.0",
        "pypdf>=3.0.0",
        "python-docx>=0.8.11",
        "beautifulsoup4>=4.11.0",
        
        # CrewAI and LangChain
        "crewai>=0.1.0",
        "crewai-tools>=0.1.0",
        "langchain>=0.1.0",
        "langchain-groq>=0.0.1",
        "langchain-community>=0.0.1",
        
        # Vector store and embeddings
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        
        # Utilities
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gradio": [
            "gradio>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="rag llm crewai agents llamaindex analysis nlp",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/rag_analysis/issues",
        "Source": "https://github.com/yourusername/rag_analysis",
    },
)
