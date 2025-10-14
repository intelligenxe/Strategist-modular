"""
Example: Building a Knowledge Base
This script shows how to build and persist a knowledge base from documents
"""

import os
from pathlib import Path
from rag_analysis import KnowledgeBaseBuilder, Config

def main():
    """Build knowledge base from documents"""
    
    # Step 1: Configure
    config = Config(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        verbose=True
    )
    
    # Optional: customize index settings
    config.index.chunk_size = 512
    config.index.chunk_overlap = 50
    config.index.similarity_top_k = 5
    
    print("Building Knowledge Base")
    print("=" * 70)
    
    # Step 2: Initialize builder
    kb_builder = KnowledgeBaseBuilder(config)
    
    # Step 3: Load documents from various sources
    
    # Option A: Load specific files
    kb_builder.load_documents(
        pdf_paths=[
            "./data/report1.pdf",
            "./data/analysis.pdf"
        ],
        docx_paths=[
            "./data/document1.docx"
        ],
        txt_paths=[
            "./data/notes.txt",
            "./data/summary.txt"
        ],
        urls=[
            "https://example.com/article1",
            "https://example.com/article2"
        ]
    )
    
    # Option B: Load entire directory
    # kb_builder.load_from_directory("./data", recursive=True)
    
    # Step 4: Build and persist index
    kb_builder.build_index(persist=True)
    
    # Step 5: Test the knowledge base
    print("\n" + "=" * 70)
    print("Testing Knowledge Base")
    print("=" * 70)
    
    query_engine = kb_builder.get_query_engine()
    
    # Test query
    test_query = "What are the main topics covered in the documents?"
    print(f"\nQuery: {test_query}")
    response = query_engine.query(test_query)
    print(f"Response: {response}")
    
    # Get stats
    print("\n" + "=" * 70)
    stats = kb_builder.get_stats()
    print("Knowledge Base Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Index built: {stats['index_built']}")
    
    print("\n✓ Knowledge base built successfully!")
    print(f"  Saved to: {config.storage.index_dir}")


if __name__ == "__main__":
    main()
