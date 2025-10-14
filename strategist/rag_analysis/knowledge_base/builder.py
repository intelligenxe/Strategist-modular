"""
Part 1: Knowledge Base Builder
Handles document parsing and vector index creation using LlamaIndex
"""

from typing import List, Optional
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.node_parser import SimpleNodeParser

from ..config import Config, get_config
from .loaders import DocumentLoader


class KnowledgeBaseBuilder:
    """
    Manages the complete lifecycle of the knowledge base:
    - Load documents from multiple sources
    - Parse and chunk documents
    - Build vector index
    - Persist and load index
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the knowledge base builder
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None
        self.loader = DocumentLoader(self.config)
        
        # Configure LlamaIndex settings
        self._configure_settings()
    
    def _configure_settings(self):
        """Configure global LlamaIndex settings"""
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device
        )
        Settings.llm = LlamaGroq(
            model=self.config.llm.model_name,
            api_key=self.config.groq_api_key,
            temperature=self.config.llm.temperature
        )
    
    def load_documents(
        self,
        pdf_paths: Optional[List[str]] = None,
        docx_paths: Optional[List[str]] = None,
        txt_paths: Optional[List[str]] = None,
        urls: Optional[List[str]] = None
    ):
        """
        Load documents from various sources
        
        Args:
            pdf_paths: List of PDF file paths
            docx_paths: List of DOCX file paths
            txt_paths: List of text file paths
            urls: List of URLs to scrape
        """
        if pdf_paths:
            docs = self.loader.load_pdfs(pdf_paths)
            self.documents.extend(docs)
            if self.config.verbose:
                print(f"✓ Loaded {len(docs)} documents from {len(pdf_paths)} PDFs")
        
        if docx_paths:
            docs = self.loader.load_docx(docx_paths)
            self.documents.extend(docs)
            if self.config.verbose:
                print(f"✓ Loaded {len(docs)} documents from {len(docx_paths)} DOCX files")
        
        if txt_paths:
            docs = self.loader.load_text_files(txt_paths)
            self.documents.extend(docs)
            if self.config.verbose:
                print(f"✓ Loaded {len(docs)} documents from {len(txt_paths)} text files")
        
        if urls:
            docs = self.loader.load_web_content(urls)
            self.documents.extend(docs)
            if self.config.verbose:
                print(f"✓ Loaded {len(docs)} documents from {len(urls)} URLs")
        
        if self.config.verbose:
            print(f"\n📚 Total documents loaded: {len(self.documents)}")
    
    def load_from_directory(self, directory: str, recursive: bool = True):
        """
        Load all supported documents from a directory
        
        Args:
            directory: Directory path
            recursive: If True, search subdirectories
        """
        docs = self.loader.load_from_directory(directory, recursive)
        self.documents.extend(docs)
        if self.config.verbose:
            print(f"✓ Loaded {len(docs)} documents from directory: {directory}")
    
    def add_documents(self, documents: List[Document]):
        """
        Add pre-loaded documents to the knowledge base
        
        Args:
            documents: List of LlamaIndex Document objects
        """
        self.documents.extend(documents)
        if self.config.verbose:
            print(f"✓ Added {len(documents)} documents")
    
    def build_index(self, persist: bool = True) -> VectorStoreIndex:
        """
        Build vector index from loaded documents
        
        Args:
            persist: If True, save index to disk
            
        Returns:
            VectorStoreIndex object
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        if self.config.verbose:
            print(f"\n🔨 Building index from {len(self.documents)} documents...")
        
        # Parse documents into nodes
        parser = SimpleNodeParser.from_defaults(
            chunk_size=self.config.index.chunk_size,
            chunk_overlap=self.config.index.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(self.documents)
        
        if self.config.verbose:
            print(f"✓ Created {len(nodes)} chunks")
        
        # Create index
        self.index = VectorStoreIndex(nodes)
        
        if self.config.verbose:
            print("✓ Vector index created")
        
        # Persist to disk
        if persist:
            self.save_index()
        
        return self.index
    
    def save_index(self, persist_dir: Optional[str] = None):
        """
        Save index to disk
        
        Args:
            persist_dir: Directory to save index. If None, uses config.storage.index_dir
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        persist_dir = persist_dir or str(self.config.storage.index_dir)
        self.index.storage_context.persist(persist_dir=persist_dir)
        
        if self.config.verbose:
            print(f"💾 Index saved to: {persist_dir}")
    
    def load_index(self, persist_dir: Optional[str] = None) -> VectorStoreIndex:
        """
        Load existing index from disk
        
        Args:
            persist_dir: Directory to load index from. If None, uses config.storage.index_dir
            
        Returns:
            VectorStoreIndex object
        """
        persist_dir = persist_dir or str(self.config.storage.index_dir)
        
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage_context)
        
        if self.config.verbose:
            print(f"📂 Index loaded from: {persist_dir}")
        
        return self.index
    
    def get_query_engine(self, similarity_top_k: Optional[int] = None):
        """
        Get query engine for RAG
        
        Args:
            similarity_top_k: Number of similar documents to retrieve.
                            If None, uses config.index.similarity_top_k
                            
        Returns:
            Query engine object
        """
        if self.index is None:
            raise ValueError(
                "Index not available. Call build_index() or load_index() first."
            )
        
        similarity_top_k = similarity_top_k or self.config.index.similarity_top_k
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=self.config.index.response_mode
        )
    
    def get_stats(self) -> dict:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "index_built": self.index is not None,
            "config": self.config.to_dict()
        }


def create_knowledge_base(
    document_paths: dict,
    config: Optional[Config] = None,
    persist: bool = True
) -> KnowledgeBaseBuilder:
    """
    Convenience function to create a knowledge base in one call
    
    Args:
        document_paths: Dictionary with keys 'pdfs', 'docx', 'txt', 'urls'
        config: Configuration object
        persist: If True, save index to disk
        
    Returns:
        KnowledgeBaseBuilder with built index
        
    Example:
        kb = create_knowledge_base({
            'pdfs': ['doc1.pdf', 'doc2.pdf'],
            'txt': ['notes.txt'],
            'urls': ['https://example.com']
        })
    """
    builder = KnowledgeBaseBuilder(config)
    builder.load_documents(
        pdf_paths=document_paths.get('pdfs'),
        docx_paths=document_paths.get('docx'),
        txt_paths=document_paths.get('txt'),
        urls=document_paths.get('urls')
    )
    builder.build_index(persist=persist)
    return builder
