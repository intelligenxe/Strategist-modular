"""
Document loaders for different file formats
Handles PDF, DOCX, TXT, and web content
"""

from typing import List
from pathlib import Path

from llama_index.core import Document
from llama_index.readers.file import PDFReader, DocxReader

from ..config import Config


class DocumentLoader:
    """Handles loading documents from various sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
    
    def load_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load PDF documents
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of Document objects
        """
        documents = []
        for path in pdf_paths:
            try:
                docs = self.pdf_reader.load_data(file=path)
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = path
                    doc.metadata["type"] = "pdf"
                documents.extend(docs)
            except Exception as e:
                if self.config.verbose:
                    print(f"✗ Error loading PDF {path}: {str(e)}")
        
        return documents
    
    def load_docx(self, docx_paths: List[str]) -> List[Document]:
        """
        Load DOCX documents
        
        Args:
            docx_paths: List of DOCX file paths
            
        Returns:
            List of Document objects
        """
        documents = []
        for path in docx_paths:
            try:
                docs = self.docx_reader.load_data(file=path)
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = path
                    doc.metadata["type"] = "docx"
                documents.extend(docs)
            except Exception as e:
                if self.config.verbose:
                    print(f"✗ Error loading DOCX {path}: {str(e)}")
        
        return documents
    
    def load_text_files(self, txt_paths: List[str]) -> List[Document]:
        """
        Load plain text files
        
        Args:
            txt_paths: List of text file paths
            
        Returns:
            List of Document objects
        """
        documents = []
        for path in txt_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    text=content,
                    metadata={
                        "source": path,
                        "type": "txt"
                    }
                )
                documents.append(doc)
            except Exception as e:
                if self.config.verbose:
                    print(f"✗ Error loading text file {path}: {str(e)}")
        
        return documents
    
    def load_web_content(self, urls: List[str]) -> List[Document]:
        """
        Load content from web URLs
        
        Args:
            urls: List of URLs
            
        Returns:
            List of Document objects
        """
        try:
            from llama_index.readers.web import SimpleWebPageReader
            
            loader = SimpleWebPageReader()
            documents = []
            
            for url in urls:
                try:
                    docs = loader.load_data([url])
                    # Add source metadata
                    for doc in docs:
                        doc.metadata["source"] = url
                        doc.metadata["type"] = "web"
                    documents.extend(docs)
                except Exception as e:
                    if self.config.verbose:
                        print(f"✗ Error loading URL {url}: {str(e)}")
            
            return documents
            
        except ImportError:
            if self.config.verbose:
                print("✗ Web loader not available. Install: pip install llama-index-readers-web")
            return []
    
    def load_from_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        allowed_extensions: tuple = ('.pdf', '.docx', '.txt')
    ) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory: Directory path
            recursive: If True, search subdirectories
            allowed_extensions: Tuple of allowed file extensions
            
        Returns:
            List of Document objects
        """
        documents = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            if self.config.verbose:
                print(f"✗ Directory not found: {directory}")
            return documents
        
        # Collect files
        if recursive:
            files = [f for f in dir_path.rglob('*') if f.suffix.lower() in allowed_extensions]
        else:
            files = [f for f in dir_path.glob('*') if f.suffix.lower() in allowed_extensions]
        
        # Group by extension
        pdf_files = [str(f) for f in files if f.suffix.lower() == '.pdf']
        docx_files = [str(f) for f in files if f.suffix.lower() == '.docx']
        txt_files = [str(f) for f in files if f.suffix.lower() == '.txt']
        
        # Load each type
        if pdf_files:
            documents.extend(self.load_pdfs(pdf_files))
        if docx_files:
            documents.extend(self.load_docx(docx_files))
        if txt_files:
            documents.extend(self.load_text_files(txt_files))
        
        return documents


class CustomDocumentLoader:
    """
    Base class for custom document loaders
    Extend this to support additional document formats
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def supports(self, file_path: str) -> bool:
        """
        Check if this loader supports the given file
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if supported, False otherwise
        """
        raise NotImplementedError("Subclasses must implement supports()")


# Example custom loader
class CSVDocumentLoader(CustomDocumentLoader):
    """Loader for CSV files"""
    
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.csv')
    
    def load(self, file_path: str) -> List[Document]:
        """Load CSV file as documents (one row per document or entire file)"""
        import csv
        
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Option 1: Entire CSV as one document
                content = f"CSV File: {file_path}\n\n"
                for row in rows:
                    content += ", ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
                
                doc = Document(
                    text=content,
                    metadata={
                        "source": file_path,
                        "type": "csv",
                        "rows": len(rows)
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            if self.config.verbose:
                print(f"✗ Error loading CSV {file_path}: {str(e)}")
        
        return documents
