"""
Configuration management for RAG Analysis system
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    

@dataclass
class LLMConfig:
    """Configuration for LLM"""
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 2048
    

@dataclass
class IndexConfig:
    """Configuration for vector index"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5
    response_mode: str = "compact"


@dataclass
class StorageConfig:
    """Configuration for storage paths"""
    base_dir: Path = Path("./storage")
    index_dir: Path = Path("./storage/index")
    documents_dir: Path = Path("./data")
    reports_dir: Path = Path("./reports")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.base_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)


@dataclass
class Config:
    """Main configuration class"""
    groq_api_key: Optional[str] = None
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    index: IndexConfig = IndexConfig()
    storage: StorageConfig = StorageConfig()
    verbose: bool = True
    
    def __post_init__(self):
        """Load API key from environment if not provided"""
        if self.groq_api_key is None:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if self.groq_api_key is None:
                raise ValueError(
                    "GROQ_API_KEY not found. Set it in environment or pass to Config"
                )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            embedding=EmbeddingConfig(
                model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
            ),
            llm=LLMConfig(
                model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
            ),
            verbose=os.getenv("VERBOSE", "true").lower() == "true"
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "embedding": {
                "model_name": self.embedding.model_name,
                "device": self.embedding.device
            },
            "llm": {
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens
            },
            "index": {
                "chunk_size": self.index.chunk_size,
                "chunk_overlap": self.index.chunk_overlap,
                "similarity_top_k": self.index.similarity_top_k
            },
            "verbose": self.verbose
        }


# Global config instance (can be imported and used across modules)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config):
    """Set global config instance"""
    global _config
    _config = config


def reset_config():
    """Reset global config to None"""
    global _config
    _config = None
