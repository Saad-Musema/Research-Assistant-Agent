"""Configuration settings for the Research Assistant."""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the Research Assistant."""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Settings
    DEFAULT_LLM_MODEL = "llama3-8b-8192"  # Groq model
    GROQ_MODEL = "llama3-8b-8192"  # Fast Groq model
    EMBEDDING_MODEL = "models/embedding-001"  # Still use Google for embeddings
    
    # Vector Database Settings
    VECTOR_DB_PATH = "data/vector_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # File Paths
    BOOKS_DIR = "data/books"
    PAPERS_DIR = "data/papers"
    CACHE_DIR = "data/cache"
    
    # Search Settings
    MAX_SEARCH_RESULTS = 10
    SIMILARITY_THRESHOLD = 0.7
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        if not cls.GOOGLE_API_KEY:
            print("GOOGLE_API_KEY not found - embeddings may not work")
        
        # Create necessary directories
        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(cls.BOOKS_DIR, exist_ok=True)
        os.makedirs(cls.PAPERS_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
