"""FAISS Vector Database Manager for storing and searching books/documents."""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.config import Config

# Optional Google AI embeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False


class FAISSManager:
    """Manages FAISS vector database for document storage and retrieval."""
    
    def __init__(self, db_path: Optional[str] = None, config: Optional[Config] = None):
        """Initialize FAISS manager.

        Args:
            db_path: Path to store the FAISS database
            config: Configuration object
        """
        self.config = config or Config()
        self.db_path = db_path or self.config.VECTOR_DB_PATH
        self.embeddings = self._initialize_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        self.vector_store = None
        self.metadata_store = {}

        # Load existing database if it exists
        self.load_database()

    def _initialize_embeddings(self):
        """Initialize embeddings with fallback options."""
        # Try Google AI embeddings first if available
        if GOOGLE_AI_AVAILABLE and self.config.GOOGLE_API_KEY:
            try:
                print("Trying Google AI embeddings...")
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.config.EMBEDDING_MODEL,
                    google_api_key=self.config.GOOGLE_API_KEY
                )
                # Test the embeddings
                embeddings.embed_query("test")
                print("Using Google AI embeddings")
                return embeddings
            except Exception as e:
                print(f"Google AI embeddings failed: {e}")

        # Fallback to HuggingFace embeddings
        print("Using HuggingFace embeddings (sentence-transformers)...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("Using HuggingFace embeddings")
            return embeddings
        except Exception as e:
            print(f"HuggingFace embeddings failed: {e}")
            raise Exception("No working embeddings found. Please install sentence-transformers: pip install sentence-transformers")
    
    def load_database(self) -> bool:
        """Load existing FAISS database from disk.
        
        Returns:
            True if database was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(f"{self.db_path}/index.faiss"):
                self.vector_store = FAISS.load_local(
                    self.db_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load metadata
                metadata_path = f"{self.db_path}/metadata.pkl"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        self.metadata_store = pickle.load(f)
                
                print(f"Loaded existing FAISS database from {self.db_path}")
                return True
        except Exception as e:
            print(f"Error loading database: {e}")
        
        return False
    
    def save_database(self) -> bool:
        """Save FAISS database to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if self.vector_store:
                os.makedirs(self.db_path, exist_ok=True)
                self.vector_store.save_local(self.db_path)
                
                # Save metadata
                metadata_path = f"{self.db_path}/metadata.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.metadata_store, f)
                
                print(f"Saved FAISS database to {self.db_path}")
                return True
        except Exception as e:
            print(f"Error saving database: {e}")
        
        return False
    
    def add_documents(self, documents: List[Document], book_id: Optional[str] = None) -> bool:
        """Add documents to the vector database.
        
        Args:
            documents: List of Document objects to add
            book_id: Optional identifier for the book/source
            
        Returns:
            True if documents were added successfully
        """
        try:
            # Split documents into chunks
            doc_chunks = self.text_splitter.split_documents(documents)
            
            # Add book_id to metadata if provided
            if book_id:
                for chunk in doc_chunks:
                    chunk.metadata['book_id'] = book_id
                    chunk.metadata['source_type'] = 'book'
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(doc_chunks, self.embeddings)
            else:
                self.vector_store.add_documents(doc_chunks)
            
            # Update metadata store
            if book_id:
                self.metadata_store[book_id] = {
                    'num_chunks': len(doc_chunks),
                    'source_documents': len(documents)
                }
            
            print(f"Added {len(doc_chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents in the vector database.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.vector_store is None:
            print("No vector database loaded")
            return []
        
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=k,
                filter=filter_dict
            )
            return results
            
        except Exception as e:
            print(f"Error searching database: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            'total_books': len(self.metadata_store),
            'database_exists': self.vector_store is not None,
            'database_path': self.db_path
        }
        
        if self.vector_store:
            stats['total_vectors'] = self.vector_store.index.ntotal
        
        return stats
    
    def delete_book(self, book_id: str) -> bool:
        """Delete all chunks related to a specific book.
        
        Args:
            book_id: ID of the book to delete
            
        Returns:
            True if deletion was successful
        """
        # Note: FAISS doesn't support direct deletion by metadata
        # This would require rebuilding the index without the specified book
        # For now, we'll just remove from metadata store
        if book_id in self.metadata_store:
            del self.metadata_store[book_id]
            print(f"Removed book {book_id} from metadata (index rebuild required for full removal)")
            return True
        
        return False
