"""Document loaders for various file formats."""

import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    Docx2txtLoader
)


class DocumentLoader:
    """Unified document loader for various file formats."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyMuPDFLoader,
        '.txt': TextLoader,
        '.epub': UnstructuredEPubLoader,
        '.docx': Docx2txtLoader,
    }
    
    @classmethod
    def load_document(cls, file_path: str, **kwargs) -> List[Document]:
        """Load a document from file path.
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional arguments for the loader
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(cls.SUPPORTED_EXTENSIONS.keys())}")
        
        # Get appropriate loader class
        loader_class = cls.SUPPORTED_EXTENSIONS[ext]
        
        try:
            # Create loader instance and load documents
            loader = loader_class(file_path, **kwargs)
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_extension': ext
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    @classmethod
    def load_directory(cls, directory_path: str, recursive: bool = True) -> List[Document]:
        """Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of all loaded Document objects
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        # Walk through directory
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                # Check if file format is supported
                if ext in cls.SUPPORTED_EXTENSIONS:
                    try:
                        documents = cls.load_document(file_path)
                        all_documents.extend(documents)
                        print(f"Loaded: {file_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            
            # If not recursive, only process the top-level directory
            if not recursive:
                break
        
        return all_documents
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return list(cls.SUPPORTED_EXTENSIONS.keys())


class BookProcessor:
    """Specialized processor for book documents."""
    
    def __init__(self, loader: DocumentLoader = None):
        """Initialize book processor.
        
        Args:
            loader: Document loader instance
        """
        self.loader = loader or DocumentLoader()
    
    def process_book(self, file_path: str, book_metadata: dict = None) -> List[Document]:
        """Process a book file and add book-specific metadata.
        
        Args:
            file_path: Path to the book file
            book_metadata: Additional metadata for the book (title, author, etc.)
            
        Returns:
            List of processed Document objects
        """
        # Load the document
        documents = self.loader.load_document(file_path)
        
        # Add book-specific metadata
        book_info = book_metadata or {}
        book_info.update({
            'content_type': 'book',
            'book_id': book_info.get('book_id', os.path.splitext(os.path.basename(file_path))[0])
        })
        
        # Update all document metadata
        for doc in documents:
            doc.metadata.update(book_info)
        
        return documents
    
    def extract_book_metadata(self, file_path: str) -> dict:
        """Extract metadata from book file (title, author, etc.).
        
        Args:
            file_path: Path to the book file
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'book_id': os.path.splitext(os.path.basename(file_path))[0]
        }
        
        # TODO: Add more sophisticated metadata extraction
        # For PDF files, could extract title, author from PDF metadata
        # For EPUB files, could extract from EPUB metadata
        
        return metadata
