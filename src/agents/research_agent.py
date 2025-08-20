"""Research Agent with FAISS vector database integration."""

from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional Google AI imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("⚠️  Google AI not available - using HuggingFace embeddings instead")

from src.config import Config
from src.vector_db.faiss_manager import FAISSManager
from src.document_processing.loaders import DocumentLoader, BookProcessor


class ResearchAgent:
    """AI Research Agent with vector database capabilities."""
    
    def __init__(self, config: Config = None):
        """Initialize the research agent.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.config.validate()
        
        # Initialize LLM with fallback models
        self.llm = self._initialize_llm()
        
        # Initialize components
        self.vector_db = FAISSManager(config=self.config)
        self.document_loader = DocumentLoader()
        self.book_processor = BookProcessor()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize memory and agent
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            memory=self.memory
        )

    def _initialize_llm(self):
        """Initialize LLM with Groq (primary) and Google AI (fallback)."""

        # Try Groq first (better rate limits)
        if self.config.GROQ_API_KEY:
            groq_models = [
                "llama3-8b-8192",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ]

            for model_name in groq_models:
                try:
                    print(f"🧪 Trying Groq model: {model_name}")
                    llm = ChatGroq(
                        model=model_name,
                        groq_api_key=self.config.GROQ_API_KEY,
                        temperature=0.1
                    )

                    # Test the model with a simple query
                    test_response = llm.invoke("Hello")
                    print(f"✅ Successfully initialized Groq model: {model_name}")
                    return llm

                except Exception as e:
                    print(f"❌ Groq model {model_name} failed: {str(e)}")
                    continue

        # Fallback to Google AI if Groq fails and Google AI is available
        if self.config.GOOGLE_API_KEY and GOOGLE_AI_AVAILABLE:
            google_models = [
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]

            for model_name in google_models:
                try:
                    print(f"🧪 Trying Google AI model: {model_name}")
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=self.config.GOOGLE_API_KEY,
                        temperature=0.1
                    )

                    # Test the model with a simple query
                    test_response = llm.invoke("Hello")
                    print(f"✅ Successfully initialized Google AI model: {model_name}")
                    return llm

                except Exception as e:
                    print(f"❌ Google AI model {model_name} failed: {str(e)}")
                    continue

        # If all models fail, raise an error
        raise Exception("❌ No working LLM models found. Please check your API keys and model availability.")
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the research agent.
        
        Returns:
            List of Tool objects
        """
        tools = [
            Tool(
                name="ArxivSearch",
                func=ArxivQueryRun().run,
                description="Search for academic papers on arXiv based on a topic or query"
            ),
            Tool(
                name="SearchBooks",
                func=self.search_books,
                description="Search through the local book collection using semantic similarity"
            ),
            Tool(
                name="AddBook",
                func=self.add_book,
                description="Add a new book to the vector database from a file path"
            ),
            Tool(
                name="GetBookInfo",
                func=self.get_book_info,
                description="Get information about books in the database"
            ),
            Tool(
                name="SummarizeBook",
                func=self.summarize_book,
                description="Generate a summary of a specific book or book section"
            ),
            Tool(
                name="CompareContent",
                func=self.compare_content,
                description="Compare content between different books or papers"
            )
        ]
        
        return tools
    
    def search_books(self, query: str, num_results: int = 5) -> str:
        """Search through books in the vector database.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted search results
        """
        try:
            results = self.vector_db.search_similar(query, k=num_results)
            
            if not results:
                return "No relevant content found in the book database."
            
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                book_id = doc.metadata.get('book_id', 'Unknown')
                file_name = doc.metadata.get('file_name', 'Unknown')
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                
                formatted_results.append(
                    f"Result {i} (Score: {score:.3f}):\n"
                    f"Book: {book_id} ({file_name})\n"
                    f"Content: {content_preview}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching books: {str(e)}"
    
    def add_book(self, file_path: str, book_metadata: str = None) -> str:
        """Add a book to the vector database.
        
        Args:
            file_path: Path to the book file
            book_metadata: Optional metadata as JSON string
            
        Returns:
            Success/failure message
        """
        try:
            # Parse metadata if provided
            metadata = {}
            if book_metadata:
                import json
                metadata = json.loads(book_metadata)
            
            # Process the book
            documents = self.book_processor.process_book(file_path, metadata)
            
            # Extract book ID
            book_id = metadata.get('book_id') or documents[0].metadata.get('book_id')
            
            # Add to vector database
            success = self.vector_db.add_documents(documents, book_id)
            
            if success:
                # Save the database
                self.vector_db.save_database()
                return f"Successfully added book '{book_id}' with {len(documents)} sections to the database."
            else:
                return f"Failed to add book to database."
                
        except Exception as e:
            return f"Error adding book: {str(e)}"
    
    def get_book_info(self, query: str = None) -> str:
        """Get information about books in the database.
        
        Args:
            query: Optional query to filter books
            
        Returns:
            Information about books
        """
        try:
            stats = self.vector_db.get_database_stats()
            
            info = [
                f"Database Statistics:",
                f"- Total books: {stats['total_books']}",
                f"- Database exists: {stats['database_exists']}",
                f"- Database path: {stats['database_path']}"
            ]
            
            if stats.get('total_vectors'):
                info.append(f"- Total text chunks: {stats['total_vectors']}")
            
            # Add metadata information
            if self.vector_db.metadata_store:
                info.append("\nBooks in database:")
                for book_id, metadata in self.vector_db.metadata_store.items():
                    info.append(f"- {book_id}: {metadata.get('num_chunks', 0)} chunks")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"Error getting book info: {str(e)}"
    
    def summarize_book(self, book_query: str) -> str:
        """Generate a summary of a book or book section.
        
        Args:
            book_query: Query to identify the book or section
            
        Returns:
            Summary of the content
        """
        try:
            # Search for relevant content
            results = self.vector_db.search_similar(book_query, k=10)
            
            if not results:
                return "No content found for the specified query."
            
            # Combine content from top results
            combined_content = "\n\n".join([doc.page_content for doc, _ in results[:5]])
            
            # Create summarization prompt
            prompt = PromptTemplate(
                input_variables=["content"],
                template="Please provide a comprehensive summary of the following content:\n\n{content}\n\nSummary:"
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            summary = chain.run({"content": combined_content})
            
            return summary
            
        except Exception as e:
            return f"Error summarizing content: {str(e)}"
    
    def compare_content(self, query1: str, query2: str) -> str:
        """Compare content between different sources.
        
        Args:
            query1: First content query
            query2: Second content query
            
        Returns:
            Comparison analysis
        """
        try:
            # Search for both contents
            results1 = self.vector_db.search_similar(query1, k=3)
            results2 = self.vector_db.search_similar(query2, k=3)
            
            if not results1 or not results2:
                return "Insufficient content found for comparison."
            
            content1 = "\n".join([doc.page_content for doc, _ in results1])
            content2 = "\n".join([doc.page_content for doc, _ in results2])
            
            # Create comparison prompt
            prompt = PromptTemplate(
                input_variables=["content1", "content2"],
                template="Compare and analyze the following two pieces of content. Highlight similarities, differences, and key insights:\n\nContent 1:\n{content1}\n\nContent 2:\n{content2}\n\nComparison:"
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            comparison = chain.run({"content1": content1, "content2": content2})
            
            return comparison
            
        except Exception as e:
            return f"Error comparing content: {str(e)}"
    
    def run(self, query: str) -> str:
        """Run a query through the research agent.
        
        Args:
            query: User query
            
        Returns:
            Agent response
        """
        try:
            response = self.agent.run(query)
            return response
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the research agent.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'vector_db_stats': self.vector_db.get_database_stats(),
            'supported_formats': self.document_loader.get_supported_formats(),
            'tools_available': [tool.name for tool in self.tools]
        }
