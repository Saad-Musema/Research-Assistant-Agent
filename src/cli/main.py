"""Command Line Interface for the Research Assistant."""

import argparse
import sys
import os
from typing import Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.research_agent import ResearchAgent
from src.config import Config


class ResearchAssistantCLI:
    """Command Line Interface for the Research Assistant."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the research agent."""
        try:
            self.agent = ResearchAgent()
            print("✅ Research Assistant initialized successfully!")
            print(f"📚 Database path: {self.agent.config.VECTOR_DB_PATH}")
            
            # Show current stats
            stats = self.agent.get_stats()
            print(f"📊 Books in database: {stats['vector_db_stats']['total_books']}")
            
        except Exception as e:
            print(f"❌ Error initializing Research Assistant: {e}")
            sys.exit(1)
    
    def interactive_mode(self):
        """Run the assistant in interactive mode."""
        print("\n🔬 Research Assistant - Interactive Mode")
        print("=" * 50)
        print("Available commands:")
        print("  - Ask any research question")
        print("  - 'add book <file_path>' to add a book")
        print("  - 'search <query>' to search books")
        print("  - 'info' to see database info")
        print("  - 'help' for this message")
        print("  - 'exit' or 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n🤖 Ask your research question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.lower() == 'info':
                    self._show_info()
                    continue
                
                if query.lower().startswith('add book '):
                    file_path = query[9:].strip()
                    self._add_book_command(file_path)
                    continue
                
                if query.lower().startswith('search '):
                    search_query = query[7:].strip()
                    self._search_command(search_query)
                    continue
                
                # Regular research query
                print("\n🔍 Processing your query...")
                response = self.agent.run(query)
                print(f"\n📝 Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
🔬 Research Assistant Help
========================

Commands:
  - Ask any research question in natural language
  - add book <file_path>     : Add a book to the database
  - search <query>          : Search through your book collection
  - info                    : Show database statistics
  - help                    : Show this help message
  - exit/quit              : Exit the program

Supported file formats:
  - PDF (.pdf)
  - Text (.txt)
  - EPUB (.epub)
  - Word (.docx)

Examples:
  - "What are the main themes in machine learning?"
  - "add book /path/to/book.pdf"
  - "search artificial intelligence"
  - "Compare deep learning approaches"
        """
        print(help_text)
    
    def _show_info(self):
        """Show database information."""
        try:
            stats = self.agent.get_stats()
            print("\n📊 Database Information:")
            print("=" * 30)
            print(f"Books in database: {stats['vector_db_stats']['total_books']}")
            print(f"Database exists: {stats['vector_db_stats']['database_exists']}")
            print(f"Database path: {stats['vector_db_stats']['database_path']}")
            
            if stats['vector_db_stats'].get('total_vectors'):
                print(f"Total text chunks: {stats['vector_db_stats']['total_vectors']}")
            
            print(f"Supported formats: {', '.join(stats['supported_formats'])}")
            print(f"Available tools: {', '.join(stats['tools_available'])}")
            
        except Exception as e:
            print(f"❌ Error getting info: {e}")
    
    def _add_book_command(self, file_path: str):
        """Handle add book command."""
        if not file_path:
            print("❌ Please provide a file path")
            return
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"📚 Adding book: {file_path}")
        result = self.agent.add_book(file_path)
        print(f"📝 Result: {result}")
    
    def _search_command(self, query: str):
        """Handle search command."""
        if not query:
            print("❌ Please provide a search query")
            return
        
        print(f"🔍 Searching for: {query}")
        result = self.agent.search_books(query)
        print(f"📝 Results:\n{result}")
    
    def add_book(self, file_path: str, metadata: Optional[str] = None):
        """Add a book via CLI command."""
        try:
            result = self.agent.add_book(file_path, metadata)
            print(result)
        except Exception as e:
            print(f"❌ Error adding book: {e}")
    
    def search_books(self, query: str, num_results: int = 5):
        """Search books via CLI command."""
        try:
            result = self.agent.search_books(query, num_results)
            print(result)
        except Exception as e:
            print(f"❌ Error searching: {e}")
    
    def query(self, question: str):
        """Process a single query via CLI."""
        try:
            response = self.agent.run(question)
            print(response)
        except Exception as e:
            print(f"❌ Error processing query: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Research Assistant with FAISS Vector Database")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive mode (default)
    parser.add_argument('--interactive', '-i', action='store_true', default=True,
                       help='Run in interactive mode (default)')
    
    # Add book command
    add_parser = subparsers.add_parser('add', help='Add a book to the database')
    add_parser.add_argument('file_path', help='Path to the book file')
    add_parser.add_argument('--metadata', help='Book metadata as JSON string')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search through books')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--num-results', type=int, default=5, help='Number of results')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Ask a research question')
    query_parser.add_argument('question', help='Research question')
    
    # Info command
    subparsers.add_parser('info', help='Show database information')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = ResearchAssistantCLI()
    
    # Handle commands
    if args.command == 'add':
        cli.add_book(args.file_path, args.metadata)
    elif args.command == 'search':
        cli.search_books(args.query, args.num_results)
    elif args.command == 'query':
        cli.query(args.question)
    elif args.command == 'info':
        cli._show_info()
    else:
        # Default to interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
