#!/usr/bin/env python3
"""
Example usage of the Research Assistant
"""

import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.research_agent import ResearchAgent
from src.config import Config


def main():
    """Example usage of the Research Assistant."""
    
    print("Research Assistant Example")
    print("=" * 40)
    
    try:
        # Initialize the research agent
        print("Initializing Research Agent...")
        agent = ResearchAgent()
        
        # Show current stats
        stats = agent.get_stats()
        print(f"Current database stats:")
        print(f"   - Books: {stats['vector_db_stats']['total_books']}")
        print(f"   - Database exists: {stats['vector_db_stats']['database_exists']}")
        print(f"   - Supported formats: {', '.join(stats['supported_formats'])}")
        
        # Example 1: Add a book (if you have one)
        print("\nExample 1: Adding a book")
        print("To add a book, use: agent.add_book('/path/to/your/book.pdf')")
        
        # Example 2: Search books
        print("\nExample 2: Searching books")
        if stats['vector_db_stats']['total_books'] > 0:
            results = agent.search_books("machine learning", num_results=3)
            print("Search results:")
            print(results)
        else:
            print("No books in database yet. Add some books first!")
        
        # Example 3: Ask a research question
        print("\n Example 3: Research question")
        if stats['vector_db_stats']['total_books'] > 0:
            response = agent.run("What are the main topics covered in my book collection?")
            print("Agent response:")
            print(response)
        else:
            print("Add books to ask questions about your collection!")
        
        # Example 4: ArXiv search
        print("\n Example 4: ArXiv search")
        response = agent.run("Search for recent papers on large language models")
        print("ArXiv search results:")
        print(response)
        
    except Exception as e:
        print(f" Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with GOOGLE_API_KEY")
        print("2. Installed all requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
