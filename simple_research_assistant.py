#!/usr/bin/env python3
"""
Simple Research Assistant with FAISS Vector Database
Avoids complex agent framework to prevent version conflicts.
"""

import os
import sys
from dotenv import load_dotenv
from typing import List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

def test_imports():
    """Test if required packages are available."""
    try:
        from langchain_groq import ChatGroq
        print("Groq available")
        return True
    except ImportError as e:
        print(f"Groq not available: {e}")
        return False

def simple_groq_test():
    """Simple test of Groq functionality."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("GROQ_API_KEY not found in .env file")
        return False
    
    try:
        from langchain_groq import ChatGroq
        
        # Test different models
        models = ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        
        for model in models:
            try:
                print(f"Testing {model}...")
                llm = ChatGroq(
                    model=model,
                    groq_api_key=groq_api_key,
                    temperature=0.1
                )
                
                response = llm.invoke("Hello! Respond with just 'OK'")
                print(f"{model}: {response.content}")
                return llm  # Return the working model
                
            except Exception as e:
                print(f"{model}: {e}")
                continue
        
        print("No working Groq models found")
        return False
        
    except Exception as e:
        print(f"Groq test failed: {e}")
        return False

def simple_embeddings_test():
    """Test embeddings functionality."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("Testing HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Test embedding
        result = embeddings.embed_query("test query")
        print(f"Embeddings working: {len(result)} dimensions")
        return embeddings
        
    except Exception as e:
        print(f"Embeddings test failed: {e}")
        return False

def simple_faiss_test():
    """Test FAISS functionality."""
    try:
        import faiss
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document
        
        print("Testing FAISS...")
        
        # Get embeddings
        embeddings = simple_embeddings_test()
        if not embeddings:
            return False
        
        # Create test documents
        docs = [
            Document(page_content="This is about machine learning", metadata={"source": "test1"}),
            Document(page_content="This is about artificial intelligence", metadata={"source": "test2"}),
            Document(page_content="This is about deep learning", metadata={"source": "test3"})
        ]
        
        # Create FAISS index
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Test search
        results = vector_store.similarity_search("AI and ML", k=2)
        print(f"FAISS working: Found {len(results)} results")
        
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content[:50]}...")
        
        return vector_store
        
    except Exception as e:
        print(f"FAISS test failed: {e}")
        return False

class SimpleResearchAssistant:
    """Simple research assistant without complex agent framework."""
    
    def __init__(self):
        """Initialize the simple research assistant."""
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        
        print("Initializing Simple Research Assistant...")
        self._setup()
    
    def _setup(self):
        """Set up the assistant components."""
        # Test and setup LLM
        self.llm = simple_groq_test()
        if not self.llm:
            raise Exception("Failed to initialize LLM")
        
        # Test and setup embeddings
        self.embeddings = simple_embeddings_test()
        if not self.embeddings:
            raise Exception("Failed to initialize embeddings")
        
        print("Simple Research Assistant ready!")
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """Add documents to the vector store."""
        try:
            from langchain.schema import Document
            from langchain_community.vectorstores import FAISS
            
            # Create documents
            docs = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                docs.append(Document(page_content=text, metadata=metadata))
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vector_store.add_documents(docs)
            
            print(f"Added {len(docs)} documents to vector store")
            return True
            
        except Exception as e:
            print(f" Error adding documents: {e}")
            return False
    
    def search(self, query: str, k: int = 3):
        """Search the vector store."""
        if not self.vector_store:
            return "No documents in vector store yet. Add some documents first!"
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                return "No relevant documents found."
            
            response = f"Found {len(results)} relevant documents:\n\n"
            for i, doc in enumerate(results, 1):
                response += f"{i}. {doc.page_content[:200]}...\n"
                if doc.metadata:
                    response += f"   Source: {doc.metadata}\n"
                response += "\n"
            
            return response
            
        except Exception as e:
            return f"Search error: {e}"
    
    def ask(self, question: str):
        """Ask a question to the LLM."""
        try:
            response = self.llm.invoke(question)
            return response.content
        except Exception as e:
            return f"Error asking question: {e}"
    
    def ask_with_context(self, question: str, k: int = 5):
        """Ask a question with context from vector store - provides intelligent analysis."""
        if not self.vector_store:
            return self.ask(question)

        try:
            # Get relevant context
            results = self.vector_store.similarity_search(question, k=k)

            if results:
                context = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(results)])

                prompt = f"""You are an intelligent research assistant. Based on the provided context from books/documents, please provide a comprehensive and insightful answer to the question.

IMPORTANT INSTRUCTIONS:
1. Synthesize information from multiple sources when possible
2. Provide analysis, not just summaries
3. Draw connections between different concepts
4. Offer insights and implications
5. If you find contradictions between sources, mention them
6. Cite which sources support your points (e.g., "According to Source 1...")
7. If the context doesn't fully answer the question, say what's missing

CONTEXT FROM YOUR BOOKS:
{context}

QUESTION: {question}

COMPREHENSIVE ANSWER:"""
            else:
                prompt = f"""You are a research assistant. The user asked: "{question}"

I couldn't find relevant information in the uploaded books/documents. Please provide a general answer based on your knowledge, but clearly indicate that this is not from the user's specific collection."""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Error asking question with context: {e}"

    def summarize_topic(self, topic: str, k: int = 8):
        """Provide a comprehensive summary of a topic from your books."""
        if not self.vector_store:
            return "No documents available for summarization."

        try:
            results = self.vector_store.similarity_search(topic, k=k)

            if not results:
                return f"No information found about '{topic}' in your books."

            context = "\n\n".join([f"Excerpt {i+1}: {doc.page_content}" for i, doc in enumerate(results)])

            prompt = f"""You are a research assistant. Please provide a comprehensive summary of the topic "{topic}" based on the following excerpts from the user's book collection.

INSTRUCTIONS:
1. Create a well-structured summary with clear sections
2. Identify key themes and concepts
3. Note different perspectives if they exist
4. Highlight important insights or conclusions
5. Organize information logically
6. If there are gaps in coverage, mention what aspects might need additional research

EXCERPTS FROM YOUR BOOKS:
{context}

COMPREHENSIVE SUMMARY OF "{topic}":"""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Error summarizing topic: {e}"

    def compare_concepts(self, concept1: str, concept2: str, k: int = 6):
        """Compare two concepts based on information in your books."""
        if not self.vector_store:
            return "No documents available for comparison."

        try:
            # Search for both concepts
            results1 = self.vector_store.similarity_search(concept1, k=k//2)
            results2 = self.vector_store.similarity_search(concept2, k=k//2)

            if not results1 and not results2:
                return f"No information found about '{concept1}' or '{concept2}' in your books."

            context1 = "\n".join([doc.page_content for doc in results1]) if results1 else "No specific information found."
            context2 = "\n".join([doc.page_content for doc in results2]) if results2 else "No specific information found."

            prompt = f"""You are a research assistant. Please provide a detailed comparison between "{concept1}" and "{concept2}" based on information from the user's book collection.

INSTRUCTIONS:
1. Compare and contrast the two concepts
2. Identify similarities and differences
3. Analyze their relationships
4. Discuss their relative importance or applications
5. Note any advantages/disadvantages of each
6. Provide insights about when to use one vs the other

INFORMATION ABOUT {concept1.upper()}:
{context1}

INFORMATION ABOUT {concept2.upper()}:
{context2}

DETAILED COMPARISON:"""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Error comparing concepts: {e}"

    def find_connections(self, query: str, k: int = 10):
        """Find connections and patterns across your book collection."""
        if not self.vector_store:
            return "No documents available for analysis."

        try:
            results = self.vector_store.similarity_search(query, k=k)

            if not results:
                return f"No information found related to '{query}' in your books."

            context = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(results)])

            prompt = f"""You are a research assistant specializing in finding patterns and connections. Analyze the following excerpts from the user's book collection related to "{query}".

INSTRUCTIONS:
1. Identify recurring themes and patterns
2. Find connections between different sources
3. Highlight contradictions or debates
4. Note evolution of ideas or concepts
5. Identify gaps or areas needing more research
6. Suggest implications or applications
7. Look for interdisciplinary connections

EXCERPTS TO ANALYZE:
{context}

ANALYSIS OF CONNECTIONS AND PATTERNS:"""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Error finding connections: {e}"

def interactive_mode():
    """Run the assistant in interactive mode."""
    try:
        assistant = SimpleResearchAssistant()
        
        print("\nEnhanced Research Assistant - Interactive Mode")
        print("=" * 60)
        print("RESEARCH COMMANDS:")
        print("  - ask <question>           : Ask an intelligent question")
        print("  - summarize <topic>        : Get comprehensive topic summary")
        print("  - compare <concept1> vs <concept2> : Compare two concepts")
        print("  - connections <topic>      : Find patterns and connections")
        print("  - search <query>           : Basic search (shows excerpts)")
        print("")
        print("DOCUMENT COMMANDS:")
        print("  - add <text>               : Add a document")
        print("  - load <file_path>         : Load a book/document")
        print("")
        print("OTHER:")
        print("  - help                     : Show this help")
        print("  - exit                     : Exit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nEnter command: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nRESearch COMMANDS:")
                    print("  ask <question>           - Get intelligent analysis")
                    print("  summarize <topic>        - Comprehensive topic summary")
                    print("  compare <A> vs <B>       - Compare two concepts")
                    print("  connections <topic>      - Find patterns across books")
                    print("  search <query>           - Basic search")
                    print("  add <text>               - Add document")
                    print("  load <file>              - Load book/document")
                    continue
                
                if user_input.lower().startswith('ask '):
                    question = user_input[4:].strip()
                    if question:
                        response = assistant.ask_with_context(question)
                        print(f"\nAnswer: {response}")
                    else:
                        print("Please provide a question after 'ask'")
                    continue
                
                if user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        response = assistant.search(query)
                        print(f"\nSearch Results:\n{response}")
                    else:
                        print("Please provide a search query after 'search'")
                    continue
                
                if user_input.lower().startswith('summarize '):
                    topic = user_input[10:].strip()
                    if topic:
                        print(f"\nSummarizing '{topic}'...")
                        response = assistant.summarize_topic(topic)
                        print(f"\nSummary:\n{response}")
                    else:
                        print("Please provide a topic after 'summarize'")
                    continue

                if ' vs ' in user_input.lower() and user_input.lower().startswith('compare '):
                    comparison = user_input[8:].strip()
                    if ' vs ' in comparison:
                        concept1, concept2 = comparison.split(' vs ', 1)
                        concept1, concept2 = concept1.strip(), concept2.strip()
                        print(f"\nComparing '{concept1}' vs '{concept2}'...")
                        response = assistant.compare_concepts(concept1, concept2)
                        print(f"\nComparison:\n{response}")
                    else:
                        print("Use format: compare <concept1> vs <concept2>")
                    continue

                if user_input.lower().startswith('connections '):
                    topic = user_input[12:].strip()
                    if topic:
                        print(f"\nFinding connections for '{topic}'...")
                        response = assistant.find_connections(topic)
                        print(f"\nConnections:\n{response}")
                    else:
                        print("Please provide a topic after 'connections'")
                    continue

                if user_input.lower().startswith('add '):
                    text = user_input[4:].strip()
                    if text:
                        assistant.add_documents([text])
                    else:
                        print("Please provide text to add after 'add'")
                    continue

                if user_input.lower().startswith('load '):
                    file_path = user_input[5:].strip()
                    if file_path and os.path.exists(file_path):
                        print(f"Loading {file_path}...")
                        # Simple file loading - you can enhance this
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            assistant.add_documents([content], [{"source": file_path}])
                        except Exception as e:
                            print(f"Error loading file: {e}")
                    else:
                        print("Please provide a valid file path after 'load'")
                    continue

                # Default: treat as a question
                print(f"\nAnalyzing your question...")
                response = assistant.ask_with_context(user_input)
                print(f"\nAnalysis:\n{response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    except Exception as e:
        print(f"Failed to start assistant: {e}")

def main():
    """Main function."""
    print("Simple Research Assistant")
    print("=" * 40)
    
    # Test components first
    if not test_imports():
        print("Missing required packages. Please install:")
        print("  pip install langchain-groq sentence-transformers faiss-cpu")
        return
    
    # Run interactive mode
    interactive_mode()

if __name__ == "__main__":
    main()
