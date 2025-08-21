# Research Assistant with FAISS Vector Database

A powerful AI research assistant that uses FAISS vector database to store and search through your book collection. Built with LangChain and Google's Generative AI.

## Features

- 🔍 **Semantic Search**: Search through your books using natural language queries
- 📚 **Multi-format Support**: PDF, EPUB, DOCX, and TXT files
- 🤖 **AI Agent**: Intelligent research assistant with memory and reasoning
- 📊 **ArXiv Integration**: Search academic papers directly
- 💾 **Persistent Storage**: FAISS vector database for fast retrieval
- 🖥️ **CLI Interface**: Easy-to-use command line interface

## Project Structure

```bash
Research-Assistant-Agent/
├── src/
│   ├── config.py                 # Configuration settings
│   ├── vector_db/
│   │   ├── __init__.py
│   │   └── faiss_manager.py      # FAISS database management
│   ├── document_processing/
│   │   ├── __init__.py
│   │   └── loaders.py            # Document loaders for various formats
│   ├── agents/
│   │   ├── __init__.py
│   │   └── research_agent.py     # Main research agent
│   └── cli/
│       ├── __init__.py
│       └── main.py               # Command line interface
├── data/
│   ├── books/                    # Your book collection
│   ├── papers/                   # Research papers
│   ├── vector_db/                # FAISS database files
│   └── cache/                    # Temporary files
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Saad-Musema/Research-Assistant-Agent/
cd Research-Assistant-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Google API key
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Usage

#### Interactive Mode (Recommended)
```bash
python main.py
```

#### Command Line Usage
```bash
# Add a book to the database
python main.py add /path/to/your/book.pdf

# Search through your books
python main.py search "machine learning algorithms"

# Ask a research question
python main.py query "What are the main themes in artificial intelligence?"

# Show database information
python main.py info
```

## Usage Examples

### Adding Books
```python
# The system supports multiple formats
python main.py add data/books/ai_textbook.pdf
python main.py add data/books/novel.epub
python main.py add data/books/research_paper.docx
```

### Searching and Querying
```bash
# Semantic search through your collection
python main.py search "neural networks and deep learning"

# Ask complex research questions
python main.py query "Compare different machine learning approaches mentioned in my books"

# Get summaries
python main.py query "Summarize the key concepts about reinforcement learning"
```

## Features in Detail

### Vector Database (FAISS)
- Fast similarity search using Facebook's FAISS library
- Persistent storage of document embeddings
- Automatic chunking and indexing of documents
- Metadata preservation for source tracking

### Document Processing
- **PDF**: Full text extraction with PyMuPDF
- **EPUB**: E-book processing with unstructured
- **DOCX**: Microsoft Word document support
- **TXT**: Plain text file processing

### AI Agent Capabilities
- Natural language understanding
- Memory across conversations
- Tool integration (ArXiv search, book search, comparison)
- Contextual responses based on your book collection

## API Reference

### ResearchAgent Class

```python
from src.agents.research_agent import ResearchAgent

# Initialize agent
agent = ResearchAgent()

# Add a book
agent.add_book("/path/to/book.pdf")

# Search books
results = agent.search_books("machine learning", num_results=5)

# Ask questions
response = agent.run("What are the main themes in my collection?")
```

### FAISSManager Class

```python
from src.vector_db.faiss_manager import FAISSManager

# Initialize database
db = FAISSManager()

# Add documents
db.add_documents(documents, book_id="my_book")

# Search
results = db.search_similar("query", k=5)

# Save database
db.save_database()
```

## Configuration

The system can be configured through environment variables or the `Config` class:

```python
# Environment variables (.env file)
GOOGLE_API_KEY=your_key_here
VECTOR_DB_PATH=data/vector_db
BOOKS_DIR=data/books
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: GOOGLE_API_KEY environment variable is required
   ```
   Solution: Add your Google API key to the `.env` file

2. **File Format Not Supported**
   ```
   Error: Unsupported file format: .xyz
   ```
   Solution: Convert to PDF, EPUB, DOCX, or TXT format

3. **Memory Issues with Large Books**
   - Reduce `CHUNK_SIZE` in config
   - Process books in smaller sections

### Performance Tips

- Use SSD storage for the vector database
- Increase `CHUNK_SIZE` for better context (uses more memory)
- Use GPU version of FAISS for faster search: `pip install faiss-gpu`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Document processing with various open-source libraries
