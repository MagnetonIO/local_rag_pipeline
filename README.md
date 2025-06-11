# Local RAG Pipeline Setup

## Installation

1. **Install Python dependencies:**
```bash
pip install chromadb sentence-transformers anthropic openai tiktoken sqlite3
```

2. **Set up API keys** (add to your `.env` file or environment):
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## Quick Start

### 1. Ingest a Local Directory
```bash
python rag_pipeline.py ingest-dir /path/to/your/project --name "my_project"
```

### 2. Ingest a Git Repository
```bash
python rag_pipeline.py ingest-git https://github.com/user/repo.git --name "repo_name"
```

### 3. Query Your Data
```bash
# Using Claude
python rag_pipeline.py query "How does the authentication work?" --model claude

# Using OpenAI
python rag_pipeline.py query "What are the main components?" --model openai

# Filter by specific source
python rag_pipeline.py query "Explain the database schema" --source my_project
```

### 4. Search Without LLM
```bash
python rag_pipeline.py search "authentication logic" --limit 10
```

### 5. Manage Sources
```bash
# List all ingested sources
python rag_pipeline.py list

# Delete a source
python rag_pipeline.py delete source_id_here
```

## Features

### ✅ **Multi-Source Support**
- Local directories
- Git repositories (automatically cloned)
- Remembers git commit hashes for updates

### ✅ **Smart File Processing**
- Supports 25+ file types (Python, JS, Markdown, etc.)
- Ignores common build/cache directories
- Handles large codebases efficiently

### ✅ **Intelligent Chunking**
- Token-aware chunking with overlap
- Preserves context across chunks
- Metadata tracking for each chunk

### ✅ **Dual LLM Support**
- Claude (Anthropic) integration
- OpenAI GPT integration
- Easy model switching

### ✅ **Persistent Memory**
- ChromaDB for vector storage
- SQLite for metadata tracking
- Incremental updates (only reprocess changed files)

### ✅ **Advanced Search**
- Semantic similarity search
- Source filtering
- Configurable result limits

## File Structure

After running, you'll have:
```
rag_data/
├── chroma_db/          # Vector embeddings
├── metadata.db         # File/source metadata
└── repos/             # Cloned git repositories
    └── source_id/
        └── ...
```

## Configuration Options

### Chunk Settings
Modify in the `DocumentProcessor` class:
- `chunk_size`: Maximum tokens per chunk (default: 1000)
- `chunk_overlap`: Token overlap between chunks (default: 200)

### Supported File Types
Currently supports:
- **Code**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`, etc.
- **Documentation**: `.md`, `.txt`, `.rst`, `.org`
- **Config**: `.json`, `.yaml`, `.toml`, `.env`
- **Web**: `.html`, `.css`, `.xml`

### Ignored Patterns
Automatically skips:
- `.git`, `node_modules`, `__pycache__`
- `dist`, `build`, `.venv`, `target`
- Binary files and archives

## API Usage Examples

### Python Integration
```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline("./my_rag_data")

# Ingest sources
source_id = rag.ingest_directory("/path/to/project")
repo_id = rag.ingest_git_repo("https://github.com/user/repo.git")

# Query with context
answer = rag.query_with_llm("How does authentication work?", model="claude")
print(answer)

# Search for specific content
results = rag.search("database connection", n_results=5)
for result in results:
    print(f"File: {result['metadata']['file_path']}")
    print(f"Content: {result['content'][:100]}...")

# List and manage sources
sources = rag.list_sources()
rag.delete_source("old_source_id")
```

## Advanced Features

### Incremental Updates
The pipeline tracks file hashes and only reprocesses changed files, making updates fast and efficient.

### Source Management
- Each source gets a unique ID
- Track git commit hashes
- Easy deletion of entire sources
- Metadata includes file counts and last update times

### Memory Efficiency
- Processes files one at a time
- Configurable chunk sizes
- Automatic cleanup of old chunks when files change

## Troubleshooting

### Common Issues

1. **"anthropic package not installed"**
   ```bash
   pip install anthropic
   ```

2. **"No relevant documents found"**
   - Check if files were ingested: `python rag_pipeline.py list`
   - Verify file types are supported
   - Try broader search terms

3. **Git clone failures**
   - Ensure git is installed and accessible
   - Check repository URL and permissions
   - Network connectivity

4. **API errors**
   - Verify API keys are set correctly
   - Check API quotas and billing
   - Ensure proper model access

### Performance Tips

1. **For large repositories:**
   - Increase chunk size for better context
   - Use source filtering in queries
   - Consider excluding certain file types

2. **For faster ingestion:**
   - Exclude unnecessary directories
   - Use SSD storage for ChromaDB
   - Process smaller batches

3. **For better search results:**
   - Use specific, descriptive queries
   - Combine semantic search with LLM queries
   - Experiment with different chunk sizes