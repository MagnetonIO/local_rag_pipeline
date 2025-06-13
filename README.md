# Local RAG Pipeline Setup

## Installation

1. **Install Python dependencies:**
```bash
pip install chromadb sentence-transformers anthropic openai tiktoken sqlite3 mcp
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

### 2. Ingest a Single File
```bash
python rag_pipeline.py ingest-file /path/to/document.py --name "single_doc"
```

### 3. Ingest a Git Repository
```bash
python rag_pipeline.py ingest-git https://github.com/user/repo.git --name "repo_name"
```

### 4. Query Your Data
```bash
# Using Claude
python rag_pipeline.py query "How does the authentication work?" --model claude

# Using OpenAI
python rag_pipeline.py query "What are the main components?" --model openai

# Filter by specific source
python rag_pipeline.py query "Explain the database schema" --source my_project
```

### 5. Search Without LLM
```bash
python rag_pipeline.py search "authentication logic" --limit 10
```

### 6. Git-Specific Features
```bash
# Search commits by ticket ID
python rag_pipeline.py search-ticket "GET-1903" --source repo_name

# Incremental update (pull new commits and files)
python rag_pipeline.py incremental-update source_id

# Reprocess Git commits for existing source
python rag_pipeline.py reprocess-commits source_id
```

### 7. Manage Sources
```bash
# List all ingested sources
python rag_pipeline.py list

# Delete a source
python rag_pipeline.py delete source_id_here
```

## MCP Server Integration

The RAG pipeline now includes **Model Context Protocol (MCP) server** support, allowing direct integration with Claude Desktop and other MCP-compatible clients.

### Start MCP Server
```bash
# Basic MCP server
python mcp_rag_server.py --data-dir ./rag_data

# With different transport
python mcp_rag_server.py --transport stdio --data-dir ./rag_data

# Disable RAG functionality (server-only mode)
python mcp_rag_server.py --no-rag
```

### Claude Desktop Integration
Add to your Claude Desktop MCP configuration:
```json
{
  "mcpServers": {
    "rag-pipeline": {
      "command": "python",
      "args": ["/path/to/mcp_rag_server.py", "--data-dir", "/path/to/rag_data"]
    }
  }
}
```

### Available MCP Tools
- `search_documents` - Semantic search through documents
- `search_commits_by_ticket` - Find commits by ticket ID
- `ask_question` - AI-powered Q&A with context
- `query_with_context` - Get both answer and source context
- `list_sources` - Show all ingested sources
- `ingest_directory` - Add local directories
- `ingest_git_repository` - Clone and ingest repos
- `incremental_update` - Update existing sources
- `reprocess_git_commits` - Reprocess commit data
- `delete_source` - Remove sources
- `server_status` - Server health and capabilities

## Features

### ✅ **Multi-Source Support**
- Local directories
- Single files
- Git repositories (automatically cloned)
- Remembers git commit hashes for updates

### ✅ **Advanced Git Integration**
- Commit-level processing and search
- Ticket ID extraction from commit messages
- Incremental updates (pull new commits only)
- Commit reprocessing for enhanced data
- Git repository metadata tracking

### ✅ **Smart File Processing**
- Supports 25+ file types (Python, JS, Markdown, etc.)
- Ignores common build/cache directories
- Handles large codebases efficiently
- Single file ingestion support

### ✅ **Intelligent Chunking**
- Token-aware chunking with overlap
- Preserves context across chunks
- Metadata tracking for each chunk
- Commit-aware chunking for Git repos

### ✅ **Dual LLM Support**
- Claude (Anthropic) integration
- OpenAI GPT integration
- Easy model switching
- Context-aware responses

### ✅ **Persistent Memory**
- ChromaDB for vector storage
- SQLite for metadata tracking
- Incremental updates (only reprocess changed files)
- Git commit tracking and deduplication

### ✅ **Advanced Search**
- Semantic similarity search
- Source filtering
- Ticket-based commit search
- Configurable result limits
- Context-aware querying

### ✅ **MCP Server Integration**
- FastMCP server implementation
- Claude Desktop compatibility
- Real-time document access
- Comprehensive tool ecosystem

## File Structure

After running, you'll have:
```
rag_data/
├── chroma_db/          # Vector embeddings
├── metadata.db         # File/source metadata + Git commits
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
file_id = rag.ingest_single_file("/path/to/document.py")
repo_id = rag.ingest_git_repo("https://github.com/user/repo.git")

# Query with context
answer = rag.query_with_llm("How does authentication work?", model="claude")
print(answer)

# Search for specific content
results = rag.search("database connection", n_results=5)
for result in results:
    print(f"File: {result['metadata']['file_path']}")
    print(f"Content: {result['content'][:100]}...")

# Git-specific features
commits = rag.search_commits_by_ticket("GET-1903")
update_stats = rag.incremental_update("repo_source_id")

# List and manage sources
sources = rag.list_sources()
rag.delete_source("old_source_id")
```

### MCP Integration
```python
# The MCP server exposes all functionality through tools
# Use with Claude Desktop or other MCP clients
# Tools are automatically available when server is running
```

## Advanced Features

### Git Repository Management
- **Incremental Updates**: Only process new commits since last update
- **Commit Metadata**: Extract ticket IDs, author info, and file changes
- **Repository Tracking**: Maintain git commit hashes and branch info
- **Ticket Search**: Find specific commits by JIRA/GitHub issue numbers

### Enhanced Search Capabilities
- **Semantic Search**: Vector similarity across all document types
- **Commit Search**: Find commits by ticket ID or content
- **Source Filtering**: Limit searches to specific repositories or directories
- **Context Preservation**: Maintain document structure and relationships

### Memory Efficiency
- Processes files one at a time
- Configurable chunk sizes
- Automatic cleanup of old chunks when files change
- Efficient Git commit deduplication

### MCP Server Features
- **Real-time Integration**: Direct access from Claude Desktop
- **Comprehensive Tools**: Full RAG pipeline access via MCP protocol
- **Status Monitoring**: Server health and capability reporting
- **Error Handling**: Graceful degradation and error reporting

## Troubleshooting

### Common Issues

1. **"mcp package not installed"**
   ```bash
   pip install mcp
   ```

2. **"anthropic package not installed"**
   ```bash
   pip install anthropic
   ```

3. **"No relevant documents found"**
   - Check if files were ingested: `python rag_pipeline.py list`
   - Verify file types are supported
   - Try broader search terms

4. **Git clone failures**
   - Ensure git is installed and accessible
   - Check repository URL and permissions
   - Network connectivity

5. **API errors**
   - Verify API keys are set correctly
   - Check API quotas and billing
   - Ensure proper model access

6. **MCP Connection Issues**
   - Verify Claude Desktop MCP configuration
   - Check server startup logs
   - Ensure correct data directory paths

## Migration and Upgrades

When upgrading from older versions:
- Run database migrations if available
- Check for new CLI commands and options
- Update MCP server configuration if using Claude Desktop
- Review new Git-specific features for development workflows