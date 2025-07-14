# RAG Pipeline - Professional Document Intelligence System

A comprehensive CLI tool for ingesting, indexing, and querying documents with AI-powered intelligence. Features semantic search, Git integration, and Model Context Protocol (MCP) support.

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up the CLI for system-wide access:**
```bash
chmod +x setup.sh
./setup.sh
```

3. **Configure environment variables:**
```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your API keys and preferences
# At minimum, set:
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## 📋 Command Overview

### Ingestion Commands
```bash
# Ingest a directory of documents
rag ingest-dir ./docs --name "project-docs"

# Ingest a single file
rag ingest-file ./document.pdf --name "important-doc"

# Ingest a Git repository with full commit history
rag ingest-git https://github.com/user/repo --name "my-project"
```

### Search & Query Commands
```bash
# Semantic search across all documents
rag search "authentication flow" --limit 10

# Ask questions using AI (Claude or OpenAI)
rag query "How does the login system work?" --model claude

# Search Git commits by ticket ID
rag search-ticket "JIRA-123" --source my-project
```

### Analysis Commands
```bash
# Analyze LaTeX document structure
rag latex-structure my-latex-docs
```

### Maintenance Commands
```bash
# List all data sources
rag list

# Update a source with new content
rag incremental-update my-project

# Reprocess Git commits for enhanced metadata
rag reprocess-commits my-project

# Delete a source and all its data
rag delete old-source --confirm
```

### System Commands
```bash
# Show system status
rag status --show-sources

# Database migrations
rag migrate status
rag migrate up
```

## 🏗️ Architecture

### Professional CLI Structure
The system is built with a clean, modular architecture:

```
main.py                    # Unified CLI entry point with all commands
├── RAGPipelineCLI        # Professional CLI interface class
│   ├── setup_system()    # Automatic initialization and migration
│   ├── Ingestion methods # ingest_directory(), ingest_file(), ingest_git()
│   ├── Search methods    # search(), query(), search_ticket()
│   ├── Analysis methods  # latex_structure()
│   └── Maintenance       # list_sources(), delete_source(), etc.
│
rag_pipeline.py           # Core RAG pipeline implementation
├── DocumentProcessor     # Smart document chunking and processing
├── GitCommitProcessor    # Git-specific functionality
└── RAGPipeline          # Main pipeline orchestrator

database/                 # Database management
├── database_manager.py   # Schema migrations
└── migrations/          # Version-controlled migrations
```

### Data Storage
```
rag_data/                # Default data directory (~/.rag_pipeline)
├── metadata.db         # SQLite database for metadata
├── vector_store/       # ChromaDB vector embeddings
└── repos/             # Cloned Git repositories
    └── [source_id]/
```

## ✨ Key Features

### 📁 Multi-Source Support
- **Local Directories**: Recursively process entire project directories
- **Single Files**: Ingest individual documents
- **Git Repositories**: Clone and process with full commit history
- **Smart Filtering**: Automatically skips build artifacts, caches, etc.

### 🔍 Advanced Search Capabilities
- **Semantic Search**: Find documents by meaning, not just keywords
- **AI-Powered Q&A**: Get intelligent answers with context
- **Commit Search**: Find Git commits by ticket IDs
- **Source Filtering**: Limit searches to specific sources

### 🔧 Git Integration
- **Full Commit History**: Process and search through all commits
- **Ticket ID Extraction**: Automatically extract JIRA/GitHub issue numbers
- **Incremental Updates**: Only process new commits and changes
- **File Change Tracking**: See which files were modified in each commit

### 📊 Smart Processing
- **25+ File Types**: Support for code, docs, configs, and more
- **Token-Aware Chunking**: Intelligent document splitting with overlap
- **Metadata Preservation**: Track source, path, and context for each chunk
- **LaTeX Support**: Special handling for academic documents

### 🤖 AI Integration
- **Claude (Anthropic)**: Advanced reasoning and code understanding
- **OpenAI GPT**: Alternative AI model support
- **Context-Aware**: Answers include relevant source information
- **Model Switching**: Easy switching between AI providers

## 🛠️ Configuration

The RAG Pipeline is highly configurable through environment variables. Copy `.env.example` to `.env` and customize as needed.

### Essential Environment Variables
```bash
# API Keys (Required)
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
export OPENAI_API_KEY="sk-your-openai-key-here"

# Data Storage
export RAG_DATA_DIR="./rag_data"              # Main data directory
export RAG_DATABASE_NAME="metadata.db"        # Database filename
export RAG_VECTOR_STORE_DIR="chroma_db"       # Vector store directory
```

### AI Model Configuration
```bash
# Model Selection
export RAG_DEFAULT_AI_MODEL="claude"          # Default AI model (claude/openai)
export RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2" # Embedding model
export RAG_CLAUDE_MODEL="claude-3-sonnet-20240229"
export RAG_OPENAI_MODEL="gpt-3.5-turbo"
export RAG_AI_MAX_TOKENS="1000"               # Max tokens for responses
```

### Document Processing
```bash
# Chunk Settings
export RAG_CHUNK_SIZE="1000"                  # Standard chunk size (tokens)
export RAG_CHUNK_OVERLAP="200"                # Standard overlap (tokens)
export RAG_LATEX_CHUNK_SIZE="2000"            # LaTeX chunk size (tokens)
export RAG_LATEX_CHUNK_OVERLAP="300"          # LaTeX overlap (tokens)
export RAG_MAX_FILE_SIZE="10485760"           # Max file size (10MB)
```

### Git Processing
```bash
# Git Configuration
export RAG_MAX_COMMITS="1000"                 # Max commits to process
export RAG_GIT_LOG_TIMEOUT="60"               # Git log timeout (seconds)
export RAG_GIT_CLONE_TIMEOUT="300"            # Git clone timeout (seconds)
```

### Search Configuration
```bash
# Search Limits
export RAG_DEFAULT_SEARCH_LIMIT="5"           # Default search results
export RAG_DEFAULT_TICKET_SEARCH_LIMIT="10"   # Default ticket search results
export RAG_MAX_SEARCH_LIMIT="20"              # Maximum search results
```

### File Types and Patterns
```bash
# Customizable file types (comma-separated, no dots)
export RAG_SUPPORTED_EXTENSIONS="py,js,ts,java,cpp,md,txt,json,yaml"

# Directories to ignore (comma-separated)
export RAG_IGNORE_DIRECTORIES="node_modules,__pycache__,venv,.git,build,dist"
```

### MCP Server Configuration
```bash
# MCP Settings
export RAG_MCP_HOST="localhost"
export RAG_MCP_PORT="8000"
export RAG_MCP_TRANSPORT="stdio"              # stdio, ws, or sse
export RAG_MCP_ENABLE_RAG="true"              # Enable/disable RAG features
```

### Performance and Security
```bash
# Performance Tuning
export RAG_PROCESSING_THREADS="4"
export RAG_VECTOR_BATCH_SIZE="100"

# Security
export RAG_ALLOW_EXECUTABLE_FILES="false"
export RAG_MAX_PATH_DEPTH="10"

# Development
export RAG_VERBOSE="false"
export RAG_LOG_LEVEL="INFO"                   # DEBUG, INFO, WARNING, ERROR
```

For a complete list of all available environment variables, see `.env.example`.

### Supported File Types
- **Code**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`, `.c`, `.h`, `.cs`, `.php`, `.rb`, `.swift`, `.kt`, `.scala`, `.r`, `.m`, `.sh`, `.bash`
- **Markup**: `.md`, `.rst`, `.org`, `.tex`, `.html`, `.xml`
- **Config**: `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.conf`, `.env`
- **Docs**: `.txt`, `.css`

### Ignored Patterns
Automatically skips:
- Version control: `.git`, `.svn`
- Dependencies: `node_modules`, `vendor`, `target`
- Build outputs: `dist`, `build`, `out`
- Caches: `__pycache__`, `.pytest_cache`
- Virtual envs: `.venv`, `venv`, `.env`

## 🔌 MCP Server Integration

The RAG pipeline includes Model Context Protocol (MCP) server support for integration with Claude Desktop and other MCP clients.

### Starting the MCP Server
```bash
python mcp_rag_server.py --data-dir ./rag_data
```

### Claude Desktop Configuration
Add to your Claude Desktop MCP settings:
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
- `search_documents` - Semantic search
- `ask_question` - AI-powered Q&A
- `search_commits_by_ticket` - Git commit search
- `list_sources` - View all sources
- `ingest_directory` - Add directories
- `ingest_git_repository` - Clone repos
- And more...

## 📚 API Usage

### Python Integration
```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline("./rag_data")

# Ingest sources
source_id = rag.ingest_directory("/path/to/project", "my-project")
repo_id = rag.ingest_git_repo("https://github.com/user/repo", "repo-name")

# Search and query
results = rag.search("authentication", limit=5)
answer = rag.query_with_llm("How does the login work?", model="claude")

# Git-specific features
commits = rag.search_commits_by_ticket("JIRA-123")
stats = rag.incremental_update(repo_id)

# Manage sources
sources = rag.list_sources()
rag.delete_source("old-source")
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API key errors**
   - Verify keys are set in environment
   - Check API quotas and billing
   - Ensure model access permissions

3. **No search results**
   - Run `rag list` to verify ingestion
   - Check file types are supported
   - Try broader search terms

4. **Git clone failures**
   - Verify repository URL
   - Check network connectivity
   - Ensure git is installed

5. **Permission errors**
   - Run setup.sh with appropriate permissions
   - Check data directory access

## 🔄 Upgrading

When upgrading to a new version:

1. **Backup your data**
   ```bash
   cp -r ~/.rag_pipeline ~/.rag_pipeline.backup
   ```

2. **Run migrations**
   ```bash
   rag migrate status
   rag migrate up
   ```

3. **Update dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## 📄 License

This project is part of the Magneton AI ecosystem. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tests pass (when available)
- Documentation is updated
- Git commits are descriptive

For major changes, please open an issue first to discuss the proposed changes.