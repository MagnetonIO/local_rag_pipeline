# MCP RAG Server Setup Guide

## Overview

This MCP (Model Context Protocol) server exposes your local RAG pipeline to Claude Desktop, enabling Claude to search, query, and manage your ingested documents and codebases directly. **Now powered by FastMCP** for improved performance and reliability.

## What's New

### 🚀 **FastMCP Implementation**
- Migrated from traditional MCP to **FastMCP** for better performance
- Simplified server architecture with modern MCP patterns
- Enhanced error handling and reliability

### 🔧 **New Git-Powered Features**
- **Ticket-based commit search**: Find all commits related to specific tickets (e.g., JIRA-123, GET-1903)
- **Git commit reprocessing**: Fix and reprocess commits for existing sources
- **Incremental updates**: Add new commits and files to existing sources without full re-ingestion
- **Enhanced git analysis**: Deep integration with git history and commit metadata

### 📋 **Enhanced Tools**
- `search_commits_by_ticket` - Search commits by ticket ID
- `reprocess_git_commits` - Reprocess git commits for existing sources
- `incremental_update` - Perform incremental updates on existing sources
- Improved `server_status` with detailed capability reporting

## Installation

### 1. Install Dependencies
```bash
# Core MCP package (FastMCP)
pip install mcp

# RAG pipeline dependencies (if not already installed)
pip install chromadb sentence-transformers anthropic openai tiktoken

# Git processing dependencies
pip install gitpython

# Optional: For ngrok tunneling
# Download from https://ngrok.com/download
```

### 2. Install ngrok (Optional - for external access)
```bash
# macOS
brew install ngrok

# Ubuntu/Debian
snap install ngrok

# Or download from https://ngrok.com/download
```

## Quick Start

### 1. Start the MCP Server Locally
```bash
# Basic local server (FastMCP)
python mcp_rag_server.py --data-dir ./rag_data

# With specific transport method
python mcp_rag_server.py --data-dir ./rag_data --transport stdio

# Disable RAG for testing
python mcp_rag_server.py --data-dir ./rag_data --no-rag
```

### 2. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/mcp_rag_server.py", "--data-dir", "/path/to/rag_data"],
      "env": {
        "ANTHROPIC_API_KEY": "your-anthropic-api-key",
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

After updating the configuration, restart Claude Desktop to load the MCP server.

## Available Tools

Once configured, Claude will have access to these tools:

### 🔍 **Core Search Tools**
- **search_documents**: Search for relevant documents using semantic similarity
- **query_with_context**: Ask questions and get AI-powered answers with relevant context
- **ask_question**: Direct AI-powered question answering using the RAG pipeline

### 🔧 **New Git-Powered Tools**
- **search_commits_by_ticket**: Find commits related to specific ticket IDs (e.g., JIRA-123, GET-1903)
- **reprocess_git_commits**: Reprocess git commits for an existing source
- **incremental_update**: Perform incremental updates (add new commits/files) on existing sources

### 📁 **Data Management Tools**
- **ingest_directory**: Add a local directory to the RAG pipeline
- **ingest_git_repository**: Clone and ingest a git repository
- **list_sources**: Show all ingested sources and their statistics
- **delete_source**: Remove a source and all its documents

### 🛠️ **System Tools**
- **ping**: Test server connectivity
- **server_status**: Get detailed server status including capabilities

## Usage Examples

### Basic Queries
```
Claude: "Search my codebase for database connection logic"
Claude: "How does the authentication system work in my project?"
Claude: "Show me all the API endpoints in the ingested code"
```

### New Git-Powered Queries
```
Claude: "Find all commits related to ticket JIRA-1234"
Claude: "Search for commits related to GET-1903"
Claude: "Show me what was changed for bug ticket BUG-567"
Claude: "Reprocess git commits for my main project source"
Claude: "Perform an incremental update on my codebase source"
```

### Management
```
Claude: "Ingest my project directory at /Users/me/my-app"
Claude: "Add this GitHub repo: https://github.com/user/awesome-lib"
Claude: "List all my ingested sources"
Claude: "Delete the old project source"
```

### Advanced Queries
```
Claude: "Find security vulnerabilities in my authentication code"
Claude: "Explain how the database schema relates to the API endpoints"
Claude: "What are the main architectural patterns used in this codebase?"
Claude: "Show me the git history for authentication-related changes"
```

## Configuration Options

### Server Arguments
```bash
# Basic usage
python mcp_rag_server.py --data-dir ./rag_data

# Transport options
python mcp_rag_server.py --data-dir ./rag_data --transport stdio
python mcp_rag_server.py --data-dir ./rag_data --transport sse

# Disable RAG for testing
python mcp_rag_server.py --data-dir ./rag_data --no-rag
```

### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": [
        "/path/to/mcp_rag_server.py",
        "--data-dir", "/path/to/rag_data",
        "--transport", "stdio"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "your-anthropic-api-key",
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

## Git Integration Features

### Ticket-Based Commit Search
The server now supports searching for commits related to specific tickets:
- Automatically detects ticket patterns (JIRA-123, GET-1903, BUG-456, etc.)
- Searches commit messages, branches, and metadata
- Returns relevant commits with full context

### Incremental Updates
Instead of re-ingesting entire repositories:
- Add only new commits since last update
- Process new files and changes
- Maintains existing embeddings and metadata
- Significantly faster for large repositories

### Git Commit Reprocessing
Fix and reprocess commits for existing sources:
- Useful when git processing logic is improved
- Maintains source metadata while updating commit data
- Preserves existing file embeddings

## Security Considerations

### Authentication
- Use environment variables for API keys
- Implement IP whitelisting for production
- Use HTTPS in production environments

### Network Security
- **Local only**: No network exposure (most secure)
- **Tailscale**: Zero-trust mesh network (recommended for teams)
- **Cloudflare**: Enterprise-grade tunnel with DDoS protection

### Data Privacy
- All RAG data stays on your local machine
- Only search results and summaries are sent to Claude
- No source code is transmitted to external APIs
- Vector embeddings remain local

## Troubleshooting

### Common Issues

1. **"FastMCP import failed"**
   ```bash
   # Ensure you have the latest MCP package
   pip install --upgrade mcp
   ```

2. **"MCP server not responding"**
   ```bash
   # Check if server is running
   ps aux | grep mcp_rag_server
   
   # Check logs
   python mcp_rag_server.py --data-dir ./rag_data
   ```

3. **"No documents found"**
   ```bash
   # Verify sources are ingested
   python rag_pipeline.py list
   
   # Re-ingest if needed
   python rag_pipeline.py ingest-dir /path/to/project
   ```

4. **"Git processing failed"**
   ```bash
   # Check git repository access
   git -C /path/to/repo status
   
   # Reprocess git commits
   python rag_pipeline.py reprocess-commits <source_id>
   ```

### Debug Mode
```bash
# Run with verbose logging
python mcp_rag_server.py --data-dir ./rag_data --transport stdio
```

## Performance Optimization

### For Large Codebases
- Use incremental updates instead of full re-ingestion
- Increase chunk size for better context
- Use source filtering for specific projects
- Consider excluding certain file types

### Git Processing Optimization
- Process commits in batches
- Use incremental updates for frequent changes
- Filter by specific branches or date ranges
- Cache git metadata for faster processing

## Migration from Old MCP

If you're upgrading from the old MCP implementation:

1. **Update your configuration** - No changes needed in Claude Desktop config
2. **Restart Claude Desktop** - The new FastMCP server is backward compatible
3. **Test new features** - Try the new git-powered tools
4. **Incremental updates** - Use the new incremental update feature instead of full re-ingestion

## Next Steps

1. **Ingest your projects**: Start with your most important codebases
2. **Try git features**: Use ticket-based commit search for your development workflow
3. **Set up incremental updates**: Keep your sources up-to-date automatically
4. **Integrate with workflow**: Add to your daily development routine
5. **Scale up**: Consider distributed setup for large teams

## Support

- Check logs for error messages
- Verify all dependencies are installed
- Ensure proper file permissions
- Test git repository access for git-powered features
- Use `server_status` tool to check capabilities and configuration