# MCP RAG Server Setup Guide

## Overview

This MCP (Model Context Protocol) server exposes your local RAG pipeline to Claude Desktop, enabling Claude to search, query, and manage your ingested documents and codebases directly.

## Installation

### 1. Install Dependencies
```bash
# Core MCP package
pip install mcp

# RAG pipeline dependencies (if not already installed)
pip install chromadb sentence-transformers anthropic openai tiktoken

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
# Basic local server
python mcp_rag_server.py --data-dir ./rag_data

# With ngrok tunnel (public access)
python mcp_rag_server.py --data-dir ./rag_data --ngrok

# With authentication
python mcp_rag_server.py --data-dir ./rag_data --ngrok --auth-token your-secret-token
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

**For ngrok/remote access:**
```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/mcp_rag_server.py", "--data-dir", "/path/to/rag_data", "--ngrok"],
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

## Zero-Trust Network Setup

### Option 1: Tailscale (Recommended)

1. **Install Tailscale** on both machines:
   ```bash
   # macOS
   brew install tailscale
   
   # Ubuntu/Debian
   curl -fsSL https://tailscale.com/install.sh | sh
   ```

2. **Start Tailscale** and authenticate:
   ```bash
   sudo tailscale up
   ```

3. **Run MCP server** on your local machine:
   ```bash
   python mcp_rag_server.py --data-dir ./rag_data --port 8000
   ```

4. **Configure Claude Desktop** to use Tailscale IP:
   ```json
   {
     "mcpServers": {
       "rag-server": {
         "command": "python",
         "args": ["/path/to/mcp_rag_server.py", "--data-dir", "/path/to/rag_data"],
         "env": {
           "MCP_SERVER_URL": "http://100.x.x.x:8000"
         }
       }
     }
   }
   ```

### Option 2: Cloudflare Tunnel

1. **Install cloudflared**:
   ```bash
   # macOS
   brew install cloudflare/cloudflare/cloudflared
   
   # Ubuntu/Debian
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. **Authenticate**:
   ```bash
   cloudflared tunnel login
   ```

3. **Create tunnel**:
   ```bash
   cloudflared tunnel create rag-server
   cloudflared tunnel route dns rag-server rag.yourdomain.com
   ```

4. **Start tunnel**:
   ```bash
   cloudflared tunnel run --url http://localhost:8000 rag-server
   ```

## Available Tools

Once configured, Claude will have access to these tools:

### 🔍 **search_documents**
Search for relevant documents using semantic similarity
```
Example: "Search for authentication logic in the codebase"
```

### 💬 **query_with_context**
Ask questions and get AI-powered answers with relevant context
```
Example: "How does the user authentication system work?"
```

### 📁 **ingest_directory**
Add a local directory to the RAG pipeline
```
Example: Ingest "/path/to/my/project"
```

### 🔗 **ingest_git_repo**
Clone and ingest a git repository
```
Example: Ingest "https://github.com/user/repo.git"
```

### 📋 **list_sources**
Show all ingested sources and their statistics

### 🗑️ **delete_source**
Remove a source and all its documents

## Usage Examples

### Basic Queries
```
Claude: "Search my codebase for database connection logic"
Claude: "How does the authentication system work in my project?"
Claude: "Show me all the API endpoints in the ingested code"
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
```

## Security Considerations

### Authentication
- Use `--auth-token` for token-based authentication
- Implement IP whitelisting for production
- Use HTTPS in production environments

### Network Security
- **Local only**: No network exposure (most secure)
- **ngrok**: Temporary public tunnels (good for testing)
- **Tailscale**: Zero-trust mesh network (recommended for teams)
- **Cloudflare**: Enterprise-grade tunnel with DDoS protection

### Data Privacy
- All RAG data stays on your local machine
- Only search results and summaries are sent to Claude
- No source code is transmitted to external APIs
- Vector embeddings remain local

## Troubleshooting

### Common Issues

1. **"MCP server not responding"**
   ```bash
   # Check if server is running
   ps aux | grep mcp_rag_server
   
   # Check logs
   python mcp_rag_server.py --data-dir ./rag_data
   ```

2. **"No documents found"**
   ```bash
   # Verify sources are ingested
   python rag_pipeline.py list
   
   # Re-ingest if needed
   python rag_pipeline.py ingest-dir /path/to/project
   ```

3. **"Authentication failed"**
   - Check API keys in environment variables
   - Verify auth token if using `--auth-token`

### Debug Mode
```bash
# Run with verbose logging
python mcp_rag_server.py --data-dir ./rag_data --log-level DEBUG
```

## Performance Optimization

### For Large Codebases
- Increase chunk size for better context
- Use source filtering for specific projects
- Consider excluding certain file types

### Network Optimization
- Use compression for remote connections
- Implement caching for frequent queries
- Batch multiple operations

## Integration Examples

### VS Code Extension
Create a VS Code extension that uses the MCP server for code analysis and documentation generation.

### CI/CD Integration
Add the RAG pipeline to your CI/CD for automated code review and documentation updates.

### Team Setup
Deploy on a shared server with team access via Tailscale or VPN.

## Next Steps

1. **Ingest your projects**: Start with your most important codebases
2. **Set up secure access**: Use Tailscale or Cloudflare for remote access
3. **Integrate with workflow**: Add to your daily development routine
4. **Scale up**: Consider distributed setup for large teams

## Support

- Check logs for error messages
- Verify all dependencies are installed
- Ensure proper file permissions
- Test network connectivity for remote setups