#!/usr/bin/env python3
"""
General RAG MCP Server using FastMCP
Works with your specific RAG pipeline for any type of data
"""

import argparse
import os
import sys
from typing import Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

# Configuration from environment variables
class Config:
    DATA_DIR = os.getenv('RAG_DATA_DIR', './rag_data')
    MCP_HOST = os.getenv('RAG_MCP_HOST', 'localhost')
    MCP_PORT = int(os.getenv('RAG_MCP_PORT', '8000'))
    MCP_TRANSPORT = os.getenv('RAG_MCP_TRANSPORT', 'stdio')
    ENABLE_RAG = os.getenv('RAG_MCP_ENABLE_RAG', 'true').lower() == 'true'
    DEFAULT_SEARCH_LIMIT = int(os.getenv('RAG_DEFAULT_SEARCH_LIMIT', '5'))
    MAX_SEARCH_LIMIT = int(os.getenv('RAG_MAX_SEARCH_LIMIT', '20'))

# Modern MCP imports
try:
    from mcp.server.fastmcp import FastMCP
    print("✅ FastMCP imported successfully")
except ImportError as e:
    print(f"❌ FastMCP import failed: {e}")
    sys.exit(1)

# RAG import
try:
    from rag_pipeline import RAGPipeline
    RAG_AVAILABLE = True
    print("✅ RAG pipeline available")
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG pipeline not found")

# Initialize the FastMCP server
mcp = FastMCP("General-RAG-Server")

# Global variables for RAG
rag_pipeline = None
rag_error = None
data_directory = Config.DATA_DIR

def initialize_rag(data_dir: str):
    """Initialize RAG pipeline"""
    global rag_pipeline, rag_error, data_directory
    data_directory = data_dir
    
    if not RAG_AVAILABLE:
        rag_error = "RAG pipeline module not available"
        return
    
    try:
        print(f"📁 Loading RAG pipeline from {data_dir}...")
        rag_pipeline = RAGPipeline(data_dir)
        print("✅ RAG pipeline loaded successfully")
        
        # Check available capabilities
        capabilities = []
        if hasattr(rag_pipeline, 'search'):
            capabilities.append("search")
        if hasattr(rag_pipeline, 'query_with_llm'):
            capabilities.append("AI querying")
        if hasattr(rag_pipeline, 'list_sources'):
            capabilities.append("source management")
        if hasattr(rag_pipeline, 'ingest_directory'):
            capabilities.append("directory ingestion")
        if hasattr(rag_pipeline, 'ingest_git_repo'):
            capabilities.append("git ingestion")
        if hasattr(rag_pipeline, 'search_commits_by_ticket'):
            capabilities.append("ticket search")
        if hasattr(rag_pipeline, 'reprocess_git_commits'):
            capabilities.append("commit reprocessing")
        if hasattr(rag_pipeline, 'incremental_update'):
            capabilities.append("incremental updates")
            
        print(f"🔧 Available capabilities: {', '.join(capabilities)}")
        
    except Exception as e:
        rag_error = str(e)
        print(f"❌ RAG loading failed: {e}")

def get_content(doc):
    """Get content from document"""
    return doc.get('content', doc.get('text', ''))

def get_file_path(doc):
    """Get file path from document metadata"""
    metadata = doc.get('metadata', {})
    return metadata.get('file_path') or metadata.get('source') or metadata.get('file_name') or 'Unknown file'

def get_source_id(doc):
    """Get source ID from document metadata"""
    metadata = doc.get('metadata', {})
    return metadata.get('source_id') or metadata.get('source') or 'Unknown source'

# =============================================================================
# MCP TOOLS - General purpose tools for any RAG data
# =============================================================================

@mcp.tool()
def ping() -> str:
    """Test server connectivity - returns a simple pong message."""
    return "🏓 Pong! General RAG Server is running and ready to search your data."

@mcp.tool()
def server_status() -> str:
    """Get detailed server status including RAG pipeline information."""
    lines = [
        "🤖 **General RAG Server Status**",
        f"📁 Data Directory: {data_directory}",
        f"🔧 RAG Available: {'✅ Yes' if RAG_AVAILABLE else '❌ No'}",
        f"📦 RAG Loaded: {'✅ Yes' if rag_pipeline else '❌ No'}",
    ]
    
    if rag_error:
        lines.append(f"⚠️  RAG Error: {rag_error}")
    
    if rag_pipeline:
        try:
            # Show available methods
            available_methods = []
            method_checks = [
                ('search', 'Document search'),
                ('query_with_llm', 'AI-powered querying'),
                ('list_sources', 'Source listing'),
                ('ingest_directory', 'Directory ingestion'),
                ('ingest_git_repo', 'Git repository ingestion'),
                ('delete_source', 'Source deletion'),
                ('search_commits_by_ticket', 'Ticket-based commit search'),
                ('reprocess_git_commits', 'Git commit reprocessing'),
                ('incremental_update', 'Incremental source updates')
            ]
            
            for method, description in method_checks:
                if hasattr(rag_pipeline, method):
                    available_methods.append(f"✅ {description}")
                else:
                    available_methods.append(f"❌ {description}")
            
            # Add query_with_context as a derived capability
            if hasattr(rag_pipeline, 'search'):
                available_methods.append("✅ Query with context (search + AI)")
            else:
                available_methods.append("❌ Query with context")
            
            lines.extend([
                "",
                "🔧 **Capabilities:**"
            ] + [f"   {method}" for method in available_methods])
            
            # Show API client status
            has_openai = hasattr(rag_pipeline, 'openai_client') and rag_pipeline.openai_client
            has_anthropic = hasattr(rag_pipeline, 'anthropic_client') and rag_pipeline.anthropic_client
            
            lines.extend([
                "",
                "🤖 **AI Clients:**",
                f"   OpenAI: {'✅ Available' if has_openai else '❌ Not configured'}",
                f"   Anthropic: {'✅ Available' if has_anthropic else '❌ Not configured'}"
            ])
            
            # Get data overview
            if hasattr(rag_pipeline, 'list_sources'):
                sources = rag_pipeline.list_sources()
                total_files = sum(s.get('file_count', 0) for s in sources)
                total_chunks = sum(s.get('chunk_count', 0) for s in sources)
                total_commits = sum(s.get('commit_count', 0) for s in sources)
                
                lines.extend([
                    "",
                    f"📊 **Data Overview:**",
                    f"   📂 Sources: {len(sources)}",
                    f"   📄 Total Files: {total_files}",
                    f"   🧩 Total Chunks: {total_chunks}",
                    f"   📝 Total Commits: {total_commits}"
                ])
                
                if sources:
                    lines.extend([
                        "",
                        "📋 **Available Sources:**"
                    ])
                    for source in sources[:5]:  # Show first 5 sources
                        source_id = source.get('id', 'Unknown')
                        source_type = source.get('type', 'Unknown')
                        file_count = source.get('file_count', 0)
                        chunk_count = source.get('chunk_count', 0)
                        commit_count = source.get('commit_count', 0)
                        
                        if commit_count > 0:
                            lines.append(f"   • {source_id} ({source_type}): {file_count} files, {chunk_count} chunks, {commit_count} commits")
                        else:
                            lines.append(f"   • {source_id} ({source_type}): {file_count} files, {chunk_count} chunks")
                    
                    if len(sources) > 5:
                        lines.append(f"   ... and {len(sources) - 5} more sources")
                    
        except Exception as e:
            lines.append(f"⚠️  Status Error: {str(e)}")
    
    return "\n".join(lines)

@mcp.tool()
def search_documents(query: str, limit: int = None, source_filter: Optional[str] = None) -> str:
    """
    Search through all documents in the RAG pipeline using semantic similarity.
    
    Args:
        query: Search query to find relevant documents
        limit: Maximum number of results to return (1-20)
        source_filter: Optional source ID to limit search to specific source
    """
    if not rag_pipeline:
        if rag_error:
            return f"❌ RAG Error: {rag_error}"
        return "❌ RAG pipeline not loaded"
    
    if not query.strip():
        return "❌ Please provide a search query"
    
    if not hasattr(rag_pipeline, 'search'):
        return "❌ Search functionality not available in this RAG pipeline"
    
    # Set default and limit the results to a reasonable range
    limit = limit or Config.DEFAULT_SEARCH_LIMIT
    limit = max(1, min(limit, Config.MAX_SEARCH_LIMIT))
    
    try:
        results = rag_pipeline.search(query, limit=limit, source_filter=source_filter)
        
        if not results:
            search_info = f" in source '{source_filter}'" if source_filter else ""
            return f"🔍 No results found for '{query}'{search_info}"
        
        lines = [
            f"🔍 **Search Results for:** {query}",
        ]
        
        if source_filter:
            lines.append(f"📂 **Source Filter:** {source_filter}")
        
        lines.extend([
            f"📊 **Found:** {len(results)} relevant documents",
            ""
        ])
        
        for i, doc in enumerate(results, 1):
            metadata = doc.get('metadata', {})
            content_type = metadata.get('content_type', 'file')
            
            if content_type == 'git_commit':
                # Handle git commit results
                commit_hash = metadata.get('commit_hash', 'unknown')[:8]
                author = metadata.get('author_name', 'unknown')
                subject = metadata.get('subject', 'No subject')
                repo_name = metadata.get('repository_name', 'unknown')
                similarity = 1 - doc.get('distance', 1)
                
                lines.extend([
                    f"**{i}. Git Commit {commit_hash}** (similarity: {similarity:.3f})",
                    f"   👤 Author: {author}",
                    f"   📂 Repository: {repo_name}",
                    f"   📝 Subject: {subject}",
                    f"   📄 Preview: {get_content(doc)[:300]}...",
                    ""
                ])
            else:
                # Handle file results
                file_path = get_file_path(doc)
                source_id = get_source_id(doc)
                similarity = 1 - doc.get('distance', 1)
                content = get_content(doc)
                
                content_preview = content[:400] if content else "No content available"
                if len(content) > 400:
                    content_preview += "..."
                
                lines.extend([
                    f"**{i}. {file_path}** (similarity: {similarity:.3f})",
                    f"   📂 Source: {source_id}",
                    f"   📄 Content: {content_preview}",
                    ""
                ])
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Search error: {str(e)}"

@mcp.tool()
def search_commits_by_ticket(ticket_id: str, source_filter: Optional[str] = None) -> str:
    """
    Search for git commits related to a specific ticket ID (e.g., GET-1903, JIRA-123).
    
    Args:
        ticket_id: Ticket ID to search for (e.g., GET-1903)
        source_filter: Optional source ID to limit search to specific source
    """
    if not rag_pipeline:
        if rag_error:
            return f"❌ RAG Error: {rag_error}"
        return "❌ RAG pipeline not loaded"
    
    if not ticket_id.strip():
        return "❌ Please provide a ticket ID"
    
    if not hasattr(rag_pipeline, 'search_commits_by_ticket'):
        return "❌ Ticket-based commit search not available in this RAG pipeline"
    
    try:
        results = rag_pipeline.search_commits_by_ticket(ticket_id, source_filter)
        
        if not results:
            search_info = f" in source '{source_filter}'" if source_filter else ""
            return f"🎫 No commits found for ticket '{ticket_id}'{search_info}"
        
        lines = [
            f"🎫 **Commits for Ticket:** {ticket_id}",
        ]
        
        if source_filter:
            lines.append(f"📂 **Source Filter:** {source_filter}")
        
        lines.extend([
            f"📊 **Found:** {len(results)} related commits",
            ""
        ])
        
        for i, doc in enumerate(results, 1):
            metadata = doc.get('metadata', {})
            commit_hash = metadata.get('commit_hash', 'unknown')[:8]
            author = metadata.get('author_name', 'unknown')
            subject = metadata.get('subject', 'No subject')
            repo_name = metadata.get('repository_name', 'unknown')
            commit_date = metadata.get('commit_date', 'unknown')
            similarity = 1 - doc.get('distance', 1)
            
            lines.extend([
                f"**{i}. Commit {commit_hash}** (relevance: {similarity:.3f})",
                f"   👤 Author: {author}",
                f"   📅 Date: {commit_date}",
                f"   📂 Repository: {repo_name}",
                f"   📝 Subject: {subject}",
                f"   📄 Preview: {get_content(doc)[:300]}...",
                ""
            ])
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Ticket search error: {str(e)}"

@mcp.tool()
def ask_question(question: str, model: str = "claude") -> str:
    """
    Ask a question and get an AI-generated response using the RAG pipeline's LLM integration.
    
    Args:
        question: Question to ask about the documents
        model: AI model to use (claude, openai, gpt)
    """
    if not rag_pipeline:
        if rag_error:
            return f"❌ RAG Error: {rag_error}"
        return "❌ RAG pipeline not loaded"
    
    if not question.strip():
        return "❌ Please provide a question"
    
    if not hasattr(rag_pipeline, 'query_with_llm'):
        return "❌ AI querying not available. This RAG pipeline doesn't have LLM integration configured."
    
    try:
        # Use the RAG pipeline's native LLM querying with model selection
        response = rag_pipeline.query_with_llm(question, model=model)
        
        # Get source documents for transparency
        search_results = rag_pipeline.search(question, limit=3)
        
        result_lines = [
            f"🤖 **Question:** {question}",
            f"🧠 **Model:** {model}",
            "",
            f"**Answer:** {response}",
            ""
        ]
        
        if search_results:
            result_lines.extend([
                f"📚 **Sources consulted ({len(search_results)} documents):**",
                ""
            ])
            
            for i, doc in enumerate(search_results, 1):
                metadata = doc.get('metadata', {})
                content_type = metadata.get('content_type', 'file')
                similarity = 1 - doc.get('distance', 1)
                
                if content_type == 'git_commit':
                    commit_hash = metadata.get('commit_hash', 'unknown')[:8]
                    author = metadata.get('author_name', 'unknown')
                    result_lines.extend([
                        f"{i}. **Commit {commit_hash}** (relevance: {similarity:.3f})",
                        f"   Author: {author}"
                    ])
                else:
                    file_path = get_file_path(doc)
                    source_id = get_source_id(doc)
                    result_lines.extend([
                        f"{i}. **{file_path}** (relevance: {similarity:.3f})",
                        f"   Source: {source_id}"
                    ])
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"❌ AI query error: {str(e)}"

@mcp.tool()
def query_with_context(question: str, max_context_chunks: int = 3) -> str:
    """
    Ask a question and get both the AI response and the source context documents.
    This returns both the answer and the actual context that was used.
    
    Args:
        question: Question to ask about the documents
        max_context_chunks: Maximum number of context chunks to include (1-10)
    """
    if not rag_pipeline:
        if rag_error:
            return f"❌ RAG Error: {rag_error}"
        return "❌ RAG pipeline not loaded"
    
    if not question.strip():
        return "❌ Please provide a question"
    
    # Limit context chunks to reasonable range
    max_context_chunks = max(1, min(max_context_chunks, 10))
    
    try:
        # First, get relevant context documents
        context_docs = rag_pipeline.search(question, limit=max_context_chunks)
        
        if not context_docs:
            return f"❌ No relevant context found for question: {question}"
        
        # Generate AI response if LLM is available
        ai_response = None
        if hasattr(rag_pipeline, 'query_with_llm'):
            try:
                ai_response = rag_pipeline.query_with_llm(question)
            except Exception as e:
                ai_response = f"AI response generation failed: {str(e)}"
        
        # Build comprehensive response
        result_lines = [
            f"🤖 **Question:** {question}",
            "",
        ]
        
        # Add AI response if available
        if ai_response:
            result_lines.extend([
                f"**AI Answer:** {ai_response}",
                ""
            ])
        
        # Add context documents
        result_lines.extend([
            f"📚 **Context Documents ({len(context_docs)} sources):**",
            ""
        ])
        
        for i, doc in enumerate(context_docs, 1):
            metadata = doc.get('metadata', {})
            content_type = metadata.get('content_type', 'file')
            similarity = 1 - doc.get('distance', 1)
            content = get_content(doc)
            
            # Show substantial context
            content_preview = content[:600] if content else "No content available"
            if len(content) > 600:
                content_preview += "..."
            
            if content_type == 'git_commit':
                commit_hash = metadata.get('commit_hash', 'unknown')[:8]
                author = metadata.get('author_name', 'unknown')
                subject = metadata.get('subject', 'No subject')
                
                result_lines.extend([
                    f"**{i}. Git Commit {commit_hash}** (relevance: {similarity:.3f})",
                    f"   👤 Author: {author}",
                    f"   📝 Subject: {subject}",
                    f"   📄 Context: {content_preview}",
                    ""
                ])
            else:
                file_path = get_file_path(doc)
                source_id = get_source_id(doc)
                
                result_lines.extend([
                    f"**{i}. {file_path}** (relevance: {similarity:.3f})",
                    f"   📂 Source: {source_id}",
                    f"   📄 Context: {content_preview}",
                    ""
                ])
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"❌ Context query error: {str(e)}"

@mcp.tool()
def list_sources() -> str:
    """List all data sources in the RAG pipeline."""
    if not rag_pipeline:
        if rag_error:
            return f"❌ RAG Error: {rag_error}"
        return "❌ RAG pipeline not loaded"
    
    if not hasattr(rag_pipeline, 'list_sources'):
        return "❌ Source listing not available in this RAG pipeline"
    
    try:
        sources = rag_pipeline.list_sources()
        
        if not sources:
            return "📁 No sources found in the RAG pipeline"
        
        lines = [f"📁 **Data Sources ({len(sources)} total):**", ""]
        
        total_files = 0
        total_chunks = 0
        total_commits = 0
        
        for source in sources:
            source_id = source.get('id', 'Unknown')
            source_type = source.get('type', 'Unknown')
            file_count = source.get('file_count', 0)
            chunk_count = source.get('chunk_count', 0)
            commit_count = source.get('commit_count', 0)
            last_indexed = source.get('last_indexed', 'Unknown')
            
            total_files += file_count
            total_chunks += chunk_count
            total_commits += commit_count
            
            lines.extend([
                f"📂 **{source_id}** ({source_type})",
                f"   📄 Files: {file_count} | 🧩 Chunks: {chunk_count}",
            ])
            
            if commit_count > 0:
                lines.append(f"   📝 Commits: {commit_count}")
            
            lines.extend([
                f"   🕒 Last indexed: {last_indexed}",
                ""
            ])
        
        # Add summary
        lines.extend([
            "📊 **Summary:**",
            f"   Total files: {total_files}",
            f"   Total chunks: {total_chunks}",
            f"   Total commits: {total_commits}"
        ])
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Error listing sources: {str(e)}"

@mcp.tool()
def ingest_directory(directory_path: str, source_name: Optional[str] = None) -> str:
    """
    Ingest a local directory into the RAG pipeline.
    
    Args:
        directory_path: Path to the directory to ingest
        source_name: Optional name for the source (if not provided, uses directory name)
    """
    if not rag_pipeline:
        return "❌ RAG pipeline not loaded"
    
    if not hasattr(rag_pipeline, 'ingest_directory'):
        return "❌ Directory ingestion not available in this RAG pipeline"
    
    if not directory_path.strip():
        return "❌ Please provide a directory path"
    
    try:
        source_id = rag_pipeline.ingest_directory(directory_path, source_name)
        return f"✅ Successfully ingested directory '{directory_path}' as source: {source_id}"
    except Exception as e:
        return f"❌ Ingestion error: {str(e)}"

@mcp.tool()
def ingest_git_repository(repo_url: str, source_name: Optional[str] = None) -> str:
    """
    Ingest a git repository into the RAG pipeline.
    
    Args:
        repo_url: URL of the git repository to clone and ingest
        source_name: Optional name for the source (if not provided, uses repo name)
    """
    if not rag_pipeline:
        return "❌ RAG pipeline not loaded"
    
    if not hasattr(rag_pipeline, 'ingest_git_repo'):
        return "❌ Git repository ingestion not available in this RAG pipeline"
    
    if not repo_url.strip():
        return "❌ Please provide a repository URL"
    
    try:
        source_id = rag_pipeline.ingest_git_repo(repo_url, source_name)
        return f"✅ Successfully ingested git repository '{repo_url}' as source: {source_id}"
    except Exception as e:
        return f"❌ Ingestion error: {str(e)}"

@mcp.tool()
def reprocess_git_commits(source_id: str) -> str:
    """
    Reprocess Git commits for an existing source (useful after fixing commit processing issues).
    
    Args:
        source_id: ID of the source to reprocess commits for
    """
    if not rag_pipeline:
        return "❌ RAG pipeline not loaded"
    
    if not hasattr(rag_pipeline, 'reprocess_git_commits'):
        return "❌ Git commit reprocessing not available in this RAG pipeline"
    
    if not source_id.strip():
        return "❌ Please provide a source ID"
    
    try:
        commit_count = rag_pipeline.reprocess_git_commits(source_id)
        return f"✅ Successfully reprocessed {commit_count} commits for source: {source_id}"
    except Exception as e:
        return f"❌ Reprocessing error: {str(e)}"

@mcp.tool()
def incremental_update(source_id: str) -> str:
    """
    Perform an incremental update on an existing source (add new commits and files).
    
    Args:
        source_id: ID of the source to update
    """
    if not rag_pipeline:
        return "❌ RAG pipeline not loaded"
    
    if not hasattr(rag_pipeline, 'incremental_update'):
        return "❌ Incremental updates not available in this RAG pipeline"
    
    if not source_id.strip():
        return "❌ Please provide a source ID"
    
    try:
        stats = rag_pipeline.incremental_update(source_id)
        return f"""✅ Incremental update completed for source: {source_id}
📊 Summary:
   • New commits: {stats['new_commits']}
   • Updated files: {stats['updated_files']}
   • New files: {stats['new_files']}
   • Repositories processed: {stats['repositories_processed']}"""
    except Exception as e:
        return f"❌ Update error: {str(e)}"

@mcp.tool()
def delete_source(source_id: str) -> str:
    """
    Delete a source and all its data from the RAG pipeline.
    
    Args:
        source_id: ID of the source to delete
    """
    if not rag_pipeline:
        return "❌ RAG pipeline not loaded"
    
    if not hasattr(rag_pipeline, 'delete_source'):
        return "❌ Source deletion not available in this RAG pipeline"
    
    if not source_id.strip():
        return "❌ Please provide a source ID"
    
    try:
        rag_pipeline.delete_source(source_id)
        return f"✅ Successfully deleted source: {source_id}"
    except Exception as e:
        return f"❌ Deletion error: {str(e)}"

# =============================================================================
# MCP RESOURCES
# =============================================================================

@mcp.resource("rag://status")
def get_server_status():
    """Get detailed server status as JSON."""
    import json
    
    status = {
        "server": "running",
        "server_type": "general_rag",
        "rag_available": RAG_AVAILABLE,
        "rag_loaded": rag_pipeline is not None,
        "rag_error": rag_error,
        "data_directory": data_directory,
        "capabilities": {}
    }
    
    if rag_pipeline:
        try:
            # Check all capabilities including new ones
            capabilities = {}
            capability_methods = [
                'search', 'query_with_llm', 'list_sources', 
                'ingest_directory', 'ingest_git_repo', 'delete_source',
                'search_commits_by_ticket', 'reprocess_git_commits', 'incremental_update'
            ]
            
            for method in capability_methods:
                capabilities[method] = hasattr(rag_pipeline, method)
            
            status["capabilities"] = capabilities
            
            # Get sources info if available
            if hasattr(rag_pipeline, 'list_sources'):
                sources = rag_pipeline.list_sources()
                status["sources"] = {
                    "count": len(sources),
                    "total_files": sum(s.get('file_count', 0) for s in sources),
                    "total_chunks": sum(s.get('chunk_count', 0) for s in sources),
                    "total_commits": sum(s.get('commit_count', 0) for s in sources),
                    "details": sources
                }
            
            # Check API clients
            status["api_clients"] = {
                "openai": hasattr(rag_pipeline, 'openai_client') and rag_pipeline.openai_client is not None,
                "anthropic": hasattr(rag_pipeline, 'anthropic_client') and rag_pipeline.anthropic_client is not None
            }
            
        except Exception as e:
            status["status_error"] = str(e)
    
    return json.dumps(status, indent=2)

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run the MCP server."""
    parser = argparse.ArgumentParser(description="General RAG MCP Server")
    parser.add_argument("--data-dir", default=Config.DATA_DIR, help=f"RAG data directory (default: {Config.DATA_DIR})")
    parser.add_argument("--transport", default=Config.MCP_TRANSPORT, choices=["stdio", "sse"], 
                       help=f"Transport method (default: {Config.MCP_TRANSPORT})")
    parser.add_argument("--no-rag", action="store_true", 
                       default=not Config.ENABLE_RAG,
                       help="Disable RAG functionality")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting General RAG MCP Server")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"🚂 Transport: {args.transport}")
    
    # Initialize RAG if not disabled
    if not args.no_rag:
        initialize_rag(args.data_dir)
    else:
        print("⚠️  RAG functionality disabled")
    
    print("✅ Server configured and ready")
    
    # Run the server
    try:
        print(f"🎯 Starting General RAG MCP server...")
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()