"""
Search command handlers for RAG Pipeline CLI
"""

import logging
from typing import Optional
from cli.config import config

logger = logging.getLogger(__name__)


def search(rag_pipeline, query: str, limit: int = None, source_id: Optional[str] = None) -> bool:
    """Search for documents using semantic search"""
    try:
        if limit is None:
            limit = config.DEFAULT_SEARCH_LIMIT
        limit = max(1, min(limit, config.MAX_SEARCH_LIMIT))  # Clamp to valid range
        
        print(f"🔍 Searching for: {query}")
        if source_id:
            print(f"   Filtering by source: {source_id}")
        
        results = rag_pipeline.search(query, limit, source_id)
        
        if not results:
            print("\nNo results found")
            return True
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            file_path = metadata.get('file_path', 'Unknown')
            distance = result.get('distance', 1.0)
            similarity = 1 - distance
            content = result.get('content', '')
            
            # Truncate content for display
            preview = content[:200] + "..." if len(content) > 200 else content
            
            print(f"{'='*60}")
            print(f"Result {i}:")
            print(f"  📄 File: {file_path}")
            print(f"  📊 Similarity: {similarity:.1%}")
            print(f"  📝 Content:\n    {preview.replace(chr(10), chr(10) + '    ')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return False


def query(rag_pipeline, question: str, model: str = None, source_id: Optional[str] = None) -> bool:
    """Query using AI with context from the knowledge base"""
    try:
        model = model or config.DEFAULT_AI_MODEL
        
        print(f"🤖 Asking {model.upper()}: {question}")
        if source_id:
            print(f"   Using source: {source_id}")
        
        print("\n⏳ Thinking...\n")
        
        response = rag_pipeline.query_with_llm(question, model, source_id)
        
        print("💡 Answer:")
        print("─" * 60)
        print(response)
        print("─" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return False


def search_ticket(rag_pipeline, ticket_id: str, limit: int = None, source_id: Optional[str] = None) -> bool:
    """Search for Git commits by ticket ID"""
    try:
        if limit is None:
            limit = config.DEFAULT_TICKET_SEARCH_LIMIT
        limit = max(1, min(limit, config.MAX_SEARCH_LIMIT))  # Clamp to valid range
        
        print(f"🎫 Searching commits for ticket: {ticket_id}")
        if source_id:
            print(f"   In source: {source_id}")
        
        commits = rag_pipeline.search_commits_by_ticket(ticket_id, limit, source_id)
        
        if not commits:
            print("\nNo commits found for this ticket")
            return True
        
        print(f"\nFound {len(commits)} commits:\n")
        
        for commit in commits:
            print(f"{'='*60}")
            print(f"📝 {commit['sha'][:8]} - {commit['author']}")
            print(f"📅 {commit['date']}")
            print(f"💬 {commit['message'].strip()}")
            
            if commit.get('files_changed'):
                print(f"\n📁 Files changed:")
                for file in commit['files_changed'][:5]:  # Show first 5 files
                    print(f"   - {file}")
                if len(commit['files_changed']) > 5:
                    print(f"   ... and {len(commit['files_changed']) - 5} more")
        
        return True
        
    except Exception as e:
        logger.error(f"Ticket search failed: {e}")
        return False