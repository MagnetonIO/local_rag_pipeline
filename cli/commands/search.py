#!/usr/bin/env python3
"""
Search commands for RAG Pipeline CLI  
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def add_search_commands(subparsers):
    """Add search commands to the parser"""
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    search_parser.add_argument("--source", help="Filter by source ID")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query with AI")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--model", default="claude", choices=["claude", "openai"], help="AI model to use")
    query_parser.add_argument("--source", help="Filter by source ID")

def handle_search_command(args):
    """Handle search command"""
    try:
        from rag_pipeline import RAGPipeline
        
        data_dir = Path(args.data_dir)
        rag = RAGPipeline(str(data_dir))
        
        if args.command == "search":
            results = rag.search(args.query, args.limit, args.source)
            
            if not results:
                print(f"No results found for: {args.query}")
                return True
            
            print(f"🔍 Search Results for: {args.query}")
            print(f"Found {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                file_path = metadata.get('file_path', 'Unknown file')
                distance = result.get('distance', 1.0)
                similarity = 1 - distance
                content = result.get('content', '')[:200]
                
                print(f"--- Result {i} ---")
                print(f"📄 File: {file_path}")
                print(f"📊 Similarity: {similarity:.3f}")
                print(f"📝 Content: {content}...")
                print()
        
        elif args.command == "query":
            result = rag.query_with_llm(args.question, args.model, args.source)
            print(f"🤖 AI Response:\n{result}")
        
        return True
        
    except ImportError:
        print("❌ RAG Pipeline module not found")
        return False
    except Exception as e:
        logger.error(f"Error executing search: {e}")
        return False
