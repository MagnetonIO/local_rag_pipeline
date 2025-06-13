#!/usr/bin/env python3
"""
Status command for RAG Pipeline CLI
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def add_status_commands(subparsers):
    """Add status commands to the parser"""
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument(
        "--show-sources", 
        action="store_true",
        help="Show detailed source information"
    )

def handle_status_command(args):
    """Handle status command"""
    try:
        # Try to import and initialize RAG pipeline
        from rag_pipeline import RAGPipeline
        
        data_dir = Path(args.data_dir)
        rag = RAGPipeline(str(data_dir))
        
        print(f"🤖 RAG Pipeline Status")
        print(f"📁 Data Directory: {data_dir}")
        print(f"✅ RAG Pipeline: Loaded successfully")
        
        if args.show_sources:
            sources = rag.list_sources()
            print(f"\n📊 Sources ({len(sources)} total):")
            
            if not sources:
                print("   No sources found")
            else:
                for source in sources:
                    print(f"   📂 {source['id']} ({source['type']})")
                    print(f"      Files: {source['file_count']}, Chunks: {source['chunk_count']}")
                    if source.get('commit_count', 0) > 0:
                        print(f"      Commits: {source['commit_count']}")
                    print(f"      Last indexed: {source['last_indexed']}")
                    print()
        
        return True
        
    except ImportError:
        print("❌ RAG Pipeline module not found")
        return False
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return False
