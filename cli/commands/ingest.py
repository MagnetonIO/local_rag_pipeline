#!/usr/bin/env python3
"""
Ingest commands for RAG Pipeline CLI
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def add_ingest_commands(subparsers):
    """Add ingest commands to the parser"""
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data sources")
    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_type", help="Ingest type")
    
    # Directory ingestion
    dir_parser = ingest_subparsers.add_parser("directory", help="Ingest directory")
    dir_parser.add_argument("path", help="Directory path to ingest")
    dir_parser.add_argument("--name", help="Source name")
    
    # Git repository ingestion
    git_parser = ingest_subparsers.add_parser("git", help="Ingest git repository")
    git_parser.add_argument("url", help="Git repository URL")
    git_parser.add_argument("--name", help="Source name")
    
    # File ingestion
    file_parser = ingest_subparsers.add_parser("file", help="Ingest single file")
    file_parser.add_argument("path", help="File path to ingest")
    file_parser.add_argument("--name", help="Source name")

def handle_ingest_command(args):
    """Handle ingest command"""
    try:
        from rag_pipeline import RAGPipeline
        
        data_dir = Path(args.data_dir)
        rag = RAGPipeline(str(data_dir))
        
        if args.ingest_type == "directory":
            print(f"📁 Ingesting directory: {args.path}")
            source_id = rag.ingest_directory(args.path, args.name)
            print(f"✅ Successfully ingested as source: {source_id}")
            
        elif args.ingest_type == "git":
            print(f"📦 Ingesting git repository: {args.url}")
            source_id = rag.ingest_git_repo(args.url, args.name)
            print(f"✅ Successfully ingested as source: {source_id}")
            
        elif args.ingest_type == "file":
            print(f"📄 Ingesting file: {args.path}")
            source_id = rag.ingest_single_file(args.path, args.name)
            print(f"✅ Successfully ingested as source: {source_id}")
            
        else:
            print("❌ Please specify ingest type: directory, git, or file")
            return False
        
        return True
        
    except ImportError:
        print("❌ RAG Pipeline module not found")
        return False
    except Exception as e:
        logger.error(f"Error ingesting: {e}")
        return False
