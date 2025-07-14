"""
Ingestion command handlers for RAG Pipeline CLI
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ingest_directory(rag_pipeline, path: str, name: Optional[str] = None) -> bool:
    """Ingest a directory of documents"""
    try:
        path = Path(path).resolve()
        if not path.exists():
            logger.error(f"Directory not found: {path}")
            return False
            
        print(f"📁 Ingesting directory: {path}")
        print(f"   Name: {name or path.name}")
        
        source_id = rag_pipeline.ingest_directory(str(path), name)
        
        # Get stats
        sources = rag_pipeline.list_sources()
        source_info = next((s for s in sources if s['id'] == source_id), None)
        
        if source_info:
            print(f"\n✅ Successfully ingested directory")
            print(f"   Source ID: {source_id}")
            print(f"   Files: {source_info['file_count']}")
            print(f"   Chunks: {source_info['chunk_count']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ingest directory: {e}")
        return False


def ingest_file(rag_pipeline, path: str, name: Optional[str] = None) -> bool:
    """Ingest a single file"""
    try:
        path = Path(path).resolve()
        if not path.exists():
            logger.error(f"File not found: {path}")
            return False
            
        print(f"📄 Ingesting file: {path}")
        print(f"   Name: {name or path.name}")
        
        source_id = rag_pipeline.ingest_single_file(str(path), name)
        
        print(f"\n✅ Successfully ingested file")
        print(f"   Source ID: {source_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ingest file: {e}")
        return False


def ingest_git(rag_pipeline, url: str, name: Optional[str] = None) -> bool:
    """Ingest a Git repository"""
    try:
        print(f"📦 Ingesting Git repository: {url}")
        print(f"   Name: {name or url.split('/')[-1].replace('.git', '')}")
        
        source_id = rag_pipeline.ingest_git_repo(url, name)
        
        # Get stats
        sources = rag_pipeline.list_sources()
        source_info = next((s for s in sources if s['id'] == source_id), None)
        
        if source_info:
            print(f"\n✅ Successfully ingested repository")
            print(f"   Source ID: {source_id}")
            print(f"   Files: {source_info['file_count']}")
            print(f"   Chunks: {source_info['chunk_count']}")
            print(f"   Commits: {source_info.get('commit_count', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ingest repository: {e}")
        return False