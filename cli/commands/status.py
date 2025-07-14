"""
Status command handlers for RAG Pipeline CLI
"""

import logging
from cli.config import config

logger = logging.getLogger(__name__)


def status(rag_pipeline, data_dir, show_sources: bool = False) -> bool:
    """Show system status"""
    try:
        print("🤖 RAG Pipeline Status")
        print("─" * 60)
        print(f"📁 Data Directory: {data_dir}")
        print(f"✅ System: Online")
        
        # Get database info
        from pathlib import Path
        db_path = Path(data_dir) / config.DATABASE_NAME
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            print(f"💾 Database: {size_mb:.1f} MB")
        
        # Get vector store info
        vector_path = Path(data_dir) / config.VECTOR_STORE_DIR
        if vector_path.exists():
            print(f"🔍 Vector Store: Active")
        
        if show_sources and rag_pipeline:
            print("\n")
            from cli.commands.maintenance import list_sources
            list_sources(rag_pipeline)
        elif rag_pipeline:
            sources = rag_pipeline.list_sources()
            print(f"\n📊 Sources: {len(sources)} total")
            print("   (Use 'list' command for details)")
        else:
            print(f"\n📊 Sources: Not loaded (use 'rag status --show-sources' to initialize)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return False