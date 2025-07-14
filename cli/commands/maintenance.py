"""
Maintenance command handlers for RAG Pipeline CLI
"""

import json
import logging

logger = logging.getLogger(__name__)


def reprocess_commits(rag_pipeline, source_id: str) -> bool:
    """Reprocess Git commits for a source"""
    try:
        print(f"🔄 Reprocessing commits for source: {source_id}")
        
        success = rag_pipeline.reprocess_git_commits(source_id)
        
        if success:
            print("\n✅ Successfully reprocessed commits")
        else:
            print("\n❌ Failed to reprocess commits")
        
        return success
        
    except Exception as e:
        logger.error(f"Reprocessing failed: {e}")
        return False


def incremental_update(rag_pipeline, source_id: str) -> bool:
    """Incrementally update a source with new content"""
    try:
        print(f"🔄 Updating source: {source_id}")
        
        stats = rag_pipeline.incremental_update(source_id)
        
        print(f"\n✅ Update complete:")
        print(f"   New files: {stats.get('new_files', 0)}")
        print(f"   Updated files: {stats.get('updated_files', 0)}")
        print(f"   Deleted files: {stats.get('deleted_files', 0)}")
        print(f"   New commits: {stats.get('new_commits', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Update failed: {e}")
        return False


def list_sources(rag_pipeline) -> bool:
    """List all data sources"""
    try:
        sources = rag_pipeline.list_sources()
        
        if not sources:
            print("📭 No sources found")
            return True
        
        print(f"📊 Data Sources ({len(sources)} total):\n")
        
        for source in sources:
            print(f"{'='*60}")
            print(f"📂 {source['id']}")
            print(f"   Type: {source['type']}")
            print(f"   Files: {source['file_count']:,}")
            print(f"   Chunks: {source['chunk_count']:,}")
            
            if source.get('commit_count', 0) > 0:
                print(f"   Commits: {source['commit_count']:,}")
            
            print(f"   Created: {source.get('created_at', 'Unknown')}")
            print(f"   Updated: {source.get('last_indexed', 'Unknown')}")
            
            if source.get('metadata'):
                meta = source['metadata']
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except:
                        pass
                if isinstance(meta, dict) and meta.get('url'):
                    print(f"   URL: {meta['url']}")
        
        print(f"\n📈 Total storage: {sum(s['chunk_count'] for s in sources):,} chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        return False


def delete_source(rag_pipeline, source_id: str, confirm: bool = False) -> bool:
    """Delete a source and all its data"""
    try:
        # Get source info first
        sources = rag_pipeline.list_sources()
        source = next((s for s in sources if s['id'] == source_id), None)
        
        if not source:
            print(f"❌ Source not found: {source_id}")
            return False
        
        print(f"⚠️  About to delete source: {source_id}")
        print(f"   Type: {source['type']}")
        print(f"   Files: {source['file_count']:,}")
        print(f"   Chunks: {source['chunk_count']:,}")
        
        if not confirm:
            response = input("\nAre you sure? Type 'yes' to confirm: ")
            if response.lower() != 'yes':
                print("❌ Deletion cancelled")
                return True
        
        print("\n🗑️  Deleting source...")
        rag_pipeline.delete_source(source_id)
        
        print("✅ Source deleted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete source: {e}")
        return False