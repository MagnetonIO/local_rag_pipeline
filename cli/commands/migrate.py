#!/usr/bin/env python3
"""
Migration commands for RAG Pipeline CLI
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

def add_migrate_commands(subparsers):
    """Add migration commands to the parser"""
    migrate_parser = subparsers.add_parser("migrate", help="Database migrations")
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_type", help="Migration type")
    
    # Migration status
    status_parser = migrate_subparsers.add_parser("status", help="Show migration status")
    status_parser.add_argument("--db-path", help="Database path")
    
    # Run migrations
    up_parser = migrate_subparsers.add_parser("up", help="Run migrations")
    up_parser.add_argument("--db-path", help="Database path")

def handle_migrate_command(args):
    """Handle migration command"""
    try:
        db_path = Path(args.db_path) if args.db_path else Path(args.data_dir) / "metadata.db"
        
        if args.migrate_type == "status":
            check_migration_status(db_path)
            
        elif args.migrate_type == "up":
            run_migrations(db_path)
            
        else:
            print("❌ Please specify migration type: status or up")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Migration error: {e}")
        return False

def check_migration_status(db_path):
    """Check migration status"""
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("💡 Run 'migrate up' to create and initialize the database")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if migration table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='migrations'
        """)
        
        if not cursor.fetchone():
            print("❌ Migration table not found")
            print("💡 Run 'migrate up' to initialize migrations")
            conn.close()
            return
        
        # Check applied migrations
        cursor.execute("SELECT version, applied_at FROM migrations ORDER BY version")
        migrations = cursor.fetchall()
        
        if not migrations:
            print("❌ No migrations applied")
        else:
            print(f"✅ Database migration status:")
            for version, applied_at in migrations:
                print(f"   Version {version}: Applied at {applied_at}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking migration status: {e}")

def run_migrations(db_path):
    """Run database migrations"""
    print(f"🔄 Running migrations on: {db_path}")
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize RAG pipeline (this will create/migrate the database)
        from rag_pipeline import RAGPipeline
        
        # This will handle database initialization and migrations
        rag = RAGPipeline(str(db_path.parent))
        print("✅ Database migrations completed successfully")
        
    except ImportError:
        print("❌ RAG Pipeline module not found")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
