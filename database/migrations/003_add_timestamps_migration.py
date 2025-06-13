"""
Add timestamp tracking migration
Adds created_at, updated_at, and processed_at columns for audit trails
"""

import sqlite3
from datetime import datetime
from .migration_manager import Migration, SafeColumnManager


class TimestampsMigration(Migration):
    """Add timestamp tracking to all tables"""
    
    def __init__(self):
        super().__init__()
        self.version = "003"
        self.description = "Add timestamp columns for audit trails"
        self.dependencies = ["002"]
    
    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add timestamp columns"""
        current_time = datetime.now().isoformat()
        
        # Add timestamps to sources table
        SafeColumnManager.add_column_with_default(
            cursor, 'sources', 'created_at', 'TEXT', f"'{current_time}'"
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'sources', 'updated_at', 'TEXT', f"'{current_time}'"
        )
        
        # Add timestamps to files table
        SafeColumnManager.add_column_with_default(
            cursor, 'files', 'created_at', 'TEXT', f"'{current_time}'"
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'files', 'updated_at', 'TEXT', f"'{current_time}'"
        )
        
        # Add timestamps to git_commits table
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'processed_at', 'TEXT', f"'{current_time}'"
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'created_at', 'TEXT', f"'{current_time}'"
        )
        
        # Create indexes on timestamp columns for efficient queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sources_created_at ON sources(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sources_updated_at ON sources(updated_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_updated_at ON files(updated_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_processed_at ON git_commits(processed_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_created_at ON git_commits(created_at)')
    
    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove timestamp indexes (can't remove columns in SQLite easily)"""
        cursor.execute('DROP INDEX IF EXISTS idx_sources_created_at')
        cursor.execute('DROP INDEX IF EXISTS idx_sources_updated_at')
        cursor.execute('DROP INDEX IF EXISTS idx_files_created_at')
        cursor.execute('DROP INDEX IF EXISTS idx_files_updated_at')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_processed_at')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_created_at')
    
    def validate(self, cursor: sqlite3.Cursor) -> bool:
        """Validate timestamp columns were added"""
        tables_and_columns = [
            ('sources', ['created_at', 'updated_at']),
            ('files', ['created_at', 'updated_at']),
            ('git_commits', ['processed_at', 'created_at'])
        ]
        
        for table, expected_columns in tables_and_columns:
            cursor.execute(f"PRAGMA table_info({table})")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            for col in expected_columns:
                if col not in existing_columns:
                    return False
        
        return True
