"""
Add enhanced Git support migration
Adds repository-specific tracking and additional Git features
"""

import sqlite3
from .migration_manager import Migration, SafeColumnManager


class GitSupportMigration(Migration):
    """Enhanced Git support migration"""
    
    def __init__(self):
        super().__init__()
        self.version = "002"
        self.description = "Add enhanced Git support with repository tracking"
        self.dependencies = ["001"]
    
    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add Git support enhancements"""
        
        # Add Git-related columns to sources table
        SafeColumnManager.add_column_with_default(
            cursor, 'sources', 'commit_count', 'INTEGER', '0'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'sources', 'last_commit_processed', 'TEXT'
        )
        
        # Add repository tracking to git_commits table
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'repository_path', 'TEXT', "''"
        )
        
        # Add file size tracking to files table
        SafeColumnManager.add_column_with_default(
            cursor, 'files', 'file_size', 'INTEGER'
        )
        
        # Create additional indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sources_last_indexed ON sources(last_indexed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_modified ON files(last_modified)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_repo_path ON git_commits(repository_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_ticket_ids ON git_commits(ticket_ids)')
        
        # Create repository-specific unique constraint
        cursor.execute('DROP INDEX IF EXISTS idx_commits_unique_basic')
        cursor.execute('''
            CREATE UNIQUE INDEX idx_commits_unique 
            ON git_commits(source_id, repository_path, commit_hash)
        ''')
        
        # Create composite index for efficient queries
        cursor.execute('''
            CREATE INDEX idx_git_commits_source_repo_date 
            ON git_commits(source_id, repository_path, commit_date)
        ''')
    
    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove Git support enhancements"""
        # Note: SQLite doesn't support DROP COLUMN, so we would need to recreate tables
        # For simplicity in this example, we'll just drop the indexes we created
        cursor.execute('DROP INDEX IF EXISTS idx_sources_last_indexed')
        cursor.execute('DROP INDEX IF EXISTS idx_files_hash')
        cursor.execute('DROP INDEX IF EXISTS idx_files_modified')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_repo_path')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_ticket_ids')
        cursor.execute('DROP INDEX IF EXISTS idx_commits_unique')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_source_repo_date')
        
        # Recreate the basic unique constraint
        cursor.execute('CREATE UNIQUE INDEX idx_commits_unique_basic ON git_commits(source_id, commit_hash)')
    
    def validate(self, cursor: sqlite3.Cursor) -> bool:
        """Validate Git support enhancements"""
        # Check that new columns exist
        cursor.execute("PRAGMA table_info(sources)")
        sources_columns = [row[1] for row in cursor.fetchall()]
        
        cursor.execute("PRAGMA table_info(git_commits)")
        commits_columns = [row[1] for row in cursor.fetchall()]
        
        cursor.execute("PRAGMA table_info(files)")
        files_columns = [row[1] for row in cursor.fetchall()]
        
        # Verify expected columns exist
        expected_sources_cols = ['commit_count', 'last_commit_processed']
        expected_commits_cols = ['repository_path']
        expected_files_cols = ['file_size']
        
        for col in expected_sources_cols:
            if col not in sources_columns:
                return False
        
        for col in expected_commits_cols:
            if col not in commits_columns:
                return False
        
        for col in expected_files_cols:
            if col not in files_columns:
                return False
        
        return True
