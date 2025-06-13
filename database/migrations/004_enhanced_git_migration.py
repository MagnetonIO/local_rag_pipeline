"""
Add enhanced Git support migration
Adds enhanced git commit tracking with detailed file changes and statistics
"""

import sqlite3
from .migration_manager import Migration, SafeColumnManager


class EnhancedGitSupportMigration(Migration):
    """Enhanced Git support migration"""
    
    def __init__(self):
        super().__init__()
        self.version = "004"
        self.description = "Add enhanced Git support with detailed file changes and statistics"
        self.dependencies = ["003"]
    
    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add enhanced Git support"""
        
        # Add enhanced columns to existing git_commits table
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'insertions', 'INTEGER', '0'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'deletions', 'INTEGER', '0'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'files_changed_count', 'INTEGER', '0'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'merge_commit', 'INTEGER', '0'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'parent_commits', 'TEXT'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'branches', 'TEXT'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'net_lines_changed', 'INTEGER', '0'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'commit_size_category', 'TEXT', "'small'"
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'primary_file_types', 'TEXT'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'git_commits', 'files_changed_json', 'TEXT'
        )
        
        # Add enhanced Git tracking to sources table
        SafeColumnManager.add_column_with_default(
            cursor, 'sources', 'enhanced_git_enabled', 'INTEGER', '0'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'sources', 'last_enhanced_commit_processed', 'TEXT'
        )
        SafeColumnManager.add_column_with_default(
            cursor, 'sources', 'total_enhanced_commits', 'INTEGER', '0'
        )
        
        # Create file history table for detailed tracking
        cursor.execute('''
            CREATE TABLE file_history (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                repository_path TEXT NOT NULL,
                file_path TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                change_type TEXT NOT NULL,
                old_file_path TEXT,
                insertions INTEGER DEFAULT 0,
                deletions INTEGER DEFAULT 0,
                commit_date TEXT NOT NULL,
                author_name TEXT NOT NULL,
                author_email TEXT NOT NULL,
                file_extension TEXT,
                created_at TEXT,
                FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
            )
        ''')
        
        # Create additional indexes for enhanced queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_insertions ON git_commits(insertions)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_deletions ON git_commits(deletions)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_size_category ON git_commits(commit_size_category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_merge ON git_commits(merge_commit)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_git_commits_file_types ON git_commits(primary_file_types)')
        
        # Indexes for file_history table
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_history_source_id ON file_history(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_history_file_path ON file_history(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_history_commit_hash ON file_history(commit_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_history_change_type ON file_history(change_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_history_author ON file_history(author_email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_history_extension ON file_history(file_extension)')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_file_history_unique ON file_history(source_id, repository_path, commit_hash, file_path)')
    
    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove enhanced Git support"""
        # Drop the file_history table
        cursor.execute('DROP TABLE IF EXISTS file_history')
        
        # Drop indexes we created
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_insertions')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_deletions')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_size_category')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_merge')
        cursor.execute('DROP INDEX IF EXISTS idx_git_commits_file_types')
        
        # Note: Can't easily remove columns from git_commits and sources tables in SQLite
        # They will remain but be unused
    
    def validate(self, cursor: sqlite3.Cursor) -> bool:
        """Validate enhanced Git support"""
        # Check that file_history table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_history'")
        if not cursor.fetchone():
            return False
        
        # Check that enhanced columns exist in git_commits
        cursor.execute("PRAGMA table_info(git_commits)")
        commits_columns = [row[1] for row in cursor.fetchall()]
        
        expected_commits_cols = [
            'insertions', 'deletions', 'files_changed_count', 'merge_commit',
            'parent_commits', 'branches', 'net_lines_changed', 
            'commit_size_category', 'primary_file_types', 'files_changed_json'
        ]
        
        for col in expected_commits_cols:
            if col not in commits_columns:
                return False
        
        # Check that enhanced columns exist in sources
        cursor.execute("PRAGMA table_info(sources)")
        sources_columns = [row[1] for row in cursor.fetchall()]
        
        expected_sources_cols = [
            'enhanced_git_enabled', 'last_enhanced_commit_processed', 'total_enhanced_commits'
        ]
        
        for col in expected_sources_cols:
            if col not in sources_columns:
                return False
        
        return True