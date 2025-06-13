"""
Initial database schema migration
Creates the basic tables for the RAG pipeline
"""

import sqlite3
from .migration_manager import Migration


class InitialSchemaMigration(Migration):
    """Initial schema migration"""
    
    def __init__(self):
        super().__init__()
        self.version = "001"
        self.description = "Create initial database schema"
        self.dependencies = []
    
    def up(self, cursor: sqlite3.Cursor) -> None:
        """Create initial tables"""
        
        # Sources table
        cursor.execute('''
            CREATE TABLE sources (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_url TEXT,
                commit_hash TEXT,
                last_indexed TEXT NOT NULL,
                file_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0
            )
        ''')
        
        # Files table
        cursor.execute('''
            CREATE TABLE files (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
            )
        ''')
        
        # Git commits table (basic version)
        cursor.execute('''
            CREATE TABLE git_commits (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                author_name TEXT NOT NULL,
                author_email TEXT NOT NULL,
                commit_date TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT,
                ticket_ids TEXT,
                files_changed TEXT,
                chunk_count INTEGER DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
            )
        ''')
        
        # Create basic indexes
        cursor.execute('CREATE INDEX idx_sources_type ON sources(source_type)')
        cursor.execute('CREATE INDEX idx_files_source_id ON files(source_id)')
        cursor.execute('CREATE INDEX idx_git_commits_source_id ON git_commits(source_id)')
        cursor.execute('CREATE INDEX idx_git_commits_hash ON git_commits(commit_hash)')
        
        # Create unique constraints
        cursor.execute('CREATE UNIQUE INDEX idx_files_unique ON files(source_id, file_path)')
        cursor.execute('CREATE UNIQUE INDEX idx_commits_unique_basic ON git_commits(source_id, commit_hash)')
    
    def down(self, cursor: sqlite3.Cursor) -> None:
        """Drop all tables created in this migration"""
        cursor.execute('DROP TABLE IF EXISTS git_commits')
        cursor.execute('DROP TABLE IF EXISTS files')
        cursor.execute('DROP TABLE IF EXISTS sources')
    
    def validate(self, cursor: sqlite3.Cursor) -> bool:
        """Validate that tables were created correctly"""
        expected_tables = ['sources', 'files', 'git_commits']
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        for table in expected_tables:
            if table not in existing_tables:
                return False
        
        return True
