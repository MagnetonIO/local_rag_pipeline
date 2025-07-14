"""
Database manager for RAG Pipeline
Handles schema migrations and database operations
"""

import sqlite3
from pathlib import Path
from cli.config import config


class DatabaseManager:
    """Manages database schema and migrations"""
    
    CURRENT_SCHEMA_VERSION = config.SCHEMA_VERSION
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_current_version(self) -> int:
        """Get the current schema version from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if schema_version table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return 0  # No schema version table means version 0
            
            # Get the latest version
            cursor.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] is not None else 0
            
        except Exception as e:
            print(f"Error getting schema version: {e}")
            return 0
    
    def needs_migration(self) -> bool:
        """Check if database needs migration"""
        return self.get_current_version() < self.CURRENT_SCHEMA_VERSION
    
    def migrate(self) -> bool:
        """Migrate database to current schema version"""
        return self.migrate_to_current()
    
    def migrate_to_current(self) -> bool:
        """Migrate database to current schema version"""
        current_version = self.get_current_version()
        
        if current_version >= self.CURRENT_SCHEMA_VERSION:
            return True
        
        print(f"Migrating database from version {current_version} to {self.CURRENT_SCHEMA_VERSION}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create schema version table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''')
            
            # Create/update tables with all necessary columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_url TEXT,
                    commit_hash TEXT,
                    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    commit_count INTEGER DEFAULT 0,
                    last_commit_processed TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS git_commits (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    repository_path TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    author_email TEXT NOT NULL,
                    commit_date TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT,
                    ticket_ids TEXT,
                    files_changed TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
                )
            ''')
            
            # Add missing columns to existing tables (gracefully handle existing columns)
            columns_to_add = [
                ('sources', 'commit_count', 'INTEGER DEFAULT 0'),
                ('sources', 'last_commit_processed', 'TEXT'),
                ('sources', 'created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
                ('sources', 'updated_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
                ('files', 'file_size', 'INTEGER'),
                ('files', 'created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
                ('files', 'updated_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
                ('git_commits', 'repository_path', 'TEXT NOT NULL DEFAULT ""'),
                ('git_commits', 'processed_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
                ('git_commits', 'created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ]
            
            for table, column, definition in columns_to_add:
                try:
                    cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        print(f"Warning: Could not add {column} to {table}: {e}")
            
            # Create indexes
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type)',
                'CREATE INDEX IF NOT EXISTS idx_sources_last_indexed ON sources(last_indexed)',
                'CREATE INDEX IF NOT EXISTS idx_files_source_id ON files(source_id)',
                'CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)',
                'CREATE INDEX IF NOT EXISTS idx_files_modified ON files(last_modified)',
                'CREATE INDEX IF NOT EXISTS idx_git_commits_source_id ON git_commits(source_id)',
                'CREATE INDEX IF NOT EXISTS idx_git_commits_hash ON git_commits(commit_hash)',
                'CREATE INDEX IF NOT EXISTS idx_git_commits_repo_path ON git_commits(repository_path)',
                'CREATE INDEX IF NOT EXISTS idx_git_commits_source_repo_date ON git_commits(source_id, repository_path, commit_date)',
                'CREATE INDEX IF NOT EXISTS idx_git_commits_ticket_ids ON git_commits(ticket_ids)',
                'CREATE INDEX IF NOT EXISTS idx_git_commits_processed_at ON git_commits(processed_at)',
                'CREATE UNIQUE INDEX IF NOT EXISTS idx_files_unique ON files(source_id, file_path)',
                'CREATE UNIQUE INDEX IF NOT EXISTS idx_commits_unique ON git_commits(source_id, repository_path, commit_hash)',
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except sqlite3.OperationalError as e:
                    print(f"Warning: Could not create index: {e}")
            
            # Update schema version
            cursor.execute(
                'INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)',
                (self.CURRENT_SCHEMA_VERSION, 'Complete schema with git commits and repository tracking')
            )
            
            conn.commit()
            conn.close()
            
            print(f"✓ Database migrated to version {self.CURRENT_SCHEMA_VERSION}")
            return True
            
        except Exception as e:
            print(f"Error migrating database: {e}")
            return False