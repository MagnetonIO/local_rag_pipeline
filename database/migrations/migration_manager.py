"""
Database Migration Manager
Handles safe database schema migrations with rollback support
"""

import sqlite3
import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Migration(ABC):
    """Base class for all migrations"""
    
    def __init__(self):
        self.version: str = "0.0.0"
        self.description: str = "Base migration"
        self.dependencies: List[str] = []
    
    @abstractmethod
    def up(self, cursor: sqlite3.Cursor) -> None:
        """Apply the migration"""
        pass
    
    @abstractmethod
    def down(self, cursor: sqlite3.Cursor) -> None:
        """Rollback the migration"""
        pass
    
    def validate(self, cursor: sqlite3.Cursor) -> bool:
        """Validate that migration was applied correctly"""
        return True


class MigrationManager:
    """Manages database migrations"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.migrations_dir = Path(__file__).parent
        self._ensure_migration_table()
    
    def _ensure_migration_table(self) -> None:
        """Create the migrations tracking table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP NOT NULL,
                    execution_time_ms INTEGER,
                    checksum TEXT
                )
            ''')
            conn.commit()
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM migrations ORDER BY applied_at")
            return [row[0] for row in cursor.fetchall()]
    
    def get_available_migrations(self) -> List[str]:
        """Get list of available migration files"""
        migrations = []
        for file_path in self.migrations_dir.glob("[0-9]*.py"):
            if file_path.name != "__init__.py":
                # Extract version from filename (e.g., "001_initial_schema.py" -> "001")
                version = file_path.stem.split('_')[0]
                migrations.append(version)
        return sorted(migrations)
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of migrations that need to be applied"""
        applied = set(self.get_applied_migrations())
        available = self.get_available_migrations()
        return [v for v in available if v not in applied]
    
    def load_migration(self, version: str) -> Migration:
        """Load a migration class from file"""
        # Find the migration file
        migration_files = list(self.migrations_dir.glob(f"{version}_*.py"))
        if not migration_files:
            raise ValueError(f"Migration {version} not found")
        
        migration_file = migration_files[0]
        module_name = f"database.migrations.{migration_file.stem}"
        
        try:
            # Import the migration module
            module = importlib.import_module(module_name)
            
            # Find the migration class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, Migration) and 
                    attr is not Migration):
                    return attr()
            
            raise ValueError(f"No migration class found in {migration_file}")
            
        except ImportError as e:
            raise ValueError(f"Could not import migration {version}: {e}")
    
    def apply_migration(self, version: str) -> bool:
        """Apply a single migration"""
        logger.info(f"Applying migration {version}")
        start_time = datetime.now()
        
        try:
            migration = self.load_migration(version)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Apply the migration
                migration.up(cursor)
                
                # Validate the migration
                if not migration.validate(cursor):
                    raise Exception("Migration validation failed")
                
                # Record the migration
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                cursor.execute('''
                    INSERT INTO migrations (version, description, applied_at, execution_time_ms)
                    VALUES (?, ?, ?, ?)
                ''', (version, migration.description, datetime.now().isoformat(), execution_time))
                
                conn.commit()
                logger.info(f"Migration {version} applied successfully in {execution_time}ms")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply migration {version}: {e}")
            return False
    
    def rollback_migration(self, version: str) -> bool:
        """Rollback a single migration"""
        logger.info(f"Rolling back migration {version}")
        
        try:
            migration = self.load_migration(version)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Rollback the migration
                migration.down(cursor)
                
                # Remove from migrations table
                cursor.execute("DELETE FROM migrations WHERE version = ?", (version,))
                
                conn.commit()
                logger.info(f"Migration {version} rolled back successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False
    
    def migrate(self, target_version: Optional[str] = None) -> bool:
        """Apply all pending migrations or migrate to specific version"""
        pending = self.get_pending_migrations()
        
        if target_version:
            # Filter to only migrations up to target version
            pending = [v for v in pending if v <= target_version]
        
        if not pending:
            logger.info("No pending migrations")
            return True
        
        logger.info(f"Applying {len(pending)} migrations: {', '.join(pending)}")
        
        success_count = 0
        for version in pending:
            if self.apply_migration(version):
                success_count += 1
            else:
                logger.error(f"Migration failed at version {version}")
                break
        
        if success_count == len(pending):
            logger.info("All migrations applied successfully")
            return True
        else:
            logger.error(f"Applied {success_count}/{len(pending)} migrations")
            return False
    
    def rollback_to(self, target_version: str) -> bool:
        """Rollback to a specific version"""
        applied = self.get_applied_migrations()
        to_rollback = [v for v in reversed(applied) if v > target_version]
        
        if not to_rollback:
            logger.info(f"Already at or below version {target_version}")
            return True
        
        logger.info(f"Rolling back {len(to_rollback)} migrations: {', '.join(to_rollback)}")
        
        success_count = 0
        for version in to_rollback:
            if self.rollback_migration(version):
                success_count += 1
            else:
                logger.error(f"Rollback failed at version {version}")
                break
        
        if success_count == len(to_rollback):
            logger.info(f"Successfully rolled back to version {target_version}")
            return True
        else:
            logger.error(f"Rolled back {success_count}/{len(to_rollback)} migrations")
            return False
    
    def get_status(self) -> Dict:
        """Get migration status"""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()
        pending = self.get_pending_migrations()
        
        return {
            "database_path": str(self.db_path),
            "current_version": applied[-1] if applied else "none",
            "latest_available": available[-1] if available else "none",
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_migrations": applied,
            "pending_migrations": pending
        }
    
    def reset_database(self) -> bool:
        """Reset database by dropping all tables and reapplying migrations"""
        logger.warning("Resetting database - all data will be lost!")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Drop all tables
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
                conn.commit()
            
            # Recreate migration table and apply all migrations
            self._ensure_migration_table()
            return self.migrate()
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False


class SafeColumnManager:
    """Utility class for safe column operations"""
    
    @staticmethod
    def add_column_with_default(cursor: sqlite3.Cursor, table: str, column: str, 
                               column_type: str, default_value: str = None) -> bool:
        """Add column with proper default value handling for SQLite"""
        try:
            # Check if column already exists
            cursor.execute(f"PRAGMA table_info({table})")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            if column in existing_columns:
                logger.info(f"Column {column} already exists in {table}")
                return True
            
            # For timestamp columns with CURRENT_TIMESTAMP, handle specially
            if 'TIMESTAMP' in column_type.upper() and 'CURRENT_TIMESTAMP' in str(default_value).upper():
                # Add as TEXT column first
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
                # Set current timestamp for existing rows
                current_time = datetime.now().isoformat()
                cursor.execute(f"UPDATE {table} SET {column} = ? WHERE {column} IS NULL", (current_time,))
            else:
                # Regular column addition
                if default_value:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type} DEFAULT {default_value}")
                else:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            
            logger.info(f"Added column {column} to {table}")
            return True
            
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to add column {column} to {table}: {e}")
            return False
    
    @staticmethod
    def recreate_table_with_data(cursor: sqlite3.Cursor, table: str, new_schema: str,
                                data_transform: Callable = None) -> bool:
        """Recreate table with new schema while preserving data"""
        try:
            # Get existing data
            cursor.execute(f"SELECT * FROM {table}")
            existing_data = cursor.fetchall()
            
            # Get old column info
            cursor.execute(f"PRAGMA table_info({table})")
            old_columns = [row[1] for row in cursor.fetchall()]
            
            # Create new table
            temp_table = f"{table}_new_{int(datetime.now().timestamp())}"
            cursor.execute(f"CREATE TABLE {temp_table} ({new_schema})")
            
            # Get new column info
            cursor.execute(f"PRAGMA table_info({temp_table})")
            new_columns = [row[1] for row in cursor.fetchall()]
            
            # Migrate data
            if existing_data:
                common_columns = [col for col in old_columns if col in new_columns]
                
                if common_columns:
                    # Prepare insert statement
                    placeholders = ', '.join(['?' for _ in new_columns])
                    insert_sql = f"INSERT INTO {temp_table} ({', '.join(new_columns)}) VALUES ({placeholders})"
                    
                    for row in existing_data:
                        new_row = [None] * len(new_columns)
                        
                        # Map existing data
                        for i, old_col in enumerate(old_columns):
                            if old_col in new_columns:
                                new_idx = new_columns.index(old_col)
                                new_row[new_idx] = row[i]
                        
                        # Apply data transformation if provided
                        if data_transform:
                            new_row = data_transform(new_row, old_columns, new_columns, row)
                        
                        # Set defaults for timestamp columns
                        current_time = datetime.now().isoformat()
                        for j, col in enumerate(new_columns):
                            if new_row[j] is None and any(keyword in col.lower() for keyword in ['created_at', 'updated_at', 'processed_at']):
                                new_row[j] = current_time
                        
                        cursor.execute(insert_sql, new_row)
            
            # Replace old table
            cursor.execute(f"DROP TABLE {table}")
            cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table}")
            
            logger.info(f"Successfully recreated table {table}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate table {table}: {e}")
            return False
