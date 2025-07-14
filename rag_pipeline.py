#!/usr/bin/env python3
"""
Local RAG Pipeline with Git Commit Support and LaTeX-Aware Processing
A comprehensive RAG pipeline for ingesting and querying documents and Git commits locally
"""

import argparse
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

# Core dependencies
import chromadb
from chromadb.config import Settings
import tiktoken
from sentence_transformers import SentenceTransformer

# Optional dependencies
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Run: pip install anthropic")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Run: pip install openai")

# Configuration from environment variables
class Config:
    # Data storage
    DATA_DIR = os.getenv('RAG_DATA_DIR', './rag_data')
    DATABASE_NAME = os.getenv('RAG_DATABASE_NAME', 'metadata.db')
    VECTOR_STORE_DIR = os.getenv('RAG_VECTOR_STORE_DIR', 'chroma_db')
    REPOS_DIR = os.getenv('RAG_REPOS_DIR', 'repos')
    
    # AI Models
    EMBEDDING_MODEL = os.getenv('RAG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    CLAUDE_MODEL = os.getenv('RAG_CLAUDE_MODEL', 'claude-3-sonnet-20240229')
    OPENAI_MODEL = os.getenv('RAG_OPENAI_MODEL', 'gpt-3.5-turbo')
    DEFAULT_AI_MODEL = os.getenv('RAG_DEFAULT_AI_MODEL', 'claude')
    AI_MAX_TOKENS = int(os.getenv('RAG_AI_MAX_TOKENS', '1000'))
    
    # Document processing
    CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '200'))
    LATEX_CHUNK_SIZE = int(os.getenv('RAG_LATEX_CHUNK_SIZE', '2000'))
    LATEX_CHUNK_OVERLAP = int(os.getenv('RAG_LATEX_CHUNK_OVERLAP', '300'))
    MAX_FILE_SIZE = int(os.getenv('RAG_MAX_FILE_SIZE', '10485760'))  # 10MB
    
    # Git processing
    MAX_COMMITS = int(os.getenv('RAG_MAX_COMMITS', '1000'))
    GIT_LOG_TIMEOUT = int(os.getenv('RAG_GIT_LOG_TIMEOUT', '60'))
    GIT_DIFF_TIMEOUT = int(os.getenv('RAG_GIT_DIFF_TIMEOUT', '30'))
    GIT_CLONE_TIMEOUT = int(os.getenv('RAG_GIT_CLONE_TIMEOUT', '300'))
    GIT_VERIFY_TIMEOUT = int(os.getenv('RAG_GIT_VERIFY_TIMEOUT', '10'))
    
    # Search configuration
    DEFAULT_SEARCH_LIMIT = int(os.getenv('RAG_DEFAULT_SEARCH_LIMIT', '5'))
    DEFAULT_TICKET_SEARCH_LIMIT = int(os.getenv('RAG_DEFAULT_TICKET_SEARCH_LIMIT', '10'))
    MAX_SEARCH_LIMIT = int(os.getenv('RAG_MAX_SEARCH_LIMIT', '20'))
    
    # File processing
    @staticmethod
    def get_supported_extensions():
        extensions_str = os.getenv('RAG_SUPPORTED_EXTENSIONS', 
            'py,js,ts,jsx,tsx,java,cpp,c,h,cs,php,rb,go,rs,swift,kt,scala,md,txt,rst,org,tex,json,yaml,yml,xml,html,css,sql,sh,bash,zsh,dockerfile,gitignore,env,toml,ini,cfg')
        return {f'.{ext.strip()}' for ext in extensions_str.split(',')}
    
    @staticmethod
    def get_ignored_directories():
        dirs_str = os.getenv('RAG_IGNORE_DIRECTORIES',
            'node_modules,__pycache__,venv,.venv,env,.env,target,build,.gradle,.m2,bin,obj,vendor,.idea,.vscode,.vs,.DS_Store,logs,tmp,temp,dist,out,.pytest_cache,.git,.svn')
        return {dir.strip() for dir in dirs_str.split(',')}
    
    # Database
    SCHEMA_VERSION = int(os.getenv('RAG_SCHEMA_VERSION', '2'))
    
    # Performance
    PROCESSING_THREADS = int(os.getenv('RAG_PROCESSING_THREADS', '4'))
    VECTOR_BATCH_SIZE = int(os.getenv('RAG_VECTOR_BATCH_SIZE', '100'))
    
    # Security
    ALLOW_EXECUTABLE_FILES = os.getenv('RAG_ALLOW_EXECUTABLE_FILES', 'false').lower() == 'true'
    MAX_PATH_DEPTH = int(os.getenv('RAG_MAX_PATH_DEPTH', '10'))
    
    # Development
    DEV_MODE = os.getenv('RAG_DEV_MODE', 'false').lower() == 'true'
    SKIP_FILE_VALIDATION = os.getenv('RAG_SKIP_FILE_VALIDATION', 'false').lower() == 'true'

def timestamp() -> str:
    """Return current timestamp"""
    return datetime.now().isoformat()

class DatabaseManager:
    """Manages database schema and migrations"""
    
    CURRENT_SCHEMA_VERSION = Config.SCHEMA_VERSION
    
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

class GitCommitProcessor:
    """Handles Git commit parsing and processing"""
    
    def __init__(self):
        self.ticket_pattern = re.compile(r'\b([A-Z]+-\d+)\b', re.IGNORECASE)
    
    def extract_ticket_ids(self, message: str) -> List[str]:
        """Extract ticket IDs from commit messages (e.g., GET-1903, JIRA-123)"""
        matches = self.ticket_pattern.findall(message.upper())
        return list(set(matches))  # Remove duplicates
    
    def get_git_commits(self, repo_path: Path, max_commits: int = None, since_commit: Optional[str] = None) -> List[Dict]:
        """Extract commit information from a Git repository"""
        commits = []
        max_commits = max_commits or Config.MAX_COMMITS
        
        try:
            # Build git log command
            cmd = [
                "git", "log", 
                f"--max-count={max_commits}",
                "--pretty=format:%H|%an|%ae|%ad|%s|%b",
                "--date=iso",
                "--name-status"
            ]
            
            # If we have a since_commit, only get commits after it
            if since_commit:
                cmd.append(f"{since_commit}..HEAD")
            
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=Config.GIT_LOG_TIMEOUT
            )
            
            if result.returncode != 0:
                print(f"Error getting git log: {result.stderr}")
                return commits
                
            # Parse the output
            current_commit = None
            files_section = False
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                
                if not line:
                    if current_commit and files_section:
                        # End of current commit
                        commits.append(current_commit)
                        current_commit = None
                        files_section = False
                    continue
                
                if '|' in line and not files_section:
                    # This is a commit header line
                    if current_commit:
                        commits.append(current_commit)
                    
                    parts = line.split('|', 5)
                    if len(parts) >= 5:
                        hash_val, author_name, author_email, date_str, subject = parts[:5]
                        body = parts[5] if len(parts) > 5 else ""
                        
                        full_message = f"{subject}\n{body}".strip()
                        ticket_ids = self.extract_ticket_ids(full_message)
                        
                        current_commit = {
                            'hash': hash_val,
                            'author_name': author_name,
                            'author_email': author_email,
                            'date': date_str,
                            'subject': subject,
                            'body': body,
                            'full_message': full_message,
                            'ticket_ids': ticket_ids,
                            'files_changed': []
                        }
                        files_section = True
                        
                elif files_section and current_commit:
                    # This is a file change line (M, A, D, etc.)
                    if line and not line.startswith('commit'):
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            change_type, file_path = parts
                            current_commit['files_changed'].append({
                                'type': change_type,
                                'path': file_path
                            })
            
            # Don't forget the last commit
            if current_commit:
                commits.append(current_commit)
                
        except subprocess.TimeoutExpired:
            print("Timeout getting git commits")
        except Exception as e:
            print(f"Error processing git commits: {e}")
            
        return commits
    
    def get_commit_diff(self, repo_path: Path, commit_hash: str) -> str:
        """Get the diff for a specific commit"""
        try:
            result = subprocess.run(
                ["git", "show", "--pretty=format:", "--name-status", commit_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=Config.GIT_DIFF_TIMEOUT
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return ""
                
        except Exception as e:
            print(f"Error getting commit diff for {commit_hash}: {e}")
            return ""

class LatexAwareDocumentProcessor:
    """Enhanced document processor with LaTeX-specific chunking strategies"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.LATEX_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.LATEX_CHUNK_OVERLAP
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # LaTeX structure patterns
        self.section_patterns = [
            r'\\section\*?\{([^}]+)\}',
            r'\\subsection\*?\{([^}]+)\}',
            r'\\subsubsection\*?\{([^}]+)\}',
            r'\\chapter\*?\{([^}]+)\}',
            r'\\part\*?\{([^}]+)\}'
        ]
        
        # Math environment patterns
        self.math_env_patterns = [
            r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}',
            r'\\begin\{align\*?\}.*?\\end\{align\*?\}',
            r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}',
            r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}',
            r'\\\[.*?\\\]',
            r'\$\$.*?\$\$'
        ]
        
        # Theorem-like environments
        self.theorem_patterns = [
            r'\\begin\{theorem\}.*?\\end\{theorem\}',
            r'\\begin\{lemma\}.*?\\end\{lemma\}',
            r'\\begin\{proposition\}.*?\\end\{proposition\}',
            r'\\begin\{corollary\}.*?\\end\{corollary\}',
            r'\\begin\{definition\}.*?\\end\{definition\}',
            r'\\begin\{proof\}.*?\\end\{proof\}',
            r'\\begin\{example\}.*?\\end\{example\}',
            r'\\begin\{remark\}.*?\\end\{remark\}'
        ]

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Enhanced chunking with LaTeX awareness"""
        if not text.strip():
            return []
            
        file_extension = metadata.get('file_type', '').lower()
        
        if file_extension == '.tex':
            return self._chunk_latex(text, metadata)
        else:
            return self._chunk_generic(text, metadata)
    
    def _chunk_latex(self, text: str, metadata: Dict) -> List[Dict]:
        """LaTeX-specific chunking strategy"""
        chunks = []
        
        # First, identify major structural elements
        structural_elements = self._identify_latex_structure(text)
        
        # Process each structural element
        for element in structural_elements:
            element_chunks = self._process_latex_element(element, metadata)
            chunks.extend(element_chunks)
        
        return chunks
    
    def _identify_latex_structure(self, text: str) -> List[Dict]:
        """Identify major structural components in LaTeX document"""
        elements = []
        lines = text.split('\n')
        current_element = {'type': 'preamble', 'content': [], 'metadata': {}}
        in_document = False
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Document boundaries
            if '\\begin{document}' in line:
                if current_element['content']:
                    elements.append(current_element)
                current_element = {'type': 'document_start', 'content': [line], 'metadata': {}}
                in_document = True
                i += 1
                continue
            elif '\\end{document}' in line:
                if current_element['content']:
                    elements.append(current_element)
                elements.append({'type': 'document_end', 'content': [line], 'metadata': {}})
                break
            
            if not in_document:
                current_element['content'].append(line)
                i += 1
                continue
            
            # Check for section headers
            section_match = self._match_section(line)
            if section_match:
                if current_element['content']:
                    elements.append(current_element)
                current_element = {
                    'type': 'section',
                    'content': [line],
                    'metadata': {
                        'section_type': section_match['type'],
                        'section_title': section_match['title'],
                        'level': section_match['level']
                    }
                }
                i += 1
                continue
            
            # Check for theorem-like environments
            theorem_block = self._extract_environment_block(lines, i, self.theorem_patterns)
            if theorem_block:
                if current_element['content']:
                    elements.append(current_element)
                elements.append({
                    'type': 'theorem',
                    'content': theorem_block['content'],
                    'metadata': {'env_type': theorem_block['env_type']}
                })
                current_element = {'type': 'content', 'content': [], 'metadata': {}}
                i = theorem_block['end_index']
                continue
            
            # Check for math environments
            math_block = self._extract_environment_block(lines, i, self.math_env_patterns)
            if math_block:
                if current_element['content']:
                    elements.append(current_element)
                elements.append({
                    'type': 'math',
                    'content': math_block['content'],
                    'metadata': {'env_type': math_block['env_type']}
                })
                current_element = {'type': 'content', 'content': [], 'metadata': {}}
                i = math_block['end_index']
                continue
            
            # Regular content
            current_element['content'].append(line)
            i += 1
        
        if current_element['content']:
            elements.append(current_element)
        
        return elements
    
    def _match_section(self, line: str) -> Optional[Dict]:
        """Match section headers and extract metadata"""
        section_levels = {
            'part': 0,
            'chapter': 1,
            'section': 2,
            'subsection': 3,
            'subsubsection': 4
        }
        
        for pattern in self.section_patterns:
            match = re.search(pattern, line)
            if match:
                section_type = re.search(r'\\(\w+)', line).group(1)
                return {
                    'type': section_type,
                    'title': match.group(1),
                    'level': section_levels.get(section_type, 5)
                }
        return None
    
    def _extract_environment_block(self, lines: List[str], start_idx: int, patterns: List[str]) -> Optional[Dict]:
        """Extract complete environment blocks (theorem, equation, etc.)"""
        # Check if current line starts an environment
        current_line = lines[start_idx]
        
        for pattern in patterns:
            # Handle single-line patterns (like \[...\] or $$...$$)
            single_line_match = re.search(pattern, current_line, re.DOTALL)
            if single_line_match and not ('\\begin{' in pattern):
                return {
                    'content': [current_line],
                    'end_index': start_idx + 1,
                    'env_type': 'inline_math'
                }
            
            # Handle multi-line environments
            begin_match = re.search(r'\\begin\{(\w+\*?)\}', current_line)
            if begin_match:
                env_name = begin_match.group(1)
                end_pattern = f'\\\\end{{{env_name}}}'
                
                # Find the end of this environment
                content = [current_line]
                for i in range(start_idx + 1, len(lines)):
                    content.append(lines[i])
                    if re.search(end_pattern, lines[i]):
                        return {
                            'content': content,
                            'end_index': i + 1,
                            'env_type': env_name
                        }
        
        return None
    
    def _process_latex_element(self, element: Dict, base_metadata: Dict) -> List[Dict]:
        """Process individual LaTeX elements into chunks"""
        content = '\n'.join(element['content'])
        
        # Combine base metadata with element-specific metadata
        chunk_metadata = {**base_metadata, **element.get('metadata', {})}
        chunk_metadata['element_type'] = element['type']
        
        # For small elements, keep them whole
        if len(content) <= self.chunk_size:
            return [{
                'content': content,
                'metadata': chunk_metadata
            }]
        
        # For large elements, split more carefully
        if element['type'] in ['theorem', 'math']:
            # Keep mathematical content together as much as possible
            return self._split_mathematical_content(content, chunk_metadata)
        elif element['type'] == 'section':
            # Split section content at paragraph boundaries
            return self._split_section_content(content, chunk_metadata)
        else:
            # Default splitting for other content
            return self._split_generic_content(content, chunk_metadata)
    
    def _split_mathematical_content(self, content: str, metadata: Dict) -> List[Dict]:
        """Split mathematical content while preserving coherence"""
        chunks = []
        
        # Try to split at logical boundaries within math content
        # Look for \\ (line breaks in math), blank lines, or comment lines
        split_points = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if (line.strip() == '' or 
                line.strip().startswith('%') or 
                '\\\\' in line):
                split_points.append(i)
        
        if not split_points:
            # If no good split points, just return as single chunk
            return [{'content': content, 'metadata': metadata}]
        
        # Create chunks at split points
        current_chunk = []
        for i, line in enumerate(lines):
            current_chunk.append(line)
            if i in split_points and len('\n'.join(current_chunk)) >= self.chunk_size // 2:
                chunks.append({
                    'content': '\n'.join(current_chunk),
                    'metadata': metadata
                })
                current_chunk = []
        
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'metadata': metadata
            })
        
        return chunks
    
    def _split_section_content(self, content: str, metadata: Dict) -> List[Dict]:
        """Split section content at natural boundaries"""
        # Split at paragraph boundaries (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'content': '\n\n'.join(current_chunk),
                    'metadata': metadata
                })
                current_chunk = [paragraph]
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size
        
        if current_chunk:
            chunks.append({
                'content': '\n\n'.join(current_chunk),
                'metadata': metadata
            })
        
        return chunks
    
    def _split_generic_content(self, content: str, metadata: Dict) -> List[Dict]:
        """Generic content splitting with overlap"""
        chunks = []
        words = content.split()
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'metadata': metadata
                })
                
                # Create overlap
                overlap_words = []
                overlap_size = 0
                for w in reversed(current_chunk):
                    if overlap_size + len(w) <= self.chunk_overlap:
                        overlap_words.insert(0, w)
                        overlap_size += len(w) + 1
                    else:
                        break
                
                current_chunk = overlap_words + [word]
                current_size = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'metadata': metadata
            })
        
        return chunks
    
    def _chunk_generic(self, text: str, metadata: Dict) -> List[Dict]:
        """Generic chunking for non-LaTeX files"""
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(self.tokenizer.encode(test_chunk)) > self.chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': metadata
                })
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': metadata
            })
        
        return chunks

class DocumentProcessor:
    """Handles document parsing and chunking with LaTeX support"""

    @property
    def SUPPORTED_EXTENSIONS(self):
        return Config.get_supported_extensions()

    @property
    def IGNORE_DIRECTORIES(self):
        return Config.get_ignored_directories()

    # File patterns to ignore
    IGNORE_FILE_PATTERNS = {
        # Compiled files
        '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.dylib', '*.exe',
        '*.class', '*.jar', '*.war', '*.ear',
        '*.o', '*.obj', '*.lib', '*.a',

        # Package files
        '*.zip', '*.tar', '*.tar.gz', '*.tgz', '*.rar', '*.7z',
        '*.deb', '*.rpm', '*.msi',

        # Lock files
        'package-lock.json', 'yarn.lock', 'Pipfile.lock', 'poetry.lock',
        'Gemfile.lock', 'composer.lock', 'Cargo.lock',

        # Log files
        '*.log', '*.out', '*.err',

        # Database files
        '*.db', '*.sqlite', '*.sqlite3',

        # Media files (usually too large and not useful for code analysis)
        '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.svg', '*.ico',
        '*.mp3', '*.mp4', '*.avi', '*.mov', '*.wmv', '*.pdf',

        # IDE files
        '*.swp', '*.swo', '*~', '.DS_Store', 'Thumbs.db',

        # Environment files (may contain secrets)
        '.env', '.env.local', '.env.production', '.env.development'
    }

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize the LaTeX-aware processor
        self.latex_processor = LatexAwareDocumentProcessor(self.chunk_size, self.chunk_overlap)

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        # Check file extension
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False

        # Check if file is in an ignored directory
        path_parts = file_path.parts
        for part in path_parts:
            if part.lower() in self.IGNORE_DIRECTORIES:
                return False

        # Check file name patterns
        import fnmatch
        file_name = file_path.name.lower()
        for pattern in self.IGNORE_FILE_PATTERNS:
            if fnmatch.fnmatch(file_name, pattern.lower()):
                return False

        # Check file size (skip very large files)
        try:
            file_size = file_path.stat().st_size
            # Skip files larger than configured max size
            if file_size > Config.MAX_FILE_SIZE:
                return False
            # Skip empty files
            if file_size == 0:
                return False
        except OSError:
            return False

        return True

    def extract_text(self, file_path: Path) -> str:
        """Extract text content from file with robust error handling"""
        try:
            # First, check if file exists and is readable
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return ""

            # Check file size
            file_size = file_path.stat().st_size
            print(f"Processing {file_path} (size: {file_size} bytes)")

            # Try multiple encoding strategies
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()

                    # Verify we got reasonable content
                    if content and len(content.strip()) > 0:
                        print(f"Successfully read {file_path} with encoding {encoding}")
                        return content

                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with encoding {encoding} for {file_path}: {e}")
                    continue

            # If all encodings fail, try binary mode and convert
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()

                # Try to decode as utf-8 with error handling
                content = raw_data.decode('utf-8', errors='replace')
                print(f"Read {file_path} in binary mode with error replacement")
                return content

            except Exception as e:
                print(f"Failed to read {file_path} in binary mode: {e}")
                return ""

        except PermissionError:
            print(f"Permission denied: {file_path}")
            return ""
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return ""
        except OSError as e:
            print(f"OS error reading {file_path}: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return ""

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into chunks with metadata using LaTeX-aware processing"""
        if not text.strip():
            return []

        # Use the LaTeX-aware processor which handles both LaTeX and generic files
        return self.latex_processor.chunk_text(text, metadata)

class GitManager:
    """Handles Git operations"""

    @staticmethod
    def clone_repo(repo_url: str, target_dir: Path) -> bool:
        """Clone a git repository"""
        try:
            result = subprocess.run(
                ["git", "clone", repo_url, str(target_dir)],
                capture_output=True,
                text=True,
                timeout=Config.GIT_CLONE_TIMEOUT
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"Timeout cloning repository: {repo_url}")
            return False
        except Exception as e:
            print(f"Error cloning repository: {e}")
            return False

    @staticmethod
    def get_repo_info(repo_dir: Path) -> Dict[str, str]:
        """Get repository information"""
        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )

            commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"

            return {
                'commit_hash': commit_hash,
                'last_updated': timestamp()
            }
        except Exception:
            return {
                'commit_hash': 'unknown',
                'last_updated': timestamp()
            }

    @staticmethod
    def is_git_repo(path: Path) -> bool:
        """Check if path is a git repository"""
        # If the path itself is a .git directory
        if path.name == '.git':
            return True
        
        # Check if path contains a .git directory
        git_dir = path / '.git'
        return git_dir.exists() and (git_dir.is_dir() or git_dir.is_file())

class RAGPipeline:
    """Main RAG pipeline class with LaTeX support"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or Config.DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize components with LaTeX support
        self.processor = DocumentProcessor()
        self.git_processor = GitCommitProcessor()
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.data_dir / Config.VECTOR_STORE_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize metadata database with migration
        self.db_path = self.data_dir / Config.DATABASE_NAME
        self.db_manager = DatabaseManager(str(self.db_path))
        
        # Migrate database to current schema
        if not self.db_manager.migrate_to_current():
            raise Exception("Failed to migrate database to current schema")

        # Initialize API clients
        self.anthropic_client = None
        self.openai_client = None
        self.setup_api_clients()

    def setup_api_clients(self):
        """Setup API clients for Claude and OpenAI"""
        # Anthropic/Claude
        if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

        # OpenAI
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception:
            return ""
        return hash_md5.hexdigest()

    def _process_git_commits(self, repo_path: Path, source_id: str, incremental: bool = True) -> int:
        """Process Git commits and add them to the index"""
        # Determine the actual repository root
        if repo_path.name == '.git':
            # If we're given the .git directory, use its parent
            actual_repo_path = repo_path.parent
        else:
            # Otherwise, use the given path
            actual_repo_path = repo_path
        
        if not GitManager.is_git_repo(actual_repo_path) and not GitManager.is_git_repo(repo_path):
            print(f"Not a git repository: {repo_path}")
            return 0

        print(f"Processing Git commits from: {actual_repo_path}")
        
        # Get the last processed commit for this specific repository if doing incremental processing
        last_commit_hash = None
        if incremental:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find the most recent commit hash for this specific repository
            cursor.execute('''
                SELECT commit_hash FROM git_commits 
                WHERE source_id = ? AND repository_path = ?
                ORDER BY commit_date DESC 
                LIMIT 1
            ''', (source_id, str(actual_repo_path)))
            
            result = cursor.fetchone()
            if result:
                last_commit_hash = result[0]
                print(f"Last processed commit: {last_commit_hash[:8]}...")
                
                # Verify the commit exists in this repository
                try:
                    verify_result = subprocess.run(
                        ["git", "cat-file", "-e", last_commit_hash],
                        cwd=actual_repo_path,
                        capture_output=True,
                        timeout=Config.GIT_VERIFY_TIMEOUT
                    )
                    if verify_result.returncode != 0:
                        print(f"Warning: Last processed commit {last_commit_hash[:8]} not found in repository, doing full sync")
                        last_commit_hash = None
                except Exception as e:
                    print(f"Warning: Could not verify last commit, doing full sync: {e}")
                    last_commit_hash = None
            
            conn.close()
        
        # Get commits (only new ones if incremental and we have a valid last commit)
        commits = self.git_processor.get_git_commits(actual_repo_path, since_commit=last_commit_hash)
        
        if not commits:
            if incremental and last_commit_hash:
                print("No new commits found since last run")
            else:
                print("No commits found")
            return 0

        print(f"Found {len(commits)} new commits to process")
        
        total_commit_chunks = 0
        processed_commits = 0
        
        for commit in commits:
            try:
                # Check if this commit is already processed (extra safety check)
                if incremental:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM git_commits WHERE commit_hash = ? AND source_id = ? AND repository_path = ?", 
                                 (commit['hash'], source_id, str(actual_repo_path)))
                    if cursor.fetchone():
                        print(f"Commit {commit['hash'][:8]} already processed, skipping")
                        conn.close()
                        continue
                    conn.close()
                
                # Create commit document content
                files_changed_text = ""
                if commit['files_changed']:
                    files_list = [f"{f['type']} {f['path']}" for f in commit['files_changed']]
                    files_changed_text = f"Files changed:\n" + "\n".join(files_list)

                ticket_text = ""
                if commit['ticket_ids']:
                    ticket_text = f"Related tickets: {', '.join(commit['ticket_ids'])}"

                commit_content = f"""Git Commit: {commit['hash'][:8]}

Author: {commit['author_name']} <{commit['author_email']}>
Date: {commit['date']}
Repository: {actual_repo_path.name}
{ticket_text}

Subject: {commit['subject']}

Message:
{commit['full_message']}

{files_changed_text}"""

                # Create metadata for the commit
                metadata = {
                    'source_id': source_id,
                    'content_type': 'git_commit',
                    'commit_hash': commit['hash'],
                    'author_name': commit['author_name'],
                    'author_email': commit['author_email'],
                    'commit_date': commit['date'],
                    'subject': commit['subject'],
                    'repository_name': actual_repo_path.name,
                    'repository_path': str(actual_repo_path),
                    'ticket_ids': ','.join(commit['ticket_ids']) if commit['ticket_ids'] else '',
                    'files_changed_count': len(commit['files_changed'])
                }

                # Create chunks for the commit
                chunks = self.processor.chunk_text(commit_content, metadata)
                
                if chunks:
                    # Generate embeddings and store
                    chunk_ids = []
                    chunk_contents = []
                    chunk_metadatas = []

                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{source_id}_commit_{commit['hash'][:8]}_chunk_{i}"
                        chunk_ids.append(chunk_id)
                        chunk_contents.append(chunk['content'])
                        chunk_metadatas.append(chunk['metadata'])

                    # Store in ChromaDB
                    self.collection.upsert(
                        ids=chunk_ids,
                        documents=chunk_contents,
                        metadatas=chunk_metadatas
                    )

                    total_commit_chunks += len(chunks)

                    # Store commit metadata in database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO git_commits
                        (id, source_id, repository_path, commit_hash, author_name, author_email, 
                         commit_date, subject, body, ticket_ids, files_changed, chunk_count, processed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        f"{source_id}_commit_{commit['hash']}",
                        source_id,
                        str(actual_repo_path),
                        commit['hash'],
                        commit['author_name'],
                        commit['author_email'],
                        commit['date'],
                        commit['subject'],
                        commit['body'],
                        ','.join(commit['ticket_ids']),
                        str(commit['files_changed']),
                        len(chunks)
                    ))
                    conn.commit()
                    conn.close()

                    processed_commits += 1
                    print(f"Processed commit {commit['hash'][:8]}: {len(chunks)} chunks")

            except Exception as e:
                print(f"Error processing commit {commit['hash']}: {e}")
                continue

        print(f"Processed {processed_commits} new commits, created {total_commit_chunks} chunks")
        return processed_commits

    def ingest_single_file(self, file_path: str, source_name: Optional[str] = None) -> str:
        """Ingest a single file with LaTeX support"""
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.processor.should_process_file(file_path):
            raise ValueError(f"File type not supported: {file_path}")

        source_id = source_name or f"file_{file_path.stem}_{int(datetime.now().timestamp())}"

        print(f"Ingesting single file: {file_path}")
        
        # Special handling for LaTeX files
        if file_path.suffix.lower() == '.tex':
            print("Detected LaTeX file - using specialized LaTeX processing")

        try:
            file_id = self._process_file(file_path, source_id, str(file_path.parent))
            if file_id:
                # Get chunk count for this file
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_count FROM files WHERE id = ?", (file_id,))
                result = cursor.fetchone()
                chunk_count = result[0] if result else 0
                conn.close()

                # Update source metadata
                self._update_source_metadata(source_id, "file", str(file_path),
                                           None, None, 1, chunk_count, 0)

                print(f"Successfully ingested file with {chunk_count} chunks")
                return source_id
            else:
                raise Exception("Failed to process file")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            raise

    def ingest_directory(self, dir_path: str, source_name: Optional[str] = None) -> str:
        """Ingest a local directory with LaTeX support"""
        dir_path = Path(dir_path).resolve()
        source_id = source_name or f"dir_{dir_path.name}_{int(datetime.now().timestamp())}"

        print(f"Ingesting directory: {dir_path}")

        # Find all Git repositories in the directory tree
        git_repos = []
        for item in dir_path.rglob('.git'):
            if item.is_dir():
                repo_root = item.parent
                git_repos.append(repo_root)
                print(f"Found Git repository: {repo_root}")

        # If the root directory itself is a Git repository, add it
        if GitManager.is_git_repo(dir_path) and dir_path not in git_repos:
            git_repos.append(dir_path)
            print(f"Root directory is a Git repository: {dir_path}")

        # Process commits from all discovered Git repositories
        total_commit_count = 0
        for repo_path in git_repos:
            print(f"Processing commits from: {repo_path}")
            try:
                commit_count = self._process_git_commits(repo_path, source_id, incremental=True)
                total_commit_count += commit_count
                if commit_count > 0:
                    print(f"Processed {commit_count} new commits from {repo_path}")
                else:
                    print(f"No new commits found in {repo_path}")
            except Exception as e:
                print(f"Error processing commits from {repo_path}: {e}")

        # Process files
        file_count = 0
        total_chunks = 0
        failed_files = []
        skipped_dirs = set()
        skipped_files = set()
        latex_files = 0

        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                # Check if we should skip this file due to directory rules
                should_skip_dir = False
                for part in file_path.relative_to(dir_path).parts[:-1]:  # Exclude filename
                    if part.lower() in self.processor.IGNORE_DIRECTORIES:
                        skipped_dirs.add(part)
                        should_skip_dir = True
                        break

                if should_skip_dir:
                    continue

                # Check if file should be processed
                if self.processor.should_process_file(file_path):
                    try:
                        # Count LaTeX files
                        if file_path.suffix.lower() == '.tex':
                            latex_files += 1
                            print(f"Processing LaTeX file: {file_path.relative_to(dir_path)}")
                            
                        file_id = self._process_file(file_path, source_id, str(dir_path))
                        if file_id:
                            file_count += 1
                            # Get chunk count for this file
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT chunk_count FROM files WHERE id = ?", (file_id,))
                            result = cursor.fetchone()
                            if result:
                                total_chunks += result[0]
                            conn.close()
                        else:
                            failed_files.append(str(file_path))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        failed_files.append(str(file_path))
                else:
                    # Track why files were skipped for reporting
                    if not file_path.suffix.lower() in self.processor.SUPPORTED_EXTENSIONS:
                        continue  # Don't report unsupported extensions
                    skipped_files.add(file_path.name)

        # Update source metadata
        self._update_source_metadata(source_id, "directory", str(dir_path),
                                    None, None, file_count, total_chunks, total_commit_count)

        # Print summary
        print(f"Ingested {file_count} files, {total_chunks} chunks, {total_commit_count} commits from {len(git_repos)} repositories")
        
        if latex_files > 0:
            print(f"Processed {latex_files} LaTeX files with specialized LaTeX chunking")

        if git_repos:
            print(f"Git repositories processed:")
            for repo in git_repos:
                print(f"  - {repo}")

        if skipped_dirs:
            print(f"Skipped directories: {', '.join(sorted(skipped_dirs))}")

        if len(skipped_files) > 0 and len(skipped_files) <= 20:
            print(f"Skipped files: {', '.join(sorted(skipped_files))}")
        elif len(skipped_files) > 20:
            print(f"Skipped {len(skipped_files)} files (log files, binaries, etc.)")

        if failed_files:
            print(f"Failed to process {len(failed_files)} files:")
            for failed_file in failed_files[:10]:  # Show first 10
                print(f"  - {failed_file}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")

        return source_id

    def ingest_git_repo(self, repo_url: str, source_name: Optional[str] = None) -> str:
        """Ingest a git repository with LaTeX support"""
        source_id = source_name or f"repo_{repo_url.split('/')[-1].replace('.git', '')}_{int(datetime.now().timestamp())}"
        repo_dir = self.data_dir / Config.REPOS_DIR / source_id

        print(f"Cloning repository: {repo_url}")

        # Clone repository
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.parent.mkdir(exist_ok=True)

        if not GitManager.clone_repo(repo_url, repo_dir):
            raise Exception(f"Failed to clone repository: {repo_url}")

        # Get repo info
        repo_info = GitManager.get_repo_info(repo_dir)

        # Process Git commits first
        commit_count = self._process_git_commits(repo_dir, source_id, incremental=False)  # Full processing for new repos

        # Process files
        file_count = 0
        total_chunks = 0
        skipped_dirs = set()
        latex_files = 0

        for file_path in repo_dir.rglob("*"):
            if file_path.is_file():
                # Check if we should skip this file due to directory rules
                should_skip_dir = False
                for part in file_path.relative_to(repo_dir).parts[:-1]:  # Exclude filename
                    if part.lower() in self.processor.IGNORE_DIRECTORIES:
                        skipped_dirs.add(part)
                        should_skip_dir = True
                        break

                if should_skip_dir:
                    continue

                if self.processor.should_process_file(file_path):
                    try:
                        # Count LaTeX files
                        if file_path.suffix.lower() == '.tex':
                            latex_files += 1
                            print(f"Processing LaTeX file: {file_path.relative_to(repo_dir)}")
                            
                        file_id = self._process_file(file_path, source_id, str(repo_dir))
                        if file_id:
                            file_count += 1
                            # Get chunk count for this file
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT chunk_count FROM files WHERE id = ?", (file_id,))
                            result = cursor.fetchone()
                            if result:
                                total_chunks += result[0]
                            conn.close()
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        # Update source metadata
        self._update_source_metadata(source_id, "git", str(repo_dir), repo_url,
                                    repo_info['commit_hash'], file_count, total_chunks, commit_count)

        print(f"Ingested {file_count} files, {total_chunks} chunks, {commit_count} commits")
        
        if latex_files > 0:
            print(f"Processed {latex_files} LaTeX files with specialized LaTeX chunking")
            
        if skipped_dirs:
            print(f"Skipped directories: {', '.join(sorted(skipped_dirs))}")

        return source_id

    def _process_file(self, file_path: Path, source_id: str, base_path: str) -> Optional[str]:
        """Process a single file with LaTeX support"""
        relative_path = str(file_path.relative_to(base_path))
        file_id = f"{source_id}_{hashlib.md5(relative_path.encode()).hexdigest()}"

        # Calculate file hash
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            print(f"Could not calculate hash for {file_path}")
            return None

        # Check if file already processed and unchanged
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_hash FROM files WHERE id = ?", (file_id,))
        result = cursor.fetchone()

        if result and result[0] == file_hash:
            print(f"File unchanged, skipping: {relative_path}")
            conn.close()
            return file_id

        conn.close()

        # Extract text
        content = self.processor.extract_text(file_path)
        if not content.strip():
            print(f"No content extracted from {file_path}")
            return None

        # Create chunks with LaTeX awareness
        metadata = {
            'file_path': relative_path,
            'source_id': source_id,
            'content_type': 'file',
            'file_type': file_path.suffix,
            'last_modified': timestamp()
        }

        chunks = self.processor.chunk_text(content, metadata)
        if not chunks:
            print(f"No chunks created from {file_path}")
            return None

        # Generate embeddings and store
        chunk_ids = []
        chunk_contents = []
        chunk_metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_contents.append(chunk['content'])
            chunk_metadatas.append(chunk['metadata'])

        # Store in ChromaDB
        try:
            self.collection.upsert(
                ids=chunk_ids,
                documents=chunk_contents,
                metadatas=chunk_metadatas
            )
        except Exception as e:
            print(f"Error storing chunks in ChromaDB: {e}")
            return None

        # Update file metadata
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO files
            (id, source_id, file_path, file_hash, last_modified, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (file_id, source_id, relative_path, file_hash, timestamp(), len(chunks)))
        conn.commit()
        conn.close()

        # Special logging for LaTeX files
        if file_path.suffix.lower() == '.tex':
            print(f"Processed LaTeX file {relative_path}: {len(chunks)} specialized chunks")
        else:
            print(f"Processed {relative_path}: {len(chunks)} chunks")
            
        return file_id

    def _update_source_metadata(self, source_id: str, source_type: str,
                               source_path: str, source_url: Optional[str],
                               commit_hash: Optional[str], file_count: int,
                               chunk_count: int, commit_count: int = 0):
        """Update source metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sources
            (id, source_type, source_path, source_url, commit_hash, last_indexed, file_count, chunk_count, commit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (source_id, source_type, source_path, source_url, commit_hash,
              timestamp(), file_count, chunk_count, commit_count))
        conn.commit()
        conn.close()

    def search(self, query: str, limit: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
        """Search documents and commits"""
        where_clause = {"source_id": source_filter} if source_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_clause
        )

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })

        return formatted_results

    def search_commits_by_ticket(self, ticket_id: str, source_filter: Optional[str] = None) -> List[Dict]:
        """Search specifically for commits related to a ticket ID"""
        where_clause = {
            "content_type": "git_commit"
        }
        
        if source_filter:
            where_clause["source_id"] = source_filter

        # Search for the ticket ID
        results = self.collection.query(
            query_texts=[f"ticket {ticket_id} commit"],
            n_results=20,  # Get more results for commits
            where=where_clause
        )

        # Filter results that actually contain the ticket ID
        filtered_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            content = results['documents'][0][i]
            
            # Check if ticket ID is in the content or metadata
            if (ticket_id.upper() in content.upper() or 
                (metadata.get('ticket_ids') and ticket_id.upper() in metadata['ticket_ids'].upper())):
                filtered_results.append({
                    'id': results['ids'][0][i],
                    'content': content,
                    'metadata': metadata,
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                })

        return filtered_results

    def query_with_llm(self, question: str, model: str = "claude",
                      source_filter: Optional[str] = None) -> str:
        """Query with LLM"""
        # Check if this looks like a ticket ID query
        ticket_pattern = re.compile(r'\b([A-Z]+-\d+)\b', re.IGNORECASE)
        ticket_matches = ticket_pattern.findall(question)
        
        if ticket_matches:
            # This appears to be a ticket-related query
            print(f"Detected ticket query for: {', '.join(ticket_matches)}")
            
            # Search for commits related to these tickets
            all_results = []
            for ticket_id in ticket_matches:
                commit_results = self.search_commits_by_ticket(ticket_id, source_filter)
                all_results.extend(commit_results)
            
            # Also do a general search
            general_results = self.search(question, limit=5, source_filter=source_filter)
            all_results.extend(general_results)
            
            # Remove duplicates and sort by relevance
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result['id'])
            
            # Sort by distance (lower is better)
            search_results = sorted(unique_results, key=lambda x: x['distance'])[:10]
        else:
            # Regular search
            search_results = self.search(question, limit=5, source_filter=source_filter)

        if not search_results:
            return "No relevant documents found."

        # Prepare context with special handling for LaTeX content
        context_parts = []
        for result in search_results:
            metadata = result['metadata']
            content_type = metadata.get('content_type', 'file')
            
            if content_type == 'git_commit':
                context_parts.append(f"Git Commit ({metadata.get('commit_hash', 'unknown')[:8]}):\n{result['content']}")
            else:
                file_path = metadata.get('file_path', 'unknown')
                file_type = metadata.get('file_type', '')
                
                # Special handling for LaTeX files
                if file_type == '.tex':
                    element_type = metadata.get('element_type', '')
                    if element_type:
                        context_parts.append(f"LaTeX File: {file_path} (Element: {element_type})\n{result['content']}")
                    else:
                        context_parts.append(f"LaTeX File: {file_path}\n{result['content']}")
                else:
                    context_parts.append(f"File: {file_path}\n{result['content']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Based on the following documents, git commits, and LaTeX content, please answer the question.

Context:
{context}

Question: {question}

Answer:"""

        # Query LLM
        if model == "claude" and self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model=Config.CLAUDE_MODEL,
                    max_tokens=Config.AI_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                return f"Error querying Claude: {e}"

        elif model in ["openai", "gpt"] and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=Config.AI_MAX_TOKENS
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error querying OpenAI: {e}"

        else:
            return f"Model '{model}' not available or not configured."

    def list_sources(self) -> List[Dict]:
        """List all sources"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, source_type, file_count, chunk_count, commit_count, last_indexed
            FROM sources ORDER BY last_indexed DESC
        ''')

        sources = []
        for row in cursor.fetchall():
            sources.append({
                'id': row[0],
                'type': row[1],
                'file_count': row[2],
                'chunk_count': row[3],
                'commit_count': row[4] if row[4] is not None else 0,
                'last_indexed': row[5]
            })

        conn.close()
        return sources

    def incremental_update(self, source_id: str) -> Dict[str, int]:
        """Incrementally update an existing source with new commits and changed files"""
        # Get source information
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT source_path, source_type FROM sources WHERE id = ?", (source_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"Source not found: {source_id}")
        
        source_path, source_type = result
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        print(f"Performing incremental update for source: {source_id}")
        print(f"Source path: {source_path}")
        
        stats = {
            'new_commits': 0,
            'updated_files': 0,
            'new_files': 0,
            'repositories_processed': 0,
            'latex_files_updated': 0
        }
        
        # Find all Git repositories in the directory tree
        git_repos = []
        if source_type == "directory":
            for item in source_path.rglob('.git'):
                if item.is_dir():
                    repo_root = item.parent
                    git_repos.append(repo_root)
            
            # If the root directory itself is a Git repository, add it
            if GitManager.is_git_repo(source_path) and source_path not in git_repos:
                git_repos.append(source_path)
        
        elif source_type == "git":
            git_repos = [source_path]
        
        # Process new commits from all discovered Git repositories
        total_new_commits = 0
        for repo_path in git_repos:
            print(f"Checking for new commits in: {repo_path}")
            try:
                new_commit_count = self._process_git_commits(repo_path, source_id, incremental=True)
                total_new_commits += new_commit_count
                if new_commit_count > 0:
                    print(f"Added {new_commit_count} new commits from {repo_path}")
                    stats['repositories_processed'] += 1
                else:
                    print(f"No new commits in {repo_path}")
            except Exception as e:
                print(f"Error processing commits from {repo_path}: {e}")
        
        stats['new_commits'] = total_new_commits
        
        # Check for changed files (based on modification time and hash)
        print("Checking for file changes...")
        file_changes = 0
        new_files = 0
        latex_updates = 0
        
        # Get existing files from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, file_hash, last_modified FROM files WHERE source_id = ?", (source_id,))
        existing_files = {row[0]: {'hash': row[1], 'modified': row[2]} for row in cursor.fetchall()}
        conn.close()
        
        # Process files
        for file_path in source_path.rglob("*"):
            if file_path.is_file():
                # Check if we should skip this file due to directory rules
                should_skip_dir = False
                for part in file_path.relative_to(source_path).parts[:-1]:  # Exclude filename
                    if part.lower() in self.processor.IGNORE_DIRECTORIES:
                        should_skip_dir = True
                        break

                if should_skip_dir:
                    continue

                if self.processor.should_process_file(file_path):
                    try:
                        relative_path = str(file_path.relative_to(source_path))
                        current_hash = self.calculate_file_hash(file_path)
                        
                        is_latex = file_path.suffix.lower() == '.tex'
                        
                        if relative_path in existing_files:
                            # Check if file has changed
                            if existing_files[relative_path]['hash'] != current_hash:
                                if is_latex:
                                    print(f"LaTeX file changed: {relative_path}")
                                    latex_updates += 1
                                else:
                                    print(f"File changed: {relative_path}")
                                    
                                file_id = self._process_file(file_path, source_id, str(source_path))
                                if file_id:
                                    file_changes += 1
                        else:
                            # New file
                            if is_latex:
                                print(f"New LaTeX file: {relative_path}")
                                latex_updates += 1
                            else:
                                print(f"New file: {relative_path}")
                                
                            file_id = self._process_file(file_path, source_id, str(source_path))
                            if file_id:
                                new_files += 1
                                
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        stats['updated_files'] = file_changes
        stats['new_files'] = new_files
        stats['latex_files_updated'] = latex_updates
        
        # Update source metadata
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sources 
            SET last_indexed = ?, commit_count = (
                SELECT COUNT(*) FROM git_commits WHERE source_id = ?
            )
            WHERE id = ?
        ''', (timestamp(), source_id, source_id))
        conn.commit()
        conn.close()
        
        print(f"Incremental update completed:")
        print(f"  - New commits: {stats['new_commits']}")
        print(f"  - Updated files: {stats['updated_files']}")
        print(f"  - New files: {stats['new_files']}")
        print(f"  - LaTeX files updated: {stats['latex_files_updated']}")
        print(f"  - Repositories processed: {stats['repositories_processed']}")
        
        return stats

    def reprocess_git_commits(self, source_id: str) -> int:
        """Reprocess Git commits for an existing source"""
        # Get source information
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT source_path, source_type FROM sources WHERE id = ?", (source_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"Source not found: {source_id}")
        
        source_path, source_type = result
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        print(f"Reprocessing Git commits for source: {source_id}")
        print(f"Source path: {source_path}")
        
        # Find all Git repositories in the directory tree
        git_repos = []
        if source_type == "directory":
            for item in source_path.rglob('.git'):
                if item.is_dir():
                    repo_root = item.parent
                    git_repos.append(repo_root)
                    print(f"Found Git repository: {repo_root}")
            
            # If the root directory itself is a Git repository, add it
            if GitManager.is_git_repo(source_path) and source_path not in git_repos:
                git_repos.append(source_path)
                print(f"Root directory is a Git repository: {source_path}")
        
        elif source_type == "git":
            git_repos = [source_path]
        
        if not git_repos:
            print("No Git repositories found")
            return 0
        
        # Remove existing commit data for this source
        print("Removing existing commit data...")
        try:
            # Remove from ChromaDB using correct where clause format
            results = self.collection.get(
                where={
                    "$and": [
                        {"source_id": {"$eq": source_id}},
                        {"content_type": {"$eq": "git_commit"}}
                    ]
                }
            )
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Removed {len(results['ids'])} commit chunks from vector store")
        except Exception as e:
            print(f"Error removing commit chunks from ChromaDB: {e}")
        
        # Remove from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM git_commits WHERE source_id = ?", (source_id,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"Removed {deleted_count} commit records from database")
        
        # Process commits from all discovered Git repositories
        total_commit_count = 0
        for repo_path in git_repos:
            print(f"Processing commits from: {repo_path}")
            try:
                commit_count = self._process_git_commits(repo_path, source_id, incremental=False)  # Full reprocessing
                total_commit_count += commit_count
                print(f"Processed {commit_count} commits from {repo_path}")
            except Exception as e:
                print(f"Error processing commits from {repo_path}: {e}")
        
        # Update source metadata with new commit count
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE sources SET commit_count = ? WHERE id = ?", (total_commit_count, source_id))
        conn.commit()
        conn.close()
        
        print(f"Reprocessed {total_commit_count} commits from {len(git_repos)} repositories")
        return total_commit_count

    def delete_source(self, source_id: str):
        """Delete a source and all its data"""
        # Delete from ChromaDB
        try:
            # Get all chunk IDs for this source
            results = self.collection.get(where={"source_id": source_id})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Removed {len(results['ids'])} chunks from vector store")
        except Exception as e:
            print(f"Error deleting from ChromaDB: {e}")

        # Delete from metadata database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete git commits if they exist
        cursor.execute("DELETE FROM git_commits WHERE source_id = ?", (source_id,))
        deleted_commits = cursor.rowcount
        if deleted_commits > 0:
            print(f"Removed {deleted_commits} commit records from database")
        
        # Delete files
        cursor.execute("DELETE FROM files WHERE source_id = ?", (source_id,))
        deleted_files = cursor.rowcount
        if deleted_files > 0:
            print(f"Removed {deleted_files} file records from database")
        
        # Delete source
        cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        deleted_sources = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_sources > 0:
            print(f"Deleted source: {source_id}")
        else:
            print(f"Source not found: {source_id}")

    def get_latex_document_structure(self, source_id: str, file_path: str = None) -> Dict:
        """Get structured information about LaTeX documents in a source"""
        where_clause = {
            "source_id": source_id,
            "file_type": ".tex"
        }
        
        if file_path:
            where_clause["file_path"] = file_path
        
        try:
            results = self.collection.get(where=where_clause)
            
            structure = {
                'sections': [],
                'theorems': [],
                'math_environments': [],
                'total_chunks': len(results['ids']) if results['ids'] else 0
            }
            
            for i, metadata in enumerate(results['metadatas']):
                element_type = metadata.get('element_type', '')
                
                if element_type == 'section':
                    structure['sections'].append({
                        'title': metadata.get('section_title', 'Unknown'),
                        'type': metadata.get('section_type', 'Unknown'),
                        'level': metadata.get('level', 0),
                        'file_path': metadata.get('file_path', ''),
                        'chunk_id': results['ids'][i]
                    })
                elif element_type == 'theorem':
                    structure['theorems'].append({
                        'env_type': metadata.get('env_type', 'Unknown'),
                        'file_path': metadata.get('file_path', ''),
                        'chunk_id': results['ids'][i]
                    })
                elif element_type == 'math':
                    structure['math_environments'].append({
                        'env_type': metadata.get('env_type', 'Unknown'),
                        'file_path': metadata.get('file_path', ''),
                        'chunk_id': results['ids'][i]
                    })
            
            # Sort sections by level and title
            structure['sections'].sort(key=lambda x: (x['level'], x['title']))
            
            return structure
            
        except Exception as e:
            print(f"Error getting LaTeX document structure: {e}")
            return {'sections': [], 'theorems': [], 'math_environments': [], 'total_chunks': 0}

def main():
    parser = argparse.ArgumentParser(description="Enhanced Local RAG Pipeline with LaTeX Support")
    parser.add_argument("--data-dir", default="./rag_data", help="Data directory")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest directory
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="Ingest directory")
    ingest_dir_parser.add_argument("path", help="Directory path")
    ingest_dir_parser.add_argument("--name", help="Source name")

    # Ingest single file
    ingest_file_parser = subparsers.add_parser("ingest-file", help="Ingest single file")
    ingest_file_parser.add_argument("path", help="File path")
    ingest_file_parser.add_argument("--name", help="Source name")

    # Ingest git repo
    ingest_git_parser = subparsers.add_parser("ingest-git", help="Ingest git repository")
    ingest_git_parser.add_argument("url", help="Git repository URL")
    ingest_git_parser.add_argument("--name", help="Source name")

    # Query
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--model", default="claude", choices=["claude", "openai", "gpt"], help="Model to use")
    query_parser.add_argument("--source", help="Filter by source ID")

    # Search
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--source", help="Filter by source ID")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    # Search commits by ticket
    search_ticket_parser = subparsers.add_parser("search-ticket", help="Search commits by ticket ID")
    search_ticket_parser.add_argument("ticket_id", help="Ticket ID (e.g., GET-1903)")
    search_ticket_parser.add_argument("--source", help="Filter by source ID")

    # LaTeX structure analysis
    latex_parser = subparsers.add_parser("latex-structure", help="Analyze LaTeX document structure")
    latex_parser.add_argument("source_id", help="Source ID containing LaTeX files")
    latex_parser.add_argument("--file", help="Specific file path to analyze")

    # Reprocess Git commits
    reprocess_parser = subparsers.add_parser("reprocess-commits", help="Reprocess Git commits for existing source")
    reprocess_parser.add_argument("source_id", help="Source ID to reprocess")

    # Incremental update
    incremental_parser = subparsers.add_parser("incremental-update", help="Incrementally update existing source with new commits and files")
    incremental_parser.add_argument("source_id", help="Source ID to update")

    # List sources
    subparsers.add_parser("list", help="List sources")

    # Delete source
    delete_parser = subparsers.add_parser("delete", help="Delete source")
    delete_parser.add_argument("source_id", help="Source ID to delete")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize pipeline
    try:
        rag = RAGPipeline(args.data_dir)
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        return 1

    try:
        if args.command == "ingest-dir":
            source_id = rag.ingest_directory(args.path, args.name)
            print(f"Ingested directory as source: {source_id}")

        elif args.command == "ingest-file":
            source_id = rag.ingest_single_file(args.path, args.name)
            print(f"Ingested file as source: {source_id}")

        elif args.command == "ingest-git":
            source_id = rag.ingest_git_repo(args.url, args.name)
            print(f"Ingested git repo as source: {source_id}")

        elif args.command == "query":
            result = rag.query_with_llm(args.question, args.model, args.source)
            print(result)

        elif args.command == "search":
            results = rag.search(args.query, args.limit, args.source)
            for i, result in enumerate(results, 1):
                content_type = result['metadata'].get('content_type', 'file')
                if content_type == 'git_commit':
                    print(f"\n--- Commit Result {i} ---")
                    print(f"Commit: {result['metadata'].get('commit_hash', 'unknown')[:8]}")
                    print(f"Author: {result['metadata'].get('author_name', 'unknown')}")
                    print(f"Distance: {result['distance']:.3f}")
                    print(f"Content: {result['content'][:300]}...")
                else:
                    file_type = result['metadata'].get('file_type', '')
                    element_type = result['metadata'].get('element_type', '')
                    
                    print(f"\n--- File Result {i} ---")
                    print(f"File: {result['metadata'].get('file_path', 'unknown')}")
                    if file_type == '.tex':
                        print(f"LaTeX Element: {element_type}")
                    print(f"Distance: {result['distance']:.3f}")
                    print(f"Content: {result['content'][:200]}...")

        elif args.command == "search-ticket":
            results = rag.search_commits_by_ticket(args.ticket_id, args.source)
            print(f"Found {len(results)} commits for ticket {args.ticket_id}:")
            for i, result in enumerate(results, 1):
                print(f"\n--- Commit {i} ---")
                print(f"Commit: {result['metadata'].get('commit_hash', 'unknown')[:8]}")
                print(f"Author: {result['metadata'].get('author_name', 'unknown')}")
                print(f"Subject: {result['metadata'].get('subject', 'unknown')}")
                print(f"Distance: {result['distance']:.3f}")
                print(f"Content preview: {result['content'][:300]}...")

        elif args.command == "latex-structure":
            structure = rag.get_latex_document_structure(args.source_id, args.file)
            print(f"\nLaTeX Document Structure Analysis")
            print(f"Total chunks: {structure['total_chunks']}")
            
            if structure['sections']:
                print(f"\nSections ({len(structure['sections'])}):")
                for section in structure['sections']:
                    indent = "  " * section['level']
                    print(f"{indent}- {section['type']}: {section['title']} ({section['file_path']})")
            
            if structure['theorems']:
                print(f"\nTheorem-like environments ({len(structure['theorems'])}):")
                for theorem in structure['theorems']:
                    print(f"  - {theorem['env_type']} ({theorem['file_path']})")
            
            if structure['math_environments']:
                print(f"\nMath environments ({len(structure['math_environments'])}):")
                for math_env in structure['math_environments']:
                    print(f"  - {math_env['env_type']} ({math_env['file_path']})")

        elif args.command == "reprocess-commits":
            commit_count = rag.reprocess_git_commits(args.source_id)
            print(f"Reprocessed {commit_count} commits for source: {args.source_id}")

        elif args.command == "incremental-update":
            stats = rag.incremental_update(args.source_id)
            print(f"Incremental update completed for source: {args.source_id}")
            print(f"Summary: {stats['new_commits']} new commits, {stats['updated_files']} updated files, {stats['new_files']} new files")
            if stats['latex_files_updated'] > 0:
                print(f"LaTeX files updated: {stats['latex_files_updated']}")

        elif args.command == "list":
            sources = rag.list_sources()
            print(f"\n{'ID':<30} {'Type':<10} {'Files':<6} {'Chunks':<7} {'Commits':<8} {'Last Indexed'}")
            print("-" * 90)
            for source in sources:
                print(f"{source['id']:<30} {source['type']:<10} {source['file_count']:<6} "
                      f"{source['chunk_count']:<7} {source['commit_count']:<8} {source['last_indexed'][:19]}")

        elif args.command == "delete":
            rag.delete_source(args.source_id)

    except Exception as e:
        print(f"Error executing command: {e}")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())