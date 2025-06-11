#!/usr/bin/env python3
"""
Local RAG Pipeline
A comprehensive RAG pipeline for ingesting and querying documents locally
"""

import argparse
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

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

def timestamp() -> str:
    """Return current timestamp"""
    return datetime.now().isoformat()

class DocumentProcessor:
    """Handles document parsing and chunking"""
    
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
        '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
        '.md', '.txt', '.rst', '.org', '.tex', '.json', '.yaml', '.yml',
        '.xml', '.html', '.css', '.sql', '.sh', '.bash', '.zsh',
        '.dockerfile', '.gitignore', '.env', '.toml', '.ini', '.cfg'
    }
    
    # Directory patterns to ignore (language-specific dependency and build directories)
    IGNORE_DIRECTORIES = {
        # JavaScript/Node.js
        'node_modules', 'bower_components', '.npm', '.yarn',
        
        # Python
        '__pycache__', '.pytest_cache', 'site-packages', 'dist', 'build',
        'egg-info', '.tox', '.coverage', '.mypy_cache',
        
        # Virtual environments
        'venv', '.venv', 'env', '.env', 'virtualenv',
        
        # Java/Maven/Gradle
        'target', 'build', '.gradle', '.m2',
        
        # .NET/C#
        'bin', 'obj', 'packages', '.nuget',
        
        # Go
        'vendor', 'pkg',
        
        # Rust
        'target', 'Cargo.lock',
        
        # Ruby
        'vendor', '.bundle', 'gems',
        
        # PHP
        'vendor', 'composer.lock',
        
        # C/C++
        'build', 'cmake-build-debug', 'cmake-build-release',
        
        # General IDE and tools
        '.git', '.svn', '.hg', '.bzr',
        '.idea', '.vscode', '.vs', '.DS_Store',
        '.next', '.nuxt', '.cache', 'cache',
        
        # Operating system
        'Thumbs.db', 'desktop.ini',
        
        # Logs and temporary files
        'logs', 'log', 'tmp', 'temp', '.tmp', '.temp'
    }
    
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
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
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
            # Skip files larger than 10MB (likely not source code)
            if file_size > 10 * 1024 * 1024:
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
        """Split text into chunks with metadata"""
        if not text.strip():
            return []
        
        # Simple sentence-aware chunking
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding sentence would exceed chunk size
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            if len(self.tokenizer.encode(test_chunk)) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': metadata.copy()
                })
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': metadata.copy()
            })
        
        return chunks

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
                timeout=300  # 5 minute timeout
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

class RAGPipeline:
    """Main RAG pipeline class"""
    
    def __init__(self, data_dir: str = "./rag_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.processor = DocumentProcessor()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize metadata database
        self.db_path = self.data_dir / "metadata.db"
        self.init_database()
        
        # Initialize API clients
        self.anthropic_client = None
        self.openai_client = None
        self.setup_api_clients()
    
    def init_database(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                source_type TEXT,
                source_path TEXT,
                source_url TEXT,
                commit_hash TEXT,
                last_indexed TIMESTAMP,
                file_count INTEGER,
                chunk_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                file_path TEXT,
                file_hash TEXT,
                last_modified TIMESTAMP,
                chunk_count INTEGER,
                FOREIGN KEY (source_id) REFERENCES sources (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
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
    
    def ingest_single_file(self, file_path: str, source_name: Optional[str] = None) -> str:
        """Ingest a single file"""
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.processor.should_process_file(file_path):
            raise ValueError(f"File type not supported: {file_path}")
        
        source_id = source_name or f"file_{file_path.stem}_{int(datetime.now().timestamp())}"
        
        print(f"Ingesting single file: {file_path}")
        
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
                                           None, None, 1, chunk_count)
                
                print(f"Successfully ingested file with {chunk_count} chunks")
                return source_id
            else:
                raise Exception("Failed to process file")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            raise
    
    def ingest_directory(self, dir_path: str, source_name: Optional[str] = None) -> str:
        """Ingest a local directory"""
        dir_path = Path(dir_path).resolve()
        source_id = source_name or f"dir_{dir_path.name}_{int(datetime.now().timestamp())}"
        
        print(f"Ingesting directory: {dir_path}")
        
        # Process files
        file_count = 0
        total_chunks = 0
        failed_files = []
        skipped_dirs = set()
        skipped_files = set()
        
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
                                    None, None, file_count, total_chunks)
        
        # Print summary
        print(f"Ingested {file_count} files, {total_chunks} chunks")
        
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
        """Ingest a git repository"""
        source_id = source_name or f"repo_{repo_url.split('/')[-1].replace('.git', '')}_{int(datetime.now().timestamp())}"
        repo_dir = self.data_dir / "repos" / source_id
        
        print(f"Cloning repository: {repo_url}")
        
        # Clone repository
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.parent.mkdir(exist_ok=True)
        
        if not GitManager.clone_repo(repo_url, repo_dir):
            raise Exception(f"Failed to clone repository: {repo_url}")
        
        # Get repo info
        repo_info = GitManager.get_repo_info(repo_dir)
        
        # Process files
        file_count = 0
        total_chunks = 0
        skipped_dirs = set()
        
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
                                    repo_info['commit_hash'], file_count, total_chunks)
        
        print(f"Ingested {file_count} files, {total_chunks} chunks")
        if skipped_dirs:
            print(f"Skipped directories: {', '.join(sorted(skipped_dirs))}")
        
        return source_id
    
    def _process_file(self, file_path: Path, source_id: str, base_path: str) -> Optional[str]:
        """Process a single file"""
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
        
        # Create chunks
        metadata = {
            'file_path': relative_path,
            'source_id': source_id,
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
        
        print(f"Processed {relative_path}: {len(chunks)} chunks")
        return file_id
    
    def _update_source_metadata(self, source_id: str, source_type: str, 
                               source_path: str, source_url: Optional[str],
                               commit_hash: Optional[str], file_count: int, 
                               chunk_count: int):
        """Update source metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sources 
            (id, source_type, source_path, source_url, commit_hash, last_indexed, file_count, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (source_id, source_type, source_path, source_url, commit_hash, 
              timestamp(), file_count, chunk_count))
        conn.commit()
        conn.close()
    
    def search(self, query: str, limit: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
        """Search documents"""
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
    
    def query_with_llm(self, question: str, model: str = "claude", 
                      source_filter: Optional[str] = None) -> str:
        """Query with LLM"""
        # Search for relevant documents
        search_results = self.search(question, limit=5, source_filter=source_filter)
        
        if not search_results:
            return "No relevant documents found."
        
        # Prepare context
        context = "\n\n".join([
            f"Document: {result['metadata']['file_path']}\n{result['content']}"
            for result in search_results
        ])
        
        prompt = f"""Based on the following documents, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Query LLM
        if model == "claude" and self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                return f"Error querying Claude: {e}"
        
        elif model in ["openai", "gpt"] and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
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
            SELECT id, source_type, file_count, chunk_count, last_indexed
            FROM sources ORDER BY last_indexed DESC
        ''')
        
        sources = []
        for row in cursor.fetchall():
            sources.append({
                'id': row[0],
                'type': row[1],
                'file_count': row[2],
                'chunk_count': row[3],
                'last_indexed': row[4]
            })
        
        conn.close()
        return sources
    
    def delete_source(self, source_id: str):
        """Delete a source and all its data"""
        # Delete from ChromaDB
        try:
            # Get all chunk IDs for this source
            results = self.collection.get(where={"source_id": source_id})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            print(f"Error deleting from ChromaDB: {e}")
        
        # Delete from metadata database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM files WHERE source_id = ?", (source_id,))
        cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        conn.commit()
        conn.close()
        
        print(f"Deleted source: {source_id}")

def main():
    parser = argparse.ArgumentParser(description="Local RAG Pipeline")
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
                print(f"\n--- Result {i} ---")
                print(f"File: {result['metadata']['file_path']}")
                print(f"Distance: {result['distance']:.3f}")
                print(f"Content: {result['content'][:200]}...")
        
        elif args.command == "list":
            sources = rag.list_sources()
            print(f"\n{'ID':<30} {'Type':<10} {'Files':<6} {'Chunks':<7} {'Last Indexed'}")
            print("-" * 80)
            for source in sources:
                print(f"{source['id']:<30} {source['type']:<10} {source['file_count']:<6} "
                      f"{source['chunk_count']:<7} {source['last_indexed'][:19]}")
        
        elif args.command == "delete":
            rag.delete_source(args.source_id)
    
    except Exception as e:
        print(f"Error executing command: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
