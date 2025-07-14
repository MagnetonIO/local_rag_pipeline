"""
Shared configuration module for RAG Pipeline CLI
"""

import os
from typing import Set

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass


class Config:
    """Configuration class for RAG Pipeline with environment variable support"""
    
    # Data storage
    DATA_DIR = os.getenv('RAG_DATA_DIR', './rag_data')
    DATABASE_NAME = os.getenv('RAG_DATABASE_NAME', 'metadata.db')
    VECTOR_STORE_DIR = os.getenv('RAG_VECTOR_STORE_DIR', 'chroma_db')
    REPOS_DIR = os.getenv('RAG_REPOS_DIR', 'repos')
    
    # Logging
    LOG_LEVEL = os.getenv('RAG_LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('RAG_LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    VERBOSE = os.getenv('RAG_VERBOSE', 'false').lower() == 'true'
    
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
    def get_supported_extensions() -> Set[str]:
        """Get supported file extensions from environment"""
        extensions_str = os.getenv('RAG_SUPPORTED_EXTENSIONS', 
            'py,js,ts,jsx,tsx,java,cpp,c,h,cs,php,rb,go,rs,swift,kt,scala,md,txt,rst,org,tex,json,yaml,yml,xml,html,css,sql,sh,bash,zsh,dockerfile,gitignore,env,toml,ini,cfg')
        return {f'.{ext.strip()}' for ext in extensions_str.split(',')}
    
    @staticmethod
    def get_ignored_directories() -> Set[str]:
        """Get ignored directory patterns from environment"""
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
    
    # MCP Server
    MCP_HOST = os.getenv('RAG_MCP_HOST', 'localhost')
    MCP_PORT = int(os.getenv('RAG_MCP_PORT', '8000'))
    MCP_TRANSPORT = os.getenv('RAG_MCP_TRANSPORT', 'stdio')
    MCP_ENABLE_RAG = os.getenv('RAG_MCP_ENABLE_RAG', 'true').lower() == 'true'


# Export config instance
config = Config()