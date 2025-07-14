#!/usr/bin/env python3
"""
RAG Pipeline - Professional CLI for Document Intelligence
A comprehensive tool for ingesting, indexing, and querying documents with AI
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration and setup logging
from cli.config import config

# Setup logging with WARNING as default to suppress INFO logs
logging.basicConfig(
    level=logging.WARNING,
    format=config.LOG_FORMAT
)
# Suppress specific noisy loggers by default
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class RAGPipelineCLI:
    """Professional CLI interface for RAG Pipeline operations"""
    
    def __init__(self):
        self.rag = None
        self.data_dir = None
        
    def setup_system(self, data_dir: str) -> bool:
        """Initialize the RAG pipeline system"""
        try:
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            # Only log if DEBUG level is enabled (verbose mode)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logger.debug(f"RAG Pipeline data directory: {self.data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def get_rag_pipeline(self) -> 'RAGPipeline':
        """Lazy initialization of RAG pipeline"""
        if self.rag is None:
            from rag_pipeline import RAGPipeline
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logger.debug(f"Initializing RAG Pipeline with data directory: {self.data_dir}")
            self.rag = RAGPipeline(str(self.data_dir))
        return self.rag
    
    def ensure_database_ready(self) -> bool:
        """Ensure database is ready (run migrations if needed)"""
        try:
            db_path = self.data_dir / config.DATABASE_NAME
            if db_path.exists():
                from cli.database_manager import DatabaseManager
                db_manager = DatabaseManager(str(db_path))
                if db_manager.needs_migration():
                    if logging.getLogger().isEnabledFor(logging.INFO):
                        logger.info("Database needs migration, running migrations...")
                    return db_manager.migrate()
            return True
        except Exception as e:
            logger.error(f"Failed to ensure database is ready: {e}")
            return False


def create_parser():
    """Create the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog='rag',
        description='RAG Pipeline - Professional Document Intelligence CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Command Categories:

INGESTION:
  ingest-dir PATH            Ingest a directory of documents
  ingest-file PATH           Ingest a single file
  ingest-git URL             Ingest a Git repository with commit history

SEARCH & QUERY:
  search QUERY               Semantic search across documents
  query QUESTION             Ask questions using AI (Claude/OpenAI)
  search-ticket TICKET_ID    Search Git commits by ticket ID

ANALYSIS:
  latex-structure SOURCE_ID  Analyze LaTeX document structure

MAINTENANCE:
  reprocess-commits SOURCE   Reprocess Git commits for a source
  incremental-update SOURCE  Update source with new/changed content
  list                       List all data sources
  delete SOURCE_ID           Delete a source and its data

SYSTEM:
  status                     Show system status
  migrate                    Database migration commands

Examples:
  rag ingest-dir ./docs --name "project-docs"
  rag ingest-git https://github.com/user/repo --name "my-project"
  rag search "authentication flow"
  rag query "How does the login system work?" --model claude
  rag search-ticket "JIRA-123"
  rag list
  rag delete old-source --confirm
"""
    )
    
    # Global options
    parser.add_argument(
        '--data-dir',
        default=config.DATA_DIR,
        help=f'Data directory for storage (default: {config.DATA_DIR})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # ===== INGESTION COMMANDS =====
    
    # ingest-dir
    ingest_dir = subparsers.add_parser('ingest-dir', help='Ingest a directory')
    ingest_dir.add_argument('path', help='Directory path')
    ingest_dir.add_argument('--name', help='Source name (default: directory name)')
    
    # ingest-file
    ingest_file = subparsers.add_parser('ingest-file', help='Ingest a single file')
    ingest_file.add_argument('path', help='File path')
    ingest_file.add_argument('--name', help='Source name (default: file name)')
    
    # ingest-git
    ingest_git = subparsers.add_parser('ingest-git', help='Ingest a Git repository')
    ingest_git.add_argument('url', help='Repository URL')
    ingest_git.add_argument('--name', help='Source name (default: repo name)')
    
    # ===== SEARCH COMMANDS =====
    
    # search
    search = subparsers.add_parser('search', help='Search documents')
    search.add_argument('query', help='Search query')
    search.add_argument('--limit', type=int, default=config.DEFAULT_SEARCH_LIMIT, 
                       help=f'Number of results (default: {config.DEFAULT_SEARCH_LIMIT})')
    search.add_argument('--source', help='Filter by source ID')
    
    # query
    query = subparsers.add_parser('query', help='Query with AI')
    query.add_argument('question', help='Question to ask')
    query.add_argument('--model', choices=['claude', 'openai'], default=config.DEFAULT_AI_MODEL,
                      help=f'AI model (default: {config.DEFAULT_AI_MODEL})')
    query.add_argument('--source', help='Filter by source ID')
    
    # search-ticket
    search_ticket = subparsers.add_parser('search-ticket', help='Search commits by ticket')
    search_ticket.add_argument('ticket_id', help='Ticket ID (e.g., JIRA-123)')
    search_ticket.add_argument('--limit', type=int, default=config.DEFAULT_TICKET_SEARCH_LIMIT, 
                              help=f'Number of results (default: {config.DEFAULT_TICKET_SEARCH_LIMIT})')
    search_ticket.add_argument('--source', help='Filter by source ID')
    
    # ===== ANALYSIS COMMANDS =====
    
    # latex-structure
    latex = subparsers.add_parser('latex-structure', help='Analyze LaTeX structure')
    latex.add_argument('source_id', help='Source ID containing LaTeX documents')
    
    # ===== MAINTENANCE COMMANDS =====
    
    # reprocess-commits
    reprocess = subparsers.add_parser('reprocess-commits', help='Reprocess Git commits')
    reprocess.add_argument('source_id', help='Source ID to reprocess')
    
    # incremental-update
    update = subparsers.add_parser('incremental-update', help='Update source content')
    update.add_argument('source_id', help='Source ID to update')
    
    # list
    subparsers.add_parser('list', help='List all sources')
    
    # delete
    delete = subparsers.add_parser('delete', help='Delete a source')
    delete.add_argument('source_id', help='Source ID to delete')
    delete.add_argument('--confirm', action='store_true', help='Skip confirmation')
    
    # ===== SYSTEM COMMANDS =====
    
    # status
    status = subparsers.add_parser('status', help='Show system status')
    status.add_argument('--show-sources', action='store_true', help='Show source details')
    
    # migrate
    migrate = subparsers.add_parser('migrate', help='Database migrations')
    migrate_sub = migrate.add_subparsers(dest='migrate_action')
    
    migrate_status = migrate_sub.add_parser('status', help='Show migration status')
    migrate_up = migrate_sub.add_parser('up', help='Run migrations')
    migrate_down = migrate_sub.add_parser('down', help='Rollback last migration')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging based on verbose flag or config
    if args.verbose or config.VERBOSE:
        logging.getLogger().setLevel(logging.DEBUG)
    elif hasattr(config, 'LOG_LEVEL') and config.LOG_LEVEL:
        # Use configured log level if set
        log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.WARNING)
        logging.getLogger().setLevel(log_level)
    else:
        # Default to WARNING level to suppress INFO logs from sentence transformers etc
        logging.getLogger().setLevel(logging.WARNING)
        # Also suppress specific noisy loggers
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Special handling for migrate command
    if args.command == 'migrate':
        from cli.commands.migrate import handle_migrate_command
        args.db_path = str(Path(args.data_dir) / config.DATABASE_NAME)
        return 0 if handle_migrate_command(args) else 1
    
    # Initialize CLI
    cli = RAGPipelineCLI()
    
    # Setup system (minimal setup, no RAG pipeline initialization yet)
    if not cli.setup_system(args.data_dir):
        logger.error("Failed to initialize system")
        return 1
    
    try:
        # Route commands to modular handlers
        success = False
        
        # Import command handlers
        from cli.commands import ingest, search, analysis, maintenance, status as status_cmd
        
        # Commands that need RAG pipeline
        needs_rag = args.command in ['ingest-dir', 'ingest-file', 'ingest-git', 'search', 'query', 
                                    'search-ticket', 'latex-structure', 'reprocess-commits', 
                                    'incremental-update', 'list', 'delete']
        # Status command needs RAG pipeline only if showing sources
        if args.command == 'status' and args.show_sources:
            needs_rag = True
            
        if needs_rag:
            # Ensure database is ready before initializing RAG pipeline
            if not cli.ensure_database_ready():
                logger.error("Failed to prepare database")
                return 1
            # Get RAG pipeline (lazy initialization)
            rag = cli.get_rag_pipeline()
        else:
            rag = None
        
        if args.command == 'ingest-dir':
            success = ingest.ingest_directory(rag, args.path, args.name)
        elif args.command == 'ingest-file':
            success = ingest.ingest_file(rag, args.path, args.name)
        elif args.command == 'ingest-git':
            success = ingest.ingest_git(rag, args.url, args.name)
        elif args.command == 'search':
            success = search.search(rag, args.query, args.limit, args.source)
        elif args.command == 'query':
            success = search.query(rag, args.question, args.model, args.source)
        elif args.command == 'search-ticket':
            success = search.search_ticket(rag, args.ticket_id, args.limit, args.source)
        elif args.command == 'latex-structure':
            success = analysis.latex_structure(rag, args.source_id)
        elif args.command == 'reprocess-commits':
            success = maintenance.reprocess_commits(rag, args.source_id)
        elif args.command == 'incremental-update':
            success = maintenance.incremental_update(rag, args.source_id)
        elif args.command == 'list':
            success = maintenance.list_sources(rag)
        elif args.command == 'delete':
            success = maintenance.delete_source(rag, args.source_id, args.confirm)
        elif args.command == 'status':
            success = status_cmd.status(rag, cli.data_dir, args.show_sources)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose or config.VERBOSE:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())