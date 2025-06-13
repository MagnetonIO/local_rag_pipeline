#!/usr/bin/env python3
"""
RAG Pipeline - Main CLI Entry Point
Professional RAG pipeline with proper migration system
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import command handlers
from cli.commands.migrate import add_migrate_commands, handle_migrate_command
from cli.commands.ingest import add_ingest_commands, handle_ingest_command
from cli.commands.search import add_search_commands, handle_search_command
from cli.commands.status import add_status_commands, handle_status_command


def create_parser():
    """Create the main CLI parser"""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Professional document ingestion and search system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Database migrations
  python main.py migrate status --db-path ./data/db.sqlite
  python main.py migrate up --db-path ./data/db.sqlite
  
  # Ingest documents
  python main.py ingest directory ./documents --name "my-docs"
  python main.py ingest git https://github.com/user/repo --name "my-repo"
  
  # Search and query
  python main.py search "what is the main function" --limit 5
  python main.py query "explain the authentication system" --model claude
  
  # Status and management
  python main.py status --show-sources
        """
    )
    
    parser.add_argument(
        '--data-dir', 
        default='./rag_data',
        help='Data directory for database and vector store (default: ./rag_data)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers for different command categories
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add command groups
    add_migrate_commands(subparsers)
    add_ingest_commands(subparsers)  
    add_search_commands(subparsers)
    add_status_commands(subparsers)
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure data directory exists
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Set database path for migration commands
    if hasattr(args, 'db_path') and not args.db_path:
        args.db_path = str(data_dir / "metadata.db")
    
    try:
        # Route to appropriate command handler
        success = False
        
        if args.command == 'migrate':
            success = handle_migrate_command(args)
        elif args.command == 'ingest':
            success = handle_ingest_command(args)
        elif args.command == 'search':
            success = handle_search_command(args)
        elif args.command == 'status':
            success = handle_status_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
