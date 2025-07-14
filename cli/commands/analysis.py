"""
Analysis command handlers for RAG Pipeline CLI
"""

import logging

logger = logging.getLogger(__name__)


def latex_structure(rag_pipeline, source_id: str) -> bool:
    """Analyze LaTeX document structure"""
    try:
        print(f"📚 Analyzing LaTeX structure for source: {source_id}")
        
        structure = rag_pipeline.get_latex_document_structure(source_id)
        
        if not structure:
            print("\nNo LaTeX documents found in this source")
            return True
        
        print("\n📋 Document Structure:")
        print("─" * 60)
        
        def print_section(section, indent=0):
            prefix = "  " * indent
            print(f"{prefix}• {section['title']} ({section['type']})")
            for subsection in section.get('subsections', []):
                print_section(subsection, indent + 1)
        
        for doc in structure:
            print(f"\n📄 {doc['title']}")
            for section in doc['sections']:
                print_section(section)
        
        return True
        
    except Exception as e:
        logger.error(f"LaTeX analysis failed: {e}")
        return False