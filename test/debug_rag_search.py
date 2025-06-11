#!/usr/bin/env python3
"""
Debug script to test RAG search functionality directly
Auto-detects which MCP server file you're using
"""

import sys
import os
import traceback

def test_rag_directly():
    """Test RAG pipeline directly without MCP"""
    try:
        print("🔍 Testing RAG pipeline directly...")
        
        # Import and initialize RAG
        from rag_pipeline import RAGPipeline
        rag = RAGPipeline("./rag_data")
        
        print("✅ RAG pipeline loaded successfully")
        
        # Test list_sources
        print("\n📁 Testing list_sources...")
        sources = rag.list_sources()
        print(f"Found {len(sources)} sources:")
        for source in sources:
            print(f"  • {source.get('id', 'Unknown')}: {source.get('file_count', 0)} files")
        
        # Test search with a simple query
        print("\n🔍 Testing search...")
        results = rag.search("physics", limit=3)
        print(f"Search results: {len(results)} documents")
        
        if results:
            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  File: {doc['metadata'].get('file_path', 'Unknown')}")
                print(f"  Distance: {doc.get('distance', 'Unknown')}")
                
                # Handle both 'text' and 'content' keys
                content = doc.get('text') or doc.get('content', 'No content found')
                print(f"  Content preview: {content[:100]}...")
                print(f"  Available keys: {list(doc.keys())}")  # Debug info
        else:
            print("No search results returned")
        
        # Test query_with_context
        print("\n🤖 Testing query_with_context...")
        response, context_docs = rag.query_with_context("What is functorial physics?", max_context_chunks=2)
        print(f"Response: {response[:200]}...")
        print(f"Context docs: {len(context_docs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        traceback.print_exc()
        return False

def detect_server_file():
    """Detect which MCP server file exists and is being used"""
    
    possible_files = [
        "modern_mcp_server.py",
        "mcp_rag_server.py", 
        "debug_mcp_server.py",
        "final_mcp_server.py"
    ]
    
    print("🔍 Detecting MCP server files...")
    found_files = []
    
    for filename in possible_files:
        if os.path.exists(filename):
            print(f"  ✅ Found: {filename}")
            found_files.append(filename)
        else:
            print(f"  ❌ Not found: {filename}")
    
    return found_files

def test_server_functions(server_module_name):
    """Test server functions directly"""
    try:
        print(f"\n🔧 Testing functions from {server_module_name}...")
        
        # Import the server module
        sys.path.append('.')
        server_module = __import__(server_module_name.replace('.py', ''))
        
        # Check if it's the modern FastMCP server or the old style
        if hasattr(server_module, 'mcp'):
            # Modern FastMCP server
            print("  📦 Detected: Modern FastMCP server")
            
            # Initialize RAG
            server_module.initialize_rag("./rag_data")
            
            # Test the decorated functions directly
            if hasattr(server_module, 'ping'):
                ping_result = server_module.ping()
                print(f"  🏓 Ping: {ping_result}")
            
            if hasattr(server_module, 'server_status'):
                status_result = server_module.server_status()
                print(f"  📊 Status: {status_result[:100]}...")
            
            if hasattr(server_module, 'search_documents'):
                search_result = server_module.search_documents("physics", limit=2)
                print(f"  🔍 Search: {search_result[:100]}...")
            
        elif hasattr(server_module, 'MCPRAGServer') or hasattr(server_module, 'WorkingMCPServer'):
            # Old style server with class
            print("  📦 Detected: Class-based MCP server")
            print("  ⚠️  Cannot test functions directly (they're in a class)")
            print("  💡 Recommendation: Use modern_mcp_server.py instead")
            
        else:
            print("  ❓ Unknown server type")
            print(f"  📋 Available attributes: {[attr for attr in dir(server_module) if not attr.startswith('_')]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server function test failed: {e}")
        traceback.print_exc()
        return False

def test_search_manually():
    """Manually test search functionality"""
    try:
        print("\n🔍 Manual search test...")
        
        from rag_pipeline import RAGPipeline
        rag = RAGPipeline("./rag_data")
        
        # Test specific search terms that should work with functorial physics
        test_queries = ["functor", "category", "physics", "mathematical"]
        
        for query in test_queries:
            print(f"\n  Testing query: '{query}'")
            try:
                results = rag.search(query, limit=2)
                print(f"    Results: {len(results)}")
                if results:
                    print(f"    First result: {results[0]['metadata'].get('file_path', 'Unknown')}")
                    print(f"    Similarity: {1 - results[0].get('distance', 1):.3f}")
            except Exception as e:
                print(f"    Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual search test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Debugging RAG Search Issues")
    print("=" * 50)
    
    # Detect server files
    server_files = detect_server_file()
    
    # Test 1: Direct RAG functionality
    print("\n" + "=" * 30)
    rag_success = test_rag_directly()
    
    # Test 2: Manual search test
    print("\n" + "=" * 30)
    manual_success = test_search_manually()
    
    # Test 3: Server functions (if we found server files)
    server_success = False
    if server_files:
        print("\n" + "=" * 30)
        # Test the first available server file
        server_success = test_server_functions(server_files[0])
    
    print("\n" + "=" * 50)
    print("🎯 Debug Summary:")
    print(f"  RAG Direct Test: {'✅ PASS' if rag_success else '❌ FAIL'}")
    print(f"  Manual Search Test: {'✅ PASS' if manual_success else '❌ FAIL'}")
    print(f"  Server Functions Test: {'✅ PASS' if server_success else '❌ FAIL'}")
    print(f"  Server Files Found: {', '.join(server_files) if server_files else 'None'}")
    
    if rag_success and manual_success:
        print("\n✅ RAG pipeline works! Issue is likely in MCP communication.")
        print("💡 Recommendations:")
        print("  1. Restart Claude Desktop")
        print("  2. Check Claude Desktop config points to the right server file")
        print("  3. Make sure you're using modern_mcp_server.py (FastMCP)")
    elif not rag_success:
        print("\n❌ RAG pipeline has issues. Check your data directory and dependencies.")
