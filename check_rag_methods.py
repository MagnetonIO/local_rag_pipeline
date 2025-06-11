#!/usr/bin/env python3
"""
Check what methods are available in the RAG pipeline
"""

def check_rag_pipeline():
    """Check what methods and attributes the RAG pipeline has"""
    try:
        from rag_pipeline import RAGPipeline
        
        # Create instance
        rag = RAGPipeline("./rag_data")
        
        print("🔍 RAG Pipeline Analysis:")
        print(f"📦 Class: {rag.__class__.__name__}")
        print(f"📋 Module: {rag.__class__.__module__}")
        
        # Get all methods and attributes
        all_attrs = dir(rag)
        methods = [attr for attr in all_attrs if not attr.startswith('_') and callable(getattr(rag, attr))]
        properties = [attr for attr in all_attrs if not attr.startswith('_') and not callable(getattr(rag, attr))]
        
        print(f"\n🔧 Available Methods ({len(methods)}):")
        for method in sorted(methods):
            try:
                method_obj = getattr(rag, method)
                if hasattr(method_obj, '__doc__') and method_obj.__doc__:
                    doc = method_obj.__doc__.strip().split('\n')[0][:80]
                    print(f"  • {method}() - {doc}")
                else:
                    print(f"  • {method}()")
            except:
                print(f"  • {method}() - [error getting info]")
        
        print(f"\n📊 Available Properties ({len(properties)}):")
        for prop in sorted(properties):
            try:
                value = getattr(rag, prop)
                print(f"  • {prop}: {type(value).__name__}")
            except:
                print(f"  • {prop}: [error getting value]")
        
        # Test the search method specifically
        print(f"\n🔍 Testing search method:")
        try:
            import inspect
            search_sig = inspect.signature(rag.search)
            print(f"  Signature: search{search_sig}")
            
            # Test a simple search
            results = rag.search("test", limit=1)
            print(f"  Test search returned: {len(results)} results")
            if results:
                print(f"  Result structure: {list(results[0].keys())}")
                
        except Exception as e:
            print(f"  Search test failed: {e}")
        
        # Check if there are any query-related methods
        query_methods = [m for m in methods if 'query' in m.lower()]
        if query_methods:
            print(f"\n🤖 Query-related methods found:")
            for method in query_methods:
                print(f"  • {method}()")
        else:
            print(f"\n🤖 No query-related methods found")
            print("  💡 Will need to implement query_with_context using search + AI")
        
        return rag, methods
        
    except Exception as e:
        print(f"❌ Failed to analyze RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, []

if __name__ == "__main__":
    rag, methods = check_rag_pipeline()
    
    if rag and methods:
        print(f"\n✅ RAG pipeline analysis complete!")
        print(f"📝 The pipeline has {len(methods)} methods available")
        
        if 'query_with_context' not in methods:
            print(f"\n💡 Recommendation:")
            print(f"  The RAG pipeline doesn't have query_with_context()")
            print(f"  We can implement this by:")
            print(f"  1. Using search() to find relevant documents")
            print(f"  2. Using OpenAI API to generate answers from the context")
            print(f"  3. Or creating a wrapper method in the MCP server")
