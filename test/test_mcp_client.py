#!/usr/bin/env python3
"""
Test client for the MCP RAG server
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """Test the MCP server functionality"""
    
    # Configure the server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["modern_mcp_server.py", "--data-dir", "./rag_data"]
    )
    
    print("🔌 Connecting to MCP server...")
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("✅ Connected successfully!")
                
                # Initialize the session
                await session.initialize()
                print("✅ Session initialized")
                
                # Test 1: List available tools
                print("\n📋 Listing available tools...")
                tools = await session.list_tools()
                print(f"Found {len(tools.tools)} tools:")
                for tool in tools.tools:
                    print(f"  • {tool.name}: {tool.description}")
                
                # Test 2: List available resources
                print("\n📄 Listing available resources...")
                resources = await session.list_resources()
                print(f"Found {len(resources.resources)} resources:")
                for resource in resources.resources:
                    print(f"  • {resource.name}: {resource.description}")
                
                # Test 3: Call the ping tool
                print("\n🏓 Testing ping tool...")
                ping_result = await session.call_tool("ping", {})
                print(f"Ping result: {ping_result.content[0].text}")
                
                # Test 4: Get server status
                print("\n📊 Getting server status...")
                status_result = await session.call_tool("server_status", {})
                print("Server Status:")
                print(status_result.content[0].text)
                
                # Test 5: List sources (if RAG is loaded)
                print("\n📁 Listing sources...")
                sources_result = await session.call_tool("list_sources", {})
                print("Sources:")
                print(sources_result.content[0].text)
                
                # Test 6: Try a search (if RAG is loaded)
                print("\n🔍 Testing search...")
                search_result = await session.call_tool("search_documents", {
                    "query": "test",
                    "limit": 3
                })
                print("Search Results:")
                print(search_result.content[0].text)
                
                # Test 7: Read a resource
                print("\n📖 Reading status resource...")
                status_resource = await session.read_resource("rag://status")
                print("Status Resource:")
                status_data = json.loads(status_resource.contents[0].text)
                print(json.dumps(status_data, indent=2))
                
                print("\n✅ All tests completed successfully!")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Starting MCP Server Tests")
    asyncio.run(test_mcp_server())