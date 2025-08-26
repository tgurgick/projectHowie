#!/usr/bin/env python3
"""
Test script for Howie CLI functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from howie_cli.core.agent import HowieAgent
from howie_cli.core.context import ConversationContext
from howie_cli.core.workspace import WorkspaceManager
from howie_cli.tools.registry import global_registry


async def test_basic_functionality():
    """Test basic agent functionality"""
    print("Testing Howie CLI Enhanced Features\n")
    print("=" * 50)
    
    # Test 1: Initialize agent
    print("\n1. Initializing agent...")
    try:
        agent = HowieAgent()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return False
    
    # Test 2: Context management
    print("\n2. Testing context management...")
    try:
        agent.context.add_message("user", "Test message")
        agent.context.add_player_context("Justin Jefferson", position="WR", team="MIN")
        summary = agent.context.get_conversation_summary()
        print(f"✅ Context working - Session ID: {summary['session_id']}")
    except Exception as e:
        print(f"❌ Context failed: {e}")
    
    # Test 3: Workspace management
    print("\n3. Testing workspace...")
    try:
        workspace = WorkspaceManager()
        test_data = {"test": "data"}
        file_path = workspace.write_file(test_data, "test.json")
        print(f"✅ Workspace created at: {workspace.session_path}")
    except Exception as e:
        print(f"❌ Workspace failed: {e}")
    
    # Test 4: Tool registry
    print("\n4. Testing tool registry...")
    try:
        tools = global_registry.list_tools()
        print(f"✅ {len(tools)} tools registered")
        
        categories = global_registry.get_categories()
        for cat in categories:
            cat_tools = global_registry.list_tools(cat)
            print(f"   - {cat}: {len(cat_tools)} tools")
    except Exception as e:
        print(f"❌ Tool registry failed: {e}")
    
    # Test 5: Simple query processing
    print("\n5. Testing query processing...")
    try:
        response = await agent.process_message("What tools do you have available?")
        print(f"✅ Query processed successfully")
        print(f"   Response length: {len(response)} characters")
    except Exception as e:
        print(f"❌ Query processing failed: {e}")
    
    # Test 6: Tool execution
    print("\n6. Testing tool execution...")
    try:
        from howie_cli.tools.code_generation_tools import GenerateSQLQueryTool
        tool = GenerateSQLQueryTool()
        result = await tool.execute(request="Show me top 10 QBs by fantasy points")
        if result.status.value == "success":
            print("✅ Tool execution successful")
        else:
            print(f"❌ Tool execution failed: {result.error}")
    except Exception as e:
        print(f"❌ Tool execution error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    
    return True


async def test_advanced_features():
    """Test advanced features"""
    print("\n\nTesting Advanced Features")
    print("=" * 50)
    
    agent = HowieAgent()
    
    # Test file operations
    print("\n1. Testing file operations...")
    test_queries = [
        "Create a test CSV file with sample player data",
        "Generate a Python script to analyze QB performance",
        "Create an ASCII chart showing fantasy points"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"   Test {i}: {query[:50]}...")
        try:
            response = await agent.process_message(query)
            print(f"   ✅ Success")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("Advanced testing complete!")


def main():
    """Main test function"""
    print("\n🏈 Howie CLI Enhanced - Test Suite 🏈\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Some tests may fail without API key")
        print("Set with: export OPENAI_API_KEY='your-key'\n")
    
    # Run basic tests
    success = asyncio.run(test_basic_functionality())
    
    if success and os.getenv("OPENAI_API_KEY"):
        # Run advanced tests only if API key is available
        asyncio.run(test_advanced_features())
    
    print("\n✨ All tests completed!")
    print("\nTo start using Howie CLI:")
    print("  python howie.py          # Start chat mode")
    print("  python howie.py --help   # See all commands")


if __name__ == "__main__":
    main()