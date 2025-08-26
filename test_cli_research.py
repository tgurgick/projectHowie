#!/usr/bin/env python3
"""
Test CLI research functionality
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_research_query():
    """Test a research query through the CLI"""
    try:
        from howie_cli.core.enhanced_agent import EnhancedHowieAgent
        
        # Create agent
        agent = EnhancedHowieAgent(model='gpt-4o-mini')
        
        # Test research query
        query = "What are the latest offensive coordinator changes for the 2024 NFL season?"
        
        print(f"Testing query: {query}")
        print("=" * 60)
        
        # Process the query
        response = await agent.process_message(query)
        
        print("Response:")
        print(response)
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_research_query())
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
