#!/usr/bin/env python3
"""
Test web search functionality
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_search_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test_web_search():
    """Test web search functionality"""
    logger.info("Testing web search functionality...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from howie_cli.tools.realtime_tools import search_current_nfl_info, get_latest_coaching_changes
        
        # Test 1: General search
        logger.info("Testing general NFL news search...")
        result = await search_current_nfl_info("NFL news today")
        logger.info(f"General search result length: {len(result)} characters")
        logger.info(f"Result preview: {result[:300]}...")
        
        # Test 2: Coaching changes search
        logger.info("Testing coaching changes search...")
        coaching_result = await get_latest_coaching_changes()
        logger.info(f"Coaching changes result length: {len(coaching_result)} characters")
        logger.info(f"Result preview: {coaching_result[:300]}...")
        
        # Test 3: Specific query
        logger.info("Testing specific query...")
        specific_result = await search_current_nfl_info("offensive coordinator changes 2024")
        logger.info(f"Specific query result length: {len(specific_result)} characters")
        logger.info(f"Result preview: {specific_result[:300]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Web search test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def test_enhanced_agent_with_web_search():
    """Test enhanced agent with web search"""
    logger.info("Testing enhanced agent with web search...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from howie_cli.core.enhanced_agent import EnhancedHowieAgent
        
        agent = EnhancedHowieAgent(model='gpt-4o-mini')
        
        # Test research query
        research_query = "What are the latest offensive coordinator changes for the 2024 NFL season?"
        
        logger.info(f"Testing research query: {research_query}")
        
        # Check task classification
        task_type = agent._classify_task(research_query)
        logger.info(f"Task classification: {task_type}")
        
        # Process the message
        response = await agent.process_message(research_query)
        
        logger.info(f"✅ Enhanced agent response length: {len(response)} characters")
        logger.info(f"Response preview: {response[:500]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced agent test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all web search tests"""
    logger.info("=" * 60)
    logger.info("WEB SEARCH FUNCTIONALITY TEST")
    logger.info("=" * 60)
    
    # Test 1: Web search tools
    web_search_success = await test_web_search()
    
    # Test 2: Enhanced agent with web search
    agent_success = await test_enhanced_agent_with_web_search()
    
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Web search tools: {'✅ PASS' if web_search_success else '❌ FAIL'}")
    logger.info(f"Enhanced agent: {'✅ PASS' if agent_success else '❌ FAIL'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
