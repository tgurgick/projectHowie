#!/usr/bin/env python3
"""
Test agent with logging to see what's breaking
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test_agent_initialization():
    """Test agent initialization with logging"""
    logger.info("Testing agent initialization...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        logger.info("Importing EnhancedHowieAgent...")
        from howie_cli.core.enhanced_agent import EnhancedHowieAgent
        
        logger.info("Creating agent instance...")
        agent = EnhancedHowieAgent(model='gpt-4o-mini')
        
        logger.info(f"✅ Agent initialized successfully")
        logger.info(f"  Current model: {agent.model_manager.current_model}")
        logger.info(f"  API key set: {'Yes' if agent.api_key else 'No'}")
        
        return agent
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def test_database_tool():
    """Test database tool specifically"""
    logger.info("Testing database tool...")
    
    try:
        from howie_cli.tools.database_tools import DatabaseQueryTool
        
        tool = DatabaseQueryTool()
        logger.info(f"✅ DatabaseQueryTool created")
        
        # Test a simple query
        logger.info("Testing simple database query...")
        result = await tool.execute(
            query="SELECT COUNT(*) as count FROM players",
            scoring_type="ppr",
            return_type="summary"
        )
        
        logger.info(f"✅ Database query result: {result.status}")
        if result.status.value == "success":
            logger.info(f"  Data: {result.data}")
        else:
            logger.error(f"  Error: {result.error}")
            
    except Exception as e:
        logger.error(f"❌ Database tool test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_simple_chat():
    """Test a simple chat interaction"""
    logger.info("Testing simple chat...")
    
    try:
        agent = await test_agent_initialization()
        if not agent:
            return
            
        logger.info("Testing simple message processing...")
        response = await agent.process_message("Hello, can you tell me about yourself?")
        
        logger.info(f"✅ Chat response received: {len(response)} characters")
        logger.info(f"  Response preview: {response[:200]}...")
        
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("AGENT TEST WITH LOGGING")
    logger.info("=" * 60)
    
    # Test 1: Agent initialization
    await test_agent_initialization()
    
    # Test 2: Database tool
    await test_database_tool()
    
    # Test 3: Simple chat
    await test_simple_chat()
    
    logger.info("=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
