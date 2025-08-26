#!/usr/bin/env python3
"""
Test Perplexity research functionality
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
        logging.FileHandler('perplexity_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test_task_classification():
    """Test task classification for research queries"""
    logger.info("Testing task classification...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from howie_cli.core.enhanced_agent import EnhancedHowieAgent
        
        agent = EnhancedHowieAgent(model='gpt-4o-mini')
        
        # Test queries that should trigger research
        research_queries = [
            "What are the latest offensive coordinator changes for 2024?",
            "Research current NFL news",
            "Find out about recent trades",
            "What's the latest on injuries this season?",
            "Search for 2024 draft information",
            "Who are the new head coaches in 2024?",
            "Latest updates on player signings",
            "Current playoff picture 2024"
        ]
        
        for query in research_queries:
            task_type = agent._classify_task(query)
            logger.info(f"Query: '{query}' -> Task: {task_type}")
            
    except Exception as e:
        logger.error(f"❌ Task classification test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_perplexity_model_selection():
    """Test that Perplexity models are selected for research tasks"""
    logger.info("Testing Perplexity model selection...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from howie_cli.core.enhanced_agent import EnhancedHowieAgent
        
        agent = EnhancedHowieAgent(model='gpt-4o-mini')
        
        # Check task mappings
        task_mappings = agent.model_manager.task_model_mapping
        logger.info(f"Task mappings: {task_mappings}")
        
        # Test that research maps to Perplexity
        research_model = task_mappings.get("research")
        logger.info(f"Research task maps to: {research_model}")
        
        if research_model and "perplexity" in research_model.lower():
            logger.info("✅ Research tasks correctly mapped to Perplexity")
        else:
            logger.warning("⚠️ Research tasks not mapped to Perplexity")
            
    except Exception as e:
        logger.error(f"❌ Model selection test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_perplexity_connection():
    """Test Perplexity API connection"""
    logger.info("Testing Perplexity API connection...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from howie_cli.core.model_manager import ModelManager
        
        model_manager = ModelManager()
        
        # Test Perplexity model
        logger.info("Testing perplexity-sonar model...")
        response = await model_manager.complete(
            messages=[{"role": "user", "content": "What are the latest NFL offensive coordinator changes for 2024?"}],
            model="perplexity-sonar",
            task_type="research"
        )
        
        logger.info(f"✅ Perplexity response received: {len(response)} characters")
        logger.info(f"Response preview: {response[:300]}...")
        
    except Exception as e:
        logger.error(f"❌ Perplexity connection test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_research_query():
    """Test a full research query through the agent"""
    logger.info("Testing full research query...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from howie_cli.core.enhanced_agent import EnhancedHowieAgent
        
        agent = EnhancedHowieAgent(model='gpt-4o-mini')
        
        # Test a research query
        research_query = "What are the latest offensive coordinator changes for the 2024 NFL season?"
        
        logger.info(f"Testing query: {research_query}")
        
        # Check what task type it's classified as
        task_type = agent._classify_task(research_query)
        logger.info(f"Task classification: {task_type}")
        
        # Check what model would be used
        model_name = agent.model_manager.task_model_mapping.get(task_type, "default")
        logger.info(f"Model that would be used: {model_name}")
        
        # Process the message
        response = await agent.process_message(research_query)
        
        logger.info(f"✅ Research response received: {len(response)} characters")
        logger.info(f"Response preview: {response[:500]}...")
        
    except Exception as e:
        logger.error(f"❌ Research query test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def main():
    """Run all Perplexity tests"""
    logger.info("=" * 60)
    logger.info("PERPLEXITY RESEARCH TEST")
    logger.info("=" * 60)
    
    # Test 1: Task classification
    await test_task_classification()
    
    # Test 2: Model selection
    await test_perplexity_model_selection()
    
    # Test 3: Perplexity connection
    await test_perplexity_connection()
    
    # Test 4: Full research query
    await test_research_query()
    
    logger.info("=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
