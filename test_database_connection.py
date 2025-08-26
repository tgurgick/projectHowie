#!/usr/bin/env python3
"""
Database connection test script with logging
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_database_files():
    """Test if database files exist"""
    logger.info("Testing database file existence...")
    
    db_files = [
        "data/fantasy_ppr.db",
        "data/fantasy_halfppr.db", 
        "data/fantasy_standard.db"
    ]
    
    for db_file in db_files:
        path = Path(db_file)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {db_file} exists ({size_mb:.1f} MB)")
        else:
            logger.error(f"❌ {db_file} not found")
    
    return True

def test_sqlite_connection():
    """Test direct SQLite connections"""
    logger.info("Testing direct SQLite connections...")
    
    db_files = [
        "data/fantasy_ppr.db",
        "data/fantasy_halfppr.db",
        "data/fantasy_standard.db"
    ]
    
    for db_file in db_files:
        path = Path(db_file)
        if not path.exists():
            logger.error(f"❌ {db_file} not found, skipping")
            continue
            
        try:
            logger.info(f"Testing connection to {db_file}...")
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"✅ {db_file} connected successfully. Tables: {[t[0] for t in tables]}")
            
            # Test a simple query on each table
            for table in tables:
                table_name = table[0]
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    logger.info(f"  📊 {table_name}: {count} rows")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not query {table_name}: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to {db_file}: {e}")

def test_sqlalchemy_connection():
    """Test SQLAlchemy connections"""
    logger.info("Testing SQLAlchemy connections...")
    
    db_urls = [
        "sqlite:///data/fantasy_ppr.db",
        "sqlite:///data/fantasy_halfppr.db",
        "sqlite:///data/fantasy_standard.db"
    ]
    
    for db_url in db_urls:
        try:
            logger.info(f"Testing SQLAlchemy connection to {db_url}...")
            engine = create_engine(db_url, future=True, pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Test basic query
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                tables = [row[0] for row in result]
                logger.info(f"✅ {db_url} connected successfully. Tables: {tables}")
                
                # Test pandas read
                df = pd.read_sql_query(text("SELECT COUNT(*) as count FROM players"), conn)
                logger.info(f"  📊 Players table: {df['count'].iloc[0]} rows")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to {db_url}: {e}")

def test_database_tools():
    """Test the database tools from howie_cli"""
    logger.info("Testing howie_cli database tools...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from howie_cli.tools.database_tools import DatabaseQueryTool
        
        tool = DatabaseQueryTool()
        logger.info(f"✅ DatabaseQueryTool initialized successfully")
        logger.info(f"  Database paths: {tool.db_paths}")
        
        # Test each database
        for scoring_type, db_path in tool.db_paths.items():
            if Path(db_path).exists():
                logger.info(f"✅ {scoring_type} database exists at {db_path}")
            else:
                logger.error(f"❌ {scoring_type} database not found at {db_path}")
                
    except Exception as e:
        logger.error(f"❌ Failed to test database tools: {e}")

def main():
    """Run all database tests"""
    logger.info("=" * 60)
    logger.info("DATABASE CONNECTION TEST")
    logger.info("=" * 60)
    
    # Test 1: File existence
    test_database_files()
    
    # Test 2: Direct SQLite connections
    test_sqlite_connection()
    
    # Test 3: SQLAlchemy connections
    test_sqlalchemy_connection()
    
    # Test 4: Database tools
    test_database_tools()
    
    logger.info("=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
