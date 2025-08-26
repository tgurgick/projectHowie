"""
Real-time data tools for getting current information
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Tool for searching the web for current information"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_nfl_news(self, query: str) -> Dict:
        """Search for NFL news and current information"""
        try:
            # Search multiple sources for comprehensive results
            results = {}
            
            # ESPN NFL News
            espn_results = await self._search_espn(query)
            if espn_results:
                results['ESPN'] = espn_results
            
            # NFL.com News
            nfl_results = await self._search_nfl_com(query)
            if nfl_results:
                results['NFL.com'] = nfl_results
            
            # Bleacher Report
            br_results = await self._search_bleacher_report(query)
            if br_results:
                results['Bleacher Report'] = br_results
            
            return {
                'success': True,
                'sources': results,
                'timestamp': datetime.now().isoformat(),
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error searching NFL news: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'query': query
            }
    
    async def _search_espn(self, query: str) -> Optional[List[Dict]]:
        """Search ESPN for NFL news"""
        try:
            # ESPN NFL news URL
            url = "https://www.espn.com/nfl/news"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    articles = []
                    # Look for article links
                    for article in soup.find_all('a', href=True):
                        href = article.get('href', '')
                        text = article.get_text(strip=True)
                        
                        # Filter for relevant articles
                        if any(keyword in text.lower() for keyword in query.lower().split()):
                            if 'espn.com' in href and '/story/' in href:
                                articles.append({
                                    'title': text,
                                    'url': href if href.startswith('http') else f"https://www.espn.com{href}",
                                    'source': 'ESPN'
                                })
                    
                    return articles[:5]  # Return top 5 results
                    
        except Exception as e:
            logger.error(f"Error searching ESPN: {e}")
            return None
    
    async def _search_nfl_com(self, query: str) -> Optional[List[Dict]]:
        """Search NFL.com for news"""
        try:
            url = "https://www.nfl.com/news"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    articles = []
                    # Look for article links
                    for article in soup.find_all('a', href=True):
                        href = article.get('href', '')
                        text = article.get_text(strip=True)
                        
                        # Filter for relevant articles
                        if any(keyword in text.lower() for keyword in query.lower().split()):
                            if 'nfl.com' in href and '/news/' in href:
                                articles.append({
                                    'title': text,
                                    'url': href if href.startswith('http') else f"https://www.nfl.com{href}",
                                    'source': 'NFL.com'
                                })
                    
                    return articles[:5]  # Return top 5 results
                    
        except Exception as e:
            logger.error(f"Error searching NFL.com: {e}")
            return None
    
    async def _search_bleacher_report(self, query: str) -> Optional[List[Dict]]:
        """Search Bleacher Report for NFL news"""
        try:
            url = "https://bleacherreport.com/nfl"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    articles = []
                    # Look for article links
                    for article in soup.find_all('a', href=True):
                        href = article.get('href', '')
                        text = article.get_text(strip=True)
                        
                        # Filter for relevant articles
                        if any(keyword in text.lower() for keyword in query.lower().split()):
                            if 'bleacherreport.com' in href and '/articles/' in href:
                                articles.append({
                                    'title': text,
                                    'url': href if href.startswith('http') else f"https://bleacherreport.com{href}",
                                    'source': 'Bleacher Report'
                                })
                    
                    return articles[:5]  # Return top 5 results
                    
        except Exception as e:
            logger.error(f"Error searching Bleacher Report: {e}")
            return None
    
    async def get_current_nfl_news(self) -> Dict:
        """Get current NFL news headlines"""
        return await self.search_nfl_news("NFL news today")
    
    async def get_coaching_changes(self) -> Dict:
        """Get latest coaching changes"""
        return await self.search_nfl_news("coaching changes offensive coordinator defensive coordinator head coach")
    
    async def get_injury_updates(self) -> Dict:
        """Get latest injury updates"""
        return await self.search_nfl_news("injury updates player status")
    
    async def get_trade_news(self) -> Dict:
        """Get latest trade news"""
        return await self.search_nfl_news("trades signings free agency")

async def search_current_nfl_info(query: str) -> str:
    """Search for current NFL information and return formatted results"""
    async with WebSearchTool() as search_tool:
        results = await search_tool.search_nfl_news(query)
        
        if not results['success']:
            return f"❌ Error searching for current information: {results.get('error', 'Unknown error')}"
        
        if not results['sources']:
            return "❌ No current information found for your query."
        
        # Format the results
        response = f"🔍 **Current NFL Information for: {query}**\n\n"
        response += f"*Last updated: {results['timestamp']}*\n\n"
        
        for source, articles in results['sources'].items():
            if articles:
                response += f"**{source}:**\n"
                for i, article in enumerate(articles[:3], 1):  # Top 3 per source
                    response += f"{i}. [{article['title']}]({article['url']})\n"
                response += "\n"
        
        response += "\n*Note: For the most up-to-date information, please visit the official sources linked above.*"
        
        return response

# Convenience functions for common searches
async def get_latest_coaching_changes() -> str:
    """Get the latest coaching changes"""
    return await search_current_nfl_info("coaching changes offensive coordinator defensive coordinator head coach 2024 2025")

async def get_latest_injuries() -> str:
    """Get the latest injury updates"""
    return await search_current_nfl_info("injury updates player status 2024 2025")

async def get_latest_trades() -> str:
    """Get the latest trade news"""
    return await search_current_nfl_info("trades signings free agency 2024 2025")

async def get_current_nfl_news() -> str:
    """Get current NFL news"""
    return await search_current_nfl_info("NFL news today current events")