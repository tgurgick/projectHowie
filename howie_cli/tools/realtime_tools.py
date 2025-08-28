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

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.model_manager import ModelManager
import os
from dotenv import load_dotenv

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
    # Enhance query with NFL-specific terms to avoid other sports
    enhanced_query = query.strip()
    
    # Add NFL context if not already present
    if not any(term in enhanced_query.lower() for term in ['nfl', 'football', 'qb', 'rb', 'wr', 'te', 'defense', 'offense']):
        enhanced_query = f"NFL football {enhanced_query}"
    
    # Add current season context
    if '2025' in enhanced_query or 'current' in enhanced_query.lower():
        enhanced_query += " 2025 season"
    
    async with WebSearchTool() as search_tool:
        results = await search_tool.search_nfl_news(enhanced_query)
        
        if not results['success']:
            return f"Error searching for current information: {results.get('error', 'Unknown error')}"
        
        if not results['sources']:
            return "No current NFL information found for your query."
        
        # Filter results to ensure they're NFL-related
        filtered_results = {}
        for source, articles in results['sources'].items():
            nfl_articles = []
            for article in articles:
                title = article.get('title', '').lower()
                # Filter out non-NFL content
                if any(nfl_term in title for nfl_term in ['nfl', 'football', 'qb', 'rb', 'wr', 'te', 'defense', 'offense', 'coach', 'draft', 'free agency']):
                    nfl_articles.append(article)
                elif any(baseball_term in title for baseball_term in ['mlb', 'baseball', 'pitcher', 'hitter', 'home run', 'inning']):
                    continue  # Skip baseball content
                elif any(basketball_term in title for basketball_term in ['nba', 'basketball', 'point guard', 'shooting guard']):
                    continue  # Skip basketball content
                else:
                    # If unclear, include it but mark as potentially mixed
                    nfl_articles.append(article)
            
            if nfl_articles:
                filtered_results[source] = nfl_articles[:3]  # Top 3 per source
        
        if not filtered_results:
            return "No relevant NFL information found. Please try a more specific NFL-related query."
        
        # Format the results
        response = f"**Current NFL Information for: {query}**\n\n"
        response += f"*Last updated: {results['timestamp']}*\n\n"
        
        for source, articles in filtered_results.items():
            if articles:
                response += f"**{source}:**\n"
                for i, article in enumerate(articles, 1):
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


class RealtimeSearchTool(BaseTool):
    """Tool for searching current NFL information and news"""
    
    def __init__(self):
        super().__init__()
        self.name = "realtime_search"
        self.description = "Search for current NFL news, updates, and real-time information"
        self.category = "research"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="Search query for current NFL information",
                required=True
            )
        ]
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute realtime search for NFL information using Perplexity AI"""
        try:
            query = kwargs.get('query')
            if not query or not isinstance(query, str):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="Query parameter is required and must be a string"
                )
            
            # Use Perplexity for real-time search instead of web scraping
            result = await self._search_with_perplexity(query.strip())
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                metadata={"search_method": "perplexity_api", "query": query}
            )
            
        except Exception as e:
            logger.error(f"Error in realtime search: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to search for current information: {str(e)}"
            )
    
    async def _search_with_perplexity(self, query: str) -> str:
        """Search using Perplexity Pro for real-time NFL information"""
        try:
            # Ensure .env is loaded
            load_dotenv()
            
            # Initialize model manager
            model_manager = ModelManager()
            
            # Enhance query with NFL context for better results
            enhanced_query = f"""Find current NFL information about: {query}

Focus on:
- Recent NFL news and updates
- Player moves, cuts, and signings  
- Injury reports and fantasy impact
- Team depth chart changes
- Recent game results or upcoming games

Provide specific, actionable information for fantasy football purposes. Include sources and dates when possible."""
            
            messages = [{"role": "user", "content": enhanced_query}]
            
            # Use Perplexity Pro for real-time search (should work with existing API key)
            try:
                result = await model_manager.complete(messages, model="perplexity-sonar-pro")
                if result and len(result.strip()) > 50:
                    return f"Real-time search results:\n\n{result.strip()}"
                else:
                    # Fallback to regular Perplexity if Pro gives short response
                    result = await model_manager.complete(messages, model="perplexity-sonar")
                    if result and len(result.strip()) > 50:
                        return f"Real-time search results:\n\n{result.strip()}"
            except Exception as perplexity_error:
                logger.error(f"Perplexity search failed: {perplexity_error}")
                # Continue to fallback
            
            # Fallback to Claude with guidance for real-time search
            try:
                # Use Claude to provide guidance on where to find current info
                claude_query = f"""The user is looking for current NFL information about: {query}

Since I don't have real-time internet access, provide:
1. The best current sources to check for this information
2. Specific websites, reporters, or apps to follow  
3. Key search terms they should use
4. Any relevant context about when this type of information is typically released

Be specific and actionable for fantasy football purposes."""

                claude_messages = [{"role": "user", "content": claude_query}]
                result = await model_manager.complete(claude_messages, model="claude-sonnet-4")
                
                if result:
                    setup_note = """

For optimal real-time search: Set up PERPLEXITY_API_KEY in your environment to enable direct web search capabilities."""
                    
                    return f"Current NFL Information Guide:\n\n{result.strip()}\n{setup_note}"
                    
            except Exception as claude_error:
                logger.error(f"Claude fallback failed: {claude_error}")
            
            # Final fallback with comprehensive guidance
            return f"""Searching for: {query}

Best Current Sources:
- Breaking News: @AdamSchefter, @RapSheet, @TomPelissero (Twitter/X)
- Fantasy Impact: @FantasyPros, @Rotoworld_FB
- Team News: Official team websites and beat reporters
- Comprehensive: ESPN.com/nfl/news, NFL.com/news

Setup Tip: Add PERPLEXITY_API_KEY to your environment for direct web search.

Database Alternative: Try asking about specific players/teams using our stats database."""
            
        except Exception as e:
            logger.error(f"Error in real-time search: {e}")
            return f"""Search temporarily unavailable.

Quick alternatives for "{query}":
- Check ESPN.com/nfl/news
- Follow @AdamSchefter on Twitter/X  
- Use our database tools for player stats
- Try: "Who is [player name]" or "Show me [team] stats"

Setup: Add PERPLEXITY_API_KEY for better search capabilities."""