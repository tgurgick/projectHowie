"""
Enhanced Intelligence System for Draft Analysis
Provides targeted, parallelized intelligence gathering for fantasy-relevant players
"""

import asyncio
import sqlite3
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import random
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel

console = Console()


@dataclass
class PlayerIntelligence:
    """Individual player intelligence data"""
    player_name: str
    team: str
    position: str
    is_projected_starter: Optional[bool] = None
    starter_confidence: Optional[float] = None
    depth_chart_position: Optional[int] = None
    injury_risk_level: Optional[str] = None  # LOW, MEDIUM, HIGH
    injury_details: Optional[str] = None
    current_status: Optional[str] = None  # HEALTHY, QUESTIONABLE, INJURED
    snap_share_projection: Optional[float] = None
    target_share_projection: Optional[float] = None
    usage_notes: Optional[str] = None
    last_updated: str = None
    confidence_score: float = 0.7


@dataclass
class TeamPositionIntelligence:
    """Team position group intelligence"""
    team: str
    position: str
    players: List[PlayerIntelligence]
    depth_chart: List[str]  # Player names in order
    coaching_philosophy: str
    scheme_notes: str
    last_updated: str


class EnhancedIntelligenceGatherer:
    """Enhanced intelligence system with parallel processing and targeted searches"""
    
    def __init__(self, db_path: str = None):
        if db_path:
            self.db_path = db_path
        else:
            from .database import DraftDatabaseConnector
            self.db_connector = DraftDatabaseConnector()
            self.db_path = self.db_connector.db_path
    
    async def gather_all_team_intelligence(self, agent, max_concurrent: int = 3) -> Dict[str, Dict[str, TeamPositionIntelligence]]:
        """Gather intelligence for all teams in parallel with rate limit management"""
        
        # Get all teams and their fantasy-relevant players
        team_player_mapping = self._get_fantasy_relevant_players()
        
        console.print(f"[bright_green]🚀 Starting enhanced intelligence gathering[/bright_green]")
        console.print(f"[dim]Teams: {len(team_player_mapping)}, Max concurrent: {max_concurrent}[/dim]")
        console.print(f"[dim]Enhanced with retry logic and rate limit handling[/dim]")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all team/position combinations
        tasks = []
        total_tasks = 0
        
        for team, positions in team_player_mapping.items():
            for position, players in positions.items():
                if players:  # Only process if we have players
                    task = self._gather_team_position_with_semaphore(
                        semaphore, agent, team, position, players
                    )
                    tasks.append(task)
                    total_tasks += 1
        
        # Run all tasks with progress tracking and rate limit management
        results = {}
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        with Progress() as progress:
            task_progress = progress.add_task(
                "[bright_green]Gathering intelligence...", 
                total=total_tasks
            )
            
            completed_tasks = 0
            failed_tasks = 0
            
            for coro in asyncio.as_completed(tasks):
                try:
                    team_intel = await coro
                    if team_intel:
                        team, position, intel_data = team_intel
                        if team not in results:
                            results[team] = {}
                        results[team][position] = intel_data
                        
                        completed_tasks += 1
                        consecutive_failures = 0  # Reset failure counter
                        console.print(f"[dim]✓ {team} {position.upper()} ({completed_tasks}/{total_tasks})[/dim]")
                    else:
                        failed_tasks += 1
                        consecutive_failures += 1
                        console.print(f"[yellow]⚠ Failed task ({failed_tasks} total failures)[/yellow]")
                        
                except Exception as e:
                    failed_tasks += 1
                    consecutive_failures += 1
                    console.print(f"[red]✗ Task failed: {str(e)[:100]}[/red]")
                
                # Check for persistent rate limiting
                if consecutive_failures >= max_consecutive_failures:
                    console.print(f"[red]⚠️  {consecutive_failures} consecutive failures detected[/red]")
                    console.print(f"[yellow]Pausing for 30 seconds to handle persistent rate limits...[/yellow]")
                    await asyncio.sleep(30)
                    consecutive_failures = 0
                    console.print(f"[green]Resuming intelligence gathering...[/green]")
                
                progress.update(task_progress, advance=1)
        
        success_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        console.print(f"[bright_green]✅ Intelligence gathering complete![/bright_green]")
        console.print(f"[bright_green]📊 Success: {completed_tasks}/{total_tasks} ({success_rate:.1f}%)[/bright_green]")
        if failed_tasks > 0:
            console.print(f"[yellow]⚠️  Failed: {failed_tasks} (will retry on next run)[/yellow]")
        
        return results
    
    async def gather_single_team_intelligence(self, agent, team: str) -> Dict[str, TeamPositionIntelligence]:
        """Gather intelligence for a single team (all positions)"""
        
        # Get players for this specific team
        team_player_mapping = self._get_fantasy_relevant_players()
        
        if team not in team_player_mapping:
            console.print(f"[red]Team '{team}' not found in player mapping[/red]")
            return {}
        
        positions = team_player_mapping[team]
        console.print(f"[bright_green]🎯 Analyzing {team} - {len(positions)} positions[/bright_green]")
        
        results = {}
        
        for position, players in positions.items():
            if players:  # Only process if we have players
                console.print(f"[dim]Analyzing {team} {position.upper()} - {players}[/dim]")
                
                try:
                    intel_data = await self._gather_targeted_intelligence(agent, team, position, players)
                    if intel_data:
                        results[position] = intel_data
                        console.print(f"[bright_green]✓ {team} {position.upper()} complete[/bright_green]")
                    else:
                        console.print(f"[yellow]⚠ {team} {position.upper()} failed[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]✗ {team} {position.upper()} error: {str(e)[:100]}[/red]")
        
        success_count = len(results)
        total_count = len([pos for pos, players in positions.items() if players])
        console.print(f"[bright_green]📊 {team} complete: {success_count}/{total_count} positions[/bright_green]")
        
        return results
    
    async def _gather_team_position_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        agent, 
        team: str, 
        position: str, 
        players: List[str]
    ) -> Optional[Tuple[str, str, TeamPositionIntelligence]]:
        """Gather intelligence for a team/position with concurrency control"""
        
        async with semaphore:
            try:
                intel_data = await self._gather_targeted_intelligence(agent, team, position, players)
                return (team, position, intel_data) if intel_data else None
            except Exception as e:
                console.print(f"[red]Error gathering {team} {position}: {e}[/red]")
                return None
    
    def _get_fantasy_relevant_players(self) -> Dict[str, Dict[str, List[str]]]:
        """Get fantasy-relevant players from rankings/projections, organized by team and position"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get top fantasy players by team and position
        # Focus on players likely to be drafted (top ~200 overall)
        query = """
        SELECT DISTINCT 
            pp.team_name,
            pp.position,
            pp.player_name,
            pp.fantasy_points,
            COALESCE(ad.adp_overall, 999) as adp
        FROM player_projections pp
        LEFT JOIN adp_data ad ON LOWER(pp.player_name) = LOWER(ad.player_name) 
            AND ad.season = pp.season
        WHERE pp.season = 2025 
            AND pp.projection_type = 'preseason'
            AND pp.position IN ('qb', 'rb', 'wr', 'te', 'k', 'dst')
            AND (
                (pp.position = 'qb' AND pp.fantasy_points > 5) OR
                (pp.position IN ('rb', 'wr', 'te') AND pp.fantasy_points > 20) OR
                (pp.position IN ('k', 'dst') AND pp.fantasy_points > 80)
            )
        ORDER BY pp.team_name, pp.position, pp.fantasy_points DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Organize by team and position
        team_mapping = {}
        
        for team, position, player_name, fantasy_points, adp in results:
            if not team or team == 'FA':
                continue
                
            if team not in team_mapping:
                team_mapping[team] = {}
            
            if position not in team_mapping[team]:
                team_mapping[team][position] = []
            
            # Limit to top fantasy players per position per team
            # QB: Top 2, RB: Top 5, WR: Top 5, TE: Top 3, K: Top 1, DST: Top 1
            limits = {'qb': 2, 'rb': 5, 'wr': 5, 'te': 3, 'k': 1, 'dst': 1}
            
            if len(team_mapping[team][position]) < limits.get(position, 3):
                team_mapping[team][position].append(player_name)
        
        conn.close()
        return team_mapping
    
    async def _gather_targeted_intelligence(
        self, 
        agent, 
        team: str, 
        position: str, 
        players: List[str]
    ) -> Optional[TeamPositionIntelligence]:
        """Gather targeted intelligence for specific players on a team with retry logic"""
        
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Force Claude Sonnet 4 for this analysis
                original_model = getattr(agent.model_manager, 'current_model', None)
                agent.model_manager.set_model('claude-sonnet-4')
                
                # Create structured JSON request for each player
                player_intel_requests = []
                for player_name in players:
                    player_intel_requests.append({
                        "player_name": player_name,
                        "team": team,
                        "position": position.upper(),
                        "analysis_needed": {
                            "starter_status": "Is this player projected to be a starter? Include confidence level.",
                            "injury_status": "Any recent injury concerns or health issues?",
                            "usage_projection": "Expected snap share, targets, or carries for 2025?",
                            "depth_chart_position": "Where does this player rank on the depth chart?"
                        }
                    })
                
                # Create the prompt with JSON structure
                prompt = f"""Analyze the {team} {position.upper()} position group for the 2025 NFL season. 

SPECIFIC PLAYERS TO ANALYZE: {', '.join(players)}

Please provide a JSON response with the following structure for each player:

{{
  "team_analysis": {{
    "team": "{team}",
    "position": "{position.upper()}",
    "coaching_philosophy": "How does {team} utilize {position.upper()}s?",
    "scheme_notes": "Key scheme details affecting {position.upper()} usage"
  }},
  "players": [
    {{
      "player_name": "Player Name",
      "is_projected_starter": true/false,
      "starter_confidence": 0.0-1.0,
      "depth_chart_position": 1-5,
      "injury_risk_level": "LOW/MEDIUM/HIGH",
      "injury_details": "Specific injury info or 'No concerns'",
      "current_status": "HEALTHY/QUESTIONABLE/INJURED",
      "snap_share_projection": 0.0-1.0,
      "target_share_projection": 0.0-1.0 (for skill positions),
      "usage_notes": "Key usage expectations"
    }}
  ],
  "depth_chart": ["Player1", "Player2", "Player3"]
}}

Focus on:
1. STARTER STATUS: Clear yes/no with confidence level
2. INJURY STATUS: Recent concerns, training camp reports, health outlook  
3. USAGE PROJECTIONS: Realistic expectations for usage/performance
4. DEPTH CHART: Clear ranking of players

Special considerations by position:
- QB/RB/WR/TE: Snap share, target share, usage patterns
- K: Field goal accuracy, extra point reliability, team red zone efficiency
- DST: Defensive rankings, takeaway potential, special teams impact, coaching staff changes, defensive coordinator changes, key injuries to defensive starters, scheme changes from previous season

Use current web search to find the most recent information about these specific players."""
                
                # Get the response with retry logic
                response = await self._make_request_with_retry(
                    agent, prompt, attempt, max_retries
                )
                
                # Restore original model
                if original_model:
                    agent.model_manager.set_model(original_model)
                
                # Parse the JSON response
                intel_data = self._parse_intelligence_response(response, team, position, players)
                return intel_data
                
            except Exception as e:
                if attempt < max_retries:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                    console.print(f"[yellow]Retry {attempt + 1}/{max_retries} for {team} {position} in {delay:.1f}s: {str(e)[:50]}[/yellow]")
                    await asyncio.sleep(delay)
                    continue
                else:
                    console.print(f"[red]Failed after {max_retries} retries for {team} {position}: {e}[/red]")
                    return None
        
        return None
    
    async def _make_request_with_retry(
        self, 
        agent, 
        prompt: str, 
        attempt: int, 
        max_retries: int
    ) -> str:
        """Make a request with specific error handling for rate limits"""
        
        try:
            response = await agent.model_manager.complete(
                messages=[{"role": "user", "content": prompt}],
                task_type="research",
                temperature=0.1,
                max_tokens=2500
            )
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit errors
            if any(term in error_msg for term in ['rate limit', '429', 'too many requests']):
                if attempt < max_retries:
                    # Longer delay for rate limits
                    delay = 5 + (attempt * 3) + random.uniform(1, 3)
                    console.print(f"[yellow]Rate limit hit, waiting {delay:.1f}s before retry[/yellow]")
                    await asyncio.sleep(delay)
                raise Exception(f"Rate limit exceeded (attempt {attempt + 1})")
            
            # Check for other API errors
            elif any(term in error_msg for term in ['api', 'connection', 'timeout']):
                if attempt < max_retries:
                    delay = 3 + random.uniform(0.5, 2)
                    await asyncio.sleep(delay)
                raise Exception(f"API error: {str(e)[:100]}")
            
            # Re-raise other exceptions
            else:
                raise e
    
    def _parse_intelligence_response(
        self, 
        response: str, 
        team: str, 
        position: str, 
        expected_players: List[str]
    ) -> Optional[TeamPositionIntelligence]:
        """Parse the AI response into structured intelligence data"""
        
        try:
            # Extract JSON from response
            response_text = response
            if hasattr(response, 'content'):
                response_text = response.content
            
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                # Fallback: create structured data from text
                return self._parse_text_response(response_text, team, position, expected_players)
            
            json_text = response_text[start_idx:end_idx]
            data = json.loads(json_text)
            
            # Convert to our data structures
            player_intelligences = []
            
            # Special handling for DST position
            if position == 'dst' and not any(p.get('player_name', '').endswith('DST') for p in data.get('players', [])):
                # AI analyzed individual defensive players instead of team DST
                # Create a team DST entry from expected players
                for expected_player in expected_players:
                    if expected_player.endswith('DST'):
                        dst_data = {
                            'player_name': expected_player,
                            'is_projected_starter': True,
                            'starter_confidence': 1.0,
                            'injury_risk': 'LOW',
                            'injury_details': 'No significant injuries reported',
                            'usage_notes': 'Primary team defense unit for fantasy purposes'
                        }
                        
                        # Extract defensive analysis from team analysis if available
                        team_analysis = data.get('team_analysis', {})
                        coaching_notes = team_analysis.get('coaching_philosophy', '') or team_analysis.get('coaching_style', '')
                        scheme_notes = team_analysis.get('scheme_notes', '')
                        injury_summary = team_analysis.get('injury_updates', '')
                        
                        # Create comprehensive usage notes for DST
                        usage_parts = ['Primary team defense unit for fantasy purposes']
                        if coaching_notes:
                            usage_parts.append(f"Coaching: {coaching_notes}")
                        if scheme_notes:
                            usage_parts.append(f"Scheme: {scheme_notes}")
                        if injury_summary:
                            usage_parts.append(f"Injuries: {injury_summary}")
                        
                        comprehensive_usage = '. '.join(usage_parts)
                        
                        player_intel = PlayerIntelligence(
                            player_name=expected_player,
                            team=team,
                            position=position,
                            is_projected_starter=True,
                            starter_confidence=1.0,
                            injury_risk_level='LOW' if not injury_summary or 'no significant' in injury_summary.lower() else 'MEDIUM',
                            injury_details=injury_summary or 'No significant injuries reported',
                            usage_notes=comprehensive_usage,
                            last_updated=datetime.now().isoformat(),
                            confidence_score=0.9
                        )
                        player_intelligences.append(player_intel)
                        break
            else:
                # Normal player processing
                for player_data in data.get('players', []):
                    player_name = player_data.get('player_name', '').strip()
                    
                    # Skip invalid player names
                    if not player_name or len(player_name) < 3 or player_name.lower() in ['fill in', 'player name', 'tbd', 'unknown']:
                        continue
                    
                    # Only include players we expected to analyze
                    if player_name not in expected_players:
                        # Try fuzzy matching for minor variations
                        name_matches = [p for p in expected_players if player_name.lower() in p.lower() or p.lower() in player_name.lower()]
                        if name_matches:
                            player_name = name_matches[0]  # Use the expected name
                        else:
                            continue  # Skip unexpected players
                
                    player_intel = PlayerIntelligence(
                        player_name=player_name,
                        team=team,
                        position=position,
                        is_projected_starter=player_data.get('is_projected_starter'),
                        starter_confidence=player_data.get('starter_confidence'),
                        depth_chart_position=player_data.get('depth_chart_position'),
                        injury_risk_level=player_data.get('injury_risk_level'),
                        injury_details=player_data.get('injury_details'),
                        current_status=player_data.get('current_status'),
                        snap_share_projection=player_data.get('snap_share_projection'),
                        target_share_projection=player_data.get('target_share_projection'),
                        usage_notes=player_data.get('usage_notes'),
                        last_updated=datetime.now().isoformat(),
                        confidence_score=0.8  # High confidence for structured response
                    )
                    player_intelligences.append(player_intel)
            
            team_analysis = data.get('team_analysis', {})
            
            team_intel = TeamPositionIntelligence(
                team=team,
                position=position,
                players=player_intelligences,
                depth_chart=data.get('depth_chart', expected_players),
                coaching_philosophy=team_analysis.get('coaching_philosophy', ''),
                scheme_notes=team_analysis.get('scheme_notes', ''),
                last_updated=datetime.now().isoformat()
            )
            
            return team_intel
            
        except json.JSONDecodeError:
            # Fallback to text parsing
            return self._parse_text_response(response_text, team, position, expected_players)
        except Exception as e:
            console.print(f"[yellow]Warning: Error parsing {team} {position} response: {e}[/yellow]")
            return None
    
    def _parse_text_response(
        self, 
        response_text: str, 
        team: str, 
        position: str, 
        expected_players: List[str]
    ) -> Optional[TeamPositionIntelligence]:
        """Fallback text parsing when JSON parsing fails"""
        
        try:
            player_intelligences = []
            
            # Simple text parsing for key information
            for player_name in expected_players:
                # Look for player mentions in the text
                player_intel = PlayerIntelligence(
                    player_name=player_name,
                    team=team,
                    position=position,
                    last_updated=datetime.now().isoformat(),
                    confidence_score=0.5  # Lower confidence for text parsing
                )
                
                # Try to extract starter status
                if 'starter' in response_text.lower() and player_name.lower() in response_text.lower():
                    player_intel.is_projected_starter = True
                    player_intel.starter_confidence = 0.7
                
                # Try to extract injury info
                if any(word in response_text.lower() for word in ['injury', 'hurt', 'questionable', 'injured']):
                    if player_name.lower() in response_text.lower():
                        player_intel.injury_risk_level = 'MEDIUM'
                        player_intel.injury_details = 'Potential injury concern mentioned'
                
                player_intelligences.append(player_intel)
            
            team_intel = TeamPositionIntelligence(
                team=team,
                position=position,
                players=player_intelligences,
                depth_chart=expected_players,
                coaching_philosophy='Text parsing - limited detail',
                scheme_notes='Text parsing - limited detail',
                last_updated=datetime.now().isoformat()
            )
            
            return team_intel
            
        except Exception as e:
            console.print(f"[red]Error in text parsing for {team} {position}: {e}[/red]")
            return None
    
    async def save_intelligence_to_database(self, intelligence_data: Dict[str, Dict[str, TeamPositionIntelligence]]):
        """Save the gathered intelligence to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure tables exist
        await self._create_enhanced_intelligence_tables(cursor)
        
        updates_made = 0
        
        for team, positions in intelligence_data.items():
            for position, team_intel in positions.items():
                
                # Save each player's intelligence
                for player in team_intel.players:
                    cursor.execute("""
                        INSERT OR REPLACE INTO player_draft_intelligence (
                            player_name, team, position, season,
                            is_projected_starter, starter_confidence, depth_chart_position,
                            injury_risk_level, injury_details, current_injury_status,
                            snap_share_projection, target_share_projection, usage_notes,
                            last_updated, confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player.player_name, player.team, player.position, 2025,
                        player.is_projected_starter, player.starter_confidence, player.depth_chart_position,
                        player.injury_risk_level, player.injury_details, player.current_status,
                        player.snap_share_projection, player.target_share_projection, player.usage_notes,
                        player.last_updated, player.confidence_score
                    ))
                    updates_made += 1
                
                # Save team position summary
                cursor.execute("""
                    INSERT OR REPLACE INTO team_position_intelligence (
                        team, position, season, last_updated,
                        intelligence_summary, coaching_style,
                        confidence_score, fact_check_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    team, position, 2025, team_intel.last_updated,
                    f"Depth Chart: {', '.join(team_intel.depth_chart[:3])}. {team_intel.coaching_philosophy}",
                    team_intel.scheme_notes,
                    0.8, 'enhanced_analysis'
                ))
        
        conn.commit()
        conn.close()
        
        console.print(f"[bright_green]💾 Saved {updates_made} player intelligence records to database[/bright_green]")
        return updates_made
    
    async def _create_enhanced_intelligence_tables(self, cursor):
        """Create enhanced intelligence tables with additional fields"""
        
        # Enhanced player intelligence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_draft_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                
                -- Starter Status
                is_projected_starter BOOLEAN DEFAULT NULL,
                starter_confidence REAL DEFAULT NULL,
                depth_chart_position INTEGER DEFAULT NULL,
                
                -- Injury Risk Assessment  
                injury_risk_level TEXT DEFAULT NULL,
                injury_details TEXT DEFAULT NULL,
                current_injury_status TEXT DEFAULT NULL,
                
                -- Usage Projections
                snap_share_projection REAL DEFAULT NULL,
                target_share_projection REAL DEFAULT NULL,
                usage_notes TEXT DEFAULT NULL,
                
                -- Metadata
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT NULL,
                
                UNIQUE(player_name, team, season)
            )
        """)
        
        # Team position intelligence table (existing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_position_intelligence (
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                last_updated TEXT,
                intelligence_summary TEXT,
                key_players TEXT,
                usage_notes TEXT,
                coaching_style TEXT,
                injury_updates TEXT,
                recent_changes TEXT,
                fact_check_status TEXT DEFAULT 'pending',
                fact_check_notes TEXT,
                confidence_score REAL,
                PRIMARY KEY (team, position, season)
            )
        """)
    
    def get_player_intelligence_summary(self, player_name: str, team: str = None) -> Optional[Dict]:
        """Get enhanced intelligence summary for a specific player"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if team:
            cursor.execute("""
                SELECT * FROM player_draft_intelligence 
                WHERE LOWER(player_name) = LOWER(?) AND team = ? AND season = 2025
            """, (player_name, team))
        else:
            cursor.execute("""
                SELECT * FROM player_draft_intelligence 
                WHERE LOWER(player_name) = LOWER(?) AND season = 2025
            """, (player_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Convert to dictionary
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        
        return None
