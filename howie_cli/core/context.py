"""
Context management for maintaining conversation state
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
from pathlib import Path
from collections import deque
import hashlib


class Message(BaseModel):
    """Represents a message in conversation"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[Dict]] = None


class PlayerContext(BaseModel):
    """Context for a specific player"""
    player_name: str
    player_id: Optional[str] = None
    position: str
    team: str
    last_analysis: Optional[Dict] = None
    analysis_timestamp: Optional[datetime] = None
    cached_stats: Optional[Dict] = None
    notes: List[str] = Field(default_factory=list)


class LeagueContext(BaseModel):
    """Context for user's fantasy league"""
    league_id: Optional[str] = None
    platform: Optional[str] = None  # 'espn', 'yahoo', 'sleeper', etc.
    scoring_type: str = "ppr"  # 'ppr', 'half_ppr', 'standard'
    roster_size: int = 15
    starting_lineup: Dict[str, int] = Field(default_factory=lambda: {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1
    })
    my_roster: List[str] = Field(default_factory=list)
    league_settings: Dict[str, Any] = Field(default_factory=dict)
    opponents: List[Dict] = Field(default_factory=list)


class UserPreferences(BaseModel):
    """User preferences and settings"""
    risk_tolerance: str = "medium"  # 'conservative', 'medium', 'aggressive'
    favorite_teams: List[str] = Field(default_factory=list)
    avoided_players: List[str] = Field(default_factory=list)
    trade_preferences: Dict[str, Any] = Field(default_factory=dict)
    notification_settings: Dict[str, bool] = Field(default_factory=lambda: {
        "injuries": True,
        "trade_suggestions": True,
        "lineup_reminders": True,
        "breaking_news": True
    })
    analysis_depth: str = "detailed"  # 'quick', 'standard', 'detailed'


class ConversationContext:
    """Manages conversation context and state"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.messages: deque = deque(maxlen=100)  # Keep last 100 messages
        self.current_topic: Optional[str] = None
        self.players_discussed: Dict[str, PlayerContext] = {}
        self.league_context: Optional[LeagueContext] = None
        self.user_preferences: UserPreferences = UserPreferences()
        self.analysis_cache: Dict[str, Dict] = {}
        self.tool_history: List[Dict] = []
        self.session_start: datetime = datetime.now()
        self.last_activity: datetime = datetime.now()
        self.workspace_path: Optional[Path] = None
        self.active_files: Set[str] = set()
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the conversation"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Extract and update context from message
        self._extract_context_from_message(content)
    
    def _extract_context_from_message(self, content: str):
        """Extract relevant context from message content"""
        # Look for player names, positions, teams, etc.
        # This could be enhanced with NLP
        content_lower = content.lower()
        
        # Extract scoring type mentions
        if "ppr" in content_lower:
            if "half" in content_lower:
                self.set_scoring_type("half_ppr")
            else:
                self.set_scoring_type("ppr")
        elif "standard" in content_lower:
            self.set_scoring_type("standard")
    
    def add_player_context(self, player_name: str, **kwargs):
        """Add or update player context"""
        if player_name not in self.players_discussed:
            self.players_discussed[player_name] = PlayerContext(
                player_name=player_name,
                position=kwargs.get("position", ""),
                team=kwargs.get("team", "")
            )
        
        # Update existing context
        player_ctx = self.players_discussed[player_name]
        for key, value in kwargs.items():
            if hasattr(player_ctx, key):
                setattr(player_ctx, key, value)
    
    def set_league_context(self, **kwargs):
        """Set or update league context"""
        if not self.league_context:
            self.league_context = LeagueContext()
        
        for key, value in kwargs.items():
            if hasattr(self.league_context, key):
                setattr(self.league_context, key, value)
    
    def set_scoring_type(self, scoring_type: str):
        """Set the scoring type for the league"""
        if not self.league_context:
            self.league_context = LeagueContext()
        self.league_context.scoring_type = scoring_type
    
    def cache_analysis(self, key: str, data: Dict, ttl_minutes: int = 30):
        """Cache analysis results with TTL"""
        self.analysis_cache[key] = {
            "data": data,
            "timestamp": datetime.now(),
            "expires": datetime.now() + timedelta(minutes=ttl_minutes)
        }
    
    def get_cached_analysis(self, key: str) -> Optional[Dict]:
        """Get cached analysis if not expired"""
        if key in self.analysis_cache:
            cache_entry = self.analysis_cache[key]
            if datetime.now() < cache_entry["expires"]:
                return cache_entry["data"]
            else:
                # Remove expired entry
                del self.analysis_cache[key]
        return None
    
    def add_tool_execution(self, tool_name: str, params: Dict, result: Any):
        """Record tool execution in history"""
        self.tool_history.append({
            "tool": tool_name,
            "params": params,
            "result": str(result)[:500],  # Truncate large results
            "timestamp": datetime.now()
        })
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get n most recent messages"""
        return list(self.messages)[-n:]
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation context"""
        return {
            "session_id": self.session_id,
            "duration": str(datetime.now() - self.session_start),
            "message_count": len(self.messages),
            "players_discussed": list(self.players_discussed.keys()),
            "current_topic": self.current_topic,
            "scoring_type": self.league_context.scoring_type if self.league_context else "ppr",
            "tools_used": len(self.tool_history),
            "active_files": list(self.active_files)
        }
    
    def save_session(self, path: Optional[Path] = None):
        """Save session to disk as JSON (pickle is unsafe for untrusted files)"""
        if not path:
            path = Path.home() / ".howie" / "sessions" / f"{self.session_id}.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "session_id": self.session_id,
            "messages": list(self.messages),
            "players_discussed": {
                k: v.dict() for k, v in self.players_discussed.items()
            },
            "league_context": self.league_context.dict() if self.league_context else None,
            "user_preferences": self.user_preferences.dict(),
            "analysis_cache": self.analysis_cache,
            "tool_history": self.tool_history,
            "session_start": self.session_start.isoformat(),
            "current_topic": self.current_topic,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, default=str)

    @classmethod
    def load_session(cls, session_id: str) -> 'ConversationContext':
        """Load session from disk (JSON only — legacy .pkl files are rejected)"""
        base = Path.home() / ".howie" / "sessions"
        path = base / f"{session_id}.json"

        if not path.exists():
            if (base / f"{session_id}.pkl").exists():
                raise ValueError(
                    f"Session {session_id} is a legacy pickle file, which is no "
                    "longer loaded for safety. Start a fresh session."
                )
            raise FileNotFoundError(f"Session {session_id} not found")

        with open(path) as f:
            data = json.load(f)

        context = cls(session_id=data["session_id"])
        context.messages = deque(data["messages"], maxlen=100)
        context.players_discussed = {
            k: PlayerContext(**v) for k, v in data.get("players_discussed", {}).items()
        }
        if data.get("league_context"):
            context.league_context = LeagueContext(**data["league_context"])
        context.user_preferences = UserPreferences(**data.get("user_preferences", {}))
        context.analysis_cache = data.get("analysis_cache", {})
        context.tool_history = data.get("tool_history", [])
        context.session_start = datetime.fromisoformat(data["session_start"])
        context.current_topic = data.get("current_topic")
        
        return context
    
    def clear_expired_cache(self):
        """Clear expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.analysis_cache.items()
            if now >= entry["expires"]
        ]
        for key in expired_keys:
            del self.analysis_cache[key]
    
    def to_llm_messages(self) -> List[Dict]:
        """Convert context to LLM-compatible message format"""
        llm_messages = []
        
        # Add system context
        system_context = f"""You are Howie, an expert fantasy football AI assistant.
Current session context:
- Scoring Type: {self.league_context.scoring_type if self.league_context else 'PPR'}
- Players discussed: {', '.join(self.players_discussed.keys()) if self.players_discussed else 'None'}
- User risk tolerance: {self.user_preferences.risk_tolerance}
- Analysis depth preference: {self.user_preferences.analysis_depth}
"""
        llm_messages.append({"role": "system", "content": system_context})
        
        # Add conversation messages
        for msg in self.get_recent_messages(20):  # Last 20 messages for context
            llm_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return llm_messages