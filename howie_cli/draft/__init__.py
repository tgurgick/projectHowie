"""
ProjectHowie Draft Simulation System

A comprehensive draft simulation and strategy optimization system that provides:
- Pre-draft analysis with round-by-round recommendations
- Tree search optimization for optimal draft strategy  
- Real-time draft assistance during live drafts
- Enhanced evaluation using SoS, starter status, and injury data

Integration with ProjectHowie's existing fantasy football database.
"""

__version__ = "1.0.0"

from .models import LeagueConfig, KeeperPlayer, Player, Roster
from .database import DraftDatabaseConnector
from .recommendation_engine import PickRecommendationEngine, PickRecommendation
from .analysis_generator import DraftAnalysisGenerator
from .draft_cli import DraftCLI

__all__ = [
    'LeagueConfig',
    'KeeperPlayer', 
    'Player',
    'Roster',
    'DraftDatabaseConnector',
    'PickRecommendationEngine',
    'PickRecommendation',
    'DraftAnalysisGenerator',
    'DraftCLI'
]
