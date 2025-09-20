"""
Command registry shared by CLI chat and the TUI.
This allows a single source of truth for palette and slash suggestions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Command:
    id: str
    title: str
    group: str
    keywords: Optional[List[str]] = None


# Comprehensive command registry
REGISTRY: List[Command] = [
    # AI & System Commands
    Command(id="model", title="Model: info/switch/config/save", group="AI", keywords=["default", "provider", "list"]),
    Command(id="agent", title="Agent: spawn/list/stop", group="AI", keywords=["workers", "tools"]),
    Command(id="cost", title="Cost: budgets and usage", group="AI", keywords=["pricing", "limits"]),
    Command(id="logs", title="Logs: recent system events", group="System", keywords=["events", "errors"]),
    Command(id="help", title="Help: show commands and tips", group="System", keywords=["docs", "usage", "?"]),
    Command(id="quit", title="Quit: exit Howie", group="System", keywords=["exit", "bye"]),
    
    # Data Commands
    Command(id="update", title="Update: ADP/Intel/SoS/Projections", group="Data", keywords=["adp", "intel", "pff", "sos"]),
    Command(id="adp", title="ADP: overall or format-specific", group="Data", keywords=["10", "12", "overall", "rankings"]),
    Command(id="tiers", title="Tiers: positional tier analysis", group="Data", keywords=["position", "drops", "value"]),
    Command(id="intel", title="Intel: team position intelligence", group="Data", keywords=["team", "wr", "rb", "qb"]),
    Command(id="player", title="Player: comprehensive analysis", group="Data", keywords=["search", "report", "analysis"]),
    
    # Main Draft Commands
    Command(id="draft", title="Draft: simulation and analysis", group="Draft", keywords=["tiers", "rounds", "value"]),
    Command(id="draft/help", title="Draft Help: all draft commands", group="Draft", keywords=["commands", "usage"]),
    Command(id="draft/test", title="Draft Test: database connection", group="Draft", keywords=["connection", "players", "data"]),
    Command(id="draft/quick", title="Draft Quick: fast analysis", group="Draft", keywords=["default", "simple", "fast"]),
    
    # Configuration Commands
    Command(id="draft/config", title="Draft Config: interactive setup", group="Configuration", keywords=["league", "teams", "scoring", "keeper"]),
    Command(id="draft/config/position", title="Set Draft Position", group="Configuration", keywords=["pick", "slot", "order"]),
    Command(id="draft/config/teams", title="Set Number of Teams", group="Configuration", keywords=["league", "size", "12", "10"]),
    Command(id="draft/config/scoring", title="Set Scoring Type", group="Configuration", keywords=["ppr", "standard", "half"]),
    
    # Monte Carlo Simulation Commands
    Command(id="draft/monte", title="Monte Carlo Simulation", group="Simulation", keywords=["scenarios", "outcomes", "probability"]),
    Command(id="draft/monte/enhanced", title="Enhanced Monte Carlo", group="Simulation", keywords=["distributions", "variance", "advanced"]),
    Command(id="draft/monte/config", title="Select MC Configuration", group="Simulation", keywords=["saved", "new", "default"]),
    Command(id="draft/simulate", title="Advanced Draft Simulation", group="Simulation", keywords=["ai", "opponents", "realistic"]),
    
    # Strategy Commands
    Command(id="draft/strategy", title="Strategy Management", group="Strategy", keywords=["generate", "load", "tree", "optimal"]),
    Command(id="draft/strategy/generate", title="Generate New Strategy", group="Strategy", keywords=["tree", "search", "optimal", "create"]),
    Command(id="draft/strategy/recommendations", title="Round Recommendations", group="Strategy", keywords=["16", "rounds", "options", "picks"]),
    Command(id="draft/strategy/positional", title="Positional Strategy", group="Strategy", keywords=["primary", "backup", "contingency"]),
    Command(id="draft/strategy/current", title="Show Current Strategy", group="Strategy", keywords=["details", "view", "loaded"]),
    
    # Analysis Commands
    Command(id="draft/compare", title="Player Comparison", group="Analysis", keywords=["side-by-side", "stats", "value", "vs"]),
    Command(id="draft/view", title="Results Viewer", group="Analysis", keywords=["simulation", "outcomes", "availability"]),
    Command(id="draft/view/current", title="Show Last Simulation", group="Analysis", keywords=["recent", "results", "details"]),
    Command(id="draft/view/availability", title="Player Availability Analysis", group="Analysis", keywords=["rounds", "likely", "probability"]),
    Command(id="draft/analyze", title="Full Draft Analysis", group="Analysis", keywords=["comprehensive", "deep", "complete"]),
    
    # Rapid Stats Commands
    Command(id="wr/adp", title="WR ADP: Top WRs by ADP", group="Stats", keywords=["rankings", "receivers", "wide"]),
    Command(id="qb/td", title="QB TD: Top QBs by total TDs", group="Stats", keywords=["touchdowns", "quarterbacks"]),
    Command(id="rb/yards", title="RB Yards: Top RBs by yards", group="Stats", keywords=["rushing", "running backs"]),
    Command(id="te/rec", title="TE Receptions", group="Stats", keywords=["tight ends", "volume", "targets"]),
    Command(id="k/points", title="K Points: Top Kickers", group="Stats", keywords=["kickers", "scoring"]),
    Command(id="def/points", title="DEF Points: Top Defenses", group="Stats", keywords=["defense", "dst"]),
]
