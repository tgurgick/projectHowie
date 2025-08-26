"""
Howie CLI - Claude-like Fantasy Football AI Assistant
"""

__version__ = "2.0.0"
__author__ = "Trevor Gurgick"

from .core.agent import HowieAgent
from .core.context import ConversationContext
from .tools.registry import ToolRegistry

__all__ = [
    "HowieAgent",
    "ConversationContext", 
    "ToolRegistry",
]