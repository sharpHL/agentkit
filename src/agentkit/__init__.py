"""
AgentKit - Universal agent development framework

Built on Claude Agents SDK, supporting DeepSeek, Claude, and OpenAI models.
"""

from .core.agent import Agent, configure_deepseek
from .tools.registry import ToolCollection
from .config.settings import AgentKitSettings

__version__ = "0.1.0"
__all__ = ["Agent", "ToolCollection", "AgentKitSettings", "configure_deepseek"]
