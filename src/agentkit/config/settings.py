from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AgentKitSettings(BaseSettings):
    """
    Global configuration for agentkit

    Reads from environment variables and .env file
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Model settings
    agentkit_model: str = "deepseek-chat"

    # API settings
    anthropic_api_base: Optional[str] = None
    anthropic_api_key: str

    # Agent settings
    default_tools: list[str] = ["Read", "Write", "Bash", "Glob", "Grep"]
    permission_mode: str = "acceptEdits"
    max_turns: Optional[int] = None

    @property
    def is_deepseek(self) -> bool:
        """Check if using DeepSeek API"""
        return self.anthropic_api_base is not None and "deepseek" in self.anthropic_api_base

    @property
    def is_openai(self) -> bool:
        """Check if using OpenAI API"""
        return self.anthropic_api_base is not None and "openai" in self.anthropic_api_base

    @property
    def is_claude(self) -> bool:
        """Check if using Claude API (default)"""
        return self.anthropic_api_base is None


# Singleton instance (will use env vars when available)
try:
    settings = AgentKitSettings()
except Exception:
    settings = None  # Will be initialized when env vars are set
