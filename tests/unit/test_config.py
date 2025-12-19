import pytest
import os
from agentkit.config.settings import AgentKitSettings


class TestAgentKitSettings:
    def test_default_model(self, monkeypatch):
        """Test default model is deepseek-chat"""
        monkeypatch.delenv("AGENTKIT_MODEL", raising=False)
        settings = AgentKitSettings(anthropic_api_key="test-key")
        assert settings.agentkit_model == "deepseek-chat"

    def test_model_from_env(self, monkeypatch):
        """Test model can be set from environment"""
        monkeypatch.setenv("AGENTKIT_MODEL", "claude-sonnet-4-5")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        settings = AgentKitSettings()
        assert settings.agentkit_model == "claude-sonnet-4-5"

    def test_is_deepseek_detection(self, monkeypatch):
        """Test DeepSeek API detection"""
        monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.deepseek.com")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        settings = AgentKitSettings()
        assert settings.is_deepseek is True
        assert settings.is_claude is False
        assert settings.is_openai is False

    def test_is_claude_detection(self, monkeypatch):
        """Test Claude API detection (no base URL)"""
        monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        settings = AgentKitSettings()
        assert settings.is_claude is True
        assert settings.is_deepseek is False
        assert settings.is_openai is False
