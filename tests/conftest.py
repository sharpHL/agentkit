import pytest
from unittest.mock import AsyncMock, MagicMock
from claude_agent_sdk import AssistantMessage, TextBlock


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("AGENTKIT_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://test.api.com")


@pytest.fixture
def mock_llm_response():
    """Mock LLM response generator"""
    def create_response(text: str):
        return AssistantMessage(
            content=[TextBlock(text=text)],
            model="test-model"
        )
    return create_response


class MockAsyncIterator:
    """Helper for creating async iterators in tests"""
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_sdk_client(mock_llm_response):
    """Mock ClaudeSDKClient for testing without API calls"""
    from unittest.mock import MagicMock

    client = MagicMock()

    # Set receive_response to return an async iterator directly (not an AsyncMock)
    client.receive_response = MagicMock(return_value=MockAsyncIterator([
        mock_llm_response("Test response")
    ]))
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.query = AsyncMock()

    return client
