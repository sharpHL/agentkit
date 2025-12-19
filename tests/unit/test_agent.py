import pytest
from unittest.mock import patch, AsyncMock
from agentkit.core.agent import Agent
from claude_agent_sdk import AssistantMessage, TextBlock


class TestAgent(Agent):
    """Test agent implementation"""
    async def process_response(self, message: AssistantMessage) -> str:
        return "".join(
            block.text for block in message.content
            if isinstance(block, TextBlock)
        )


@pytest.mark.unit
class TestAgentCore:
    def test_agent_initialization(self, mock_env):
        """Test agent initializes with correct config"""
        agent = TestAgent(
            model="custom-model",
            system_prompt="Test prompt",
            tools=["Read", "Write"]
        )

        assert agent.model == "custom-model"
        assert agent.options.system_prompt == "Test prompt"
        assert "Read" in agent.options.allowed_tools
        assert "Write" in agent.options.allowed_tools

    def test_agent_default_model(self, mock_env):
        """Test agent uses default model from env"""
        agent = TestAgent()
        assert agent.model == "test-model"

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, mock_env, mock_sdk_client):
        """Test agent start/stop lifecycle"""
        agent = TestAgent()

        with patch('agentkit.core.agent.ClaudeSDKClient', return_value=mock_sdk_client):
            await agent.start()
            assert agent._conversation_active is True
            assert agent.client is not None

            await agent.stop()
            assert agent._conversation_active is False

    @pytest.mark.asyncio
    async def test_agent_run(self, mock_env, mock_sdk_client):
        """Test agent.run() executes correctly"""
        agent = TestAgent()

        with patch('agentkit.core.agent.ClaudeSDKClient', return_value=mock_sdk_client):
            result = await agent.run("Test prompt")

            mock_sdk_client.connect.assert_called_once()
            mock_sdk_client.query.assert_called_once_with("Test prompt")
            mock_sdk_client.disconnect.assert_called_once()

            assert result == "Test response"

    @pytest.mark.asyncio
    async def test_agent_context_manager(self, mock_env, mock_sdk_client):
        """Test agent works as async context manager"""
        agent = TestAgent()

        with patch('agentkit.core.agent.ClaudeSDKClient', return_value=mock_sdk_client):
            async with agent:
                assert agent._conversation_active is True

            mock_sdk_client.disconnect.assert_called_once()
