from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import os


class Agent(ABC):
    """
    High-level agent abstraction built on Claude Agents SDK

    Supports any model via environment configuration:
    - Claude models (default SDK behavior)
    - DeepSeek via ANTHROPIC_API_BASE
    - OpenAI-compatible endpoints
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[list[str], dict]] = None,
        allowed_tools: Optional[list[str]] = None,
        permission_mode: str = "acceptEdits",
        **kwargs
    ):
        """
        Initialize agent with flexible model configuration

        Args:
            model: Model name (defaults to env AGENTKIT_MODEL or "deepseek-chat")
            api_base: API base URL (defaults to env ANTHROPIC_API_BASE)
            api_key: API key (defaults to env ANTHROPIC_API_KEY)
            system_prompt: Custom system prompt
            tools: Tool configuration - list of tool names or preset dict
                   e.g., {"type": "preset", "preset": "claude_code"}
            allowed_tools: List of allowed tool names (legacy, use tools instead)
            permission_mode: Permission mode (default, acceptEdits, plan, bypassPermissions)
            **kwargs: Additional ClaudeAgentOptions
        """
        # Default model priority: arg > env > deepseek-chat
        self.model = model or os.getenv("AGENTKIT_MODEL", "deepseek-chat")

        # Setup environment for SDK (it reads these variables)
        if api_base:
            os.environ["ANTHROPIC_API_BASE"] = api_base
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        # Build options based on tools type
        options_kwargs = {
            "model": self.model,
            "system_prompt": system_prompt,
            "permission_mode": permission_mode,
            **kwargs
        }

        # Handle tools parameter
        if tools is not None:
            if isinstance(tools, dict):
                # Preset configuration like {"type": "preset", "preset": "claude_code"}
                options_kwargs["tools"] = tools
            else:
                # List of tool names
                options_kwargs["allowed_tools"] = tools
        elif allowed_tools is not None:
            options_kwargs["allowed_tools"] = allowed_tools
        else:
            options_kwargs["allowed_tools"] = []

        self.options = ClaudeAgentOptions(**options_kwargs)
        self.client: Optional[ClaudeSDKClient] = None
        self._conversation_active = False

    @abstractmethod
    async def process_response(self, message: AssistantMessage) -> Any:
        """
        Process assistant response (implement in subclass)

        Args:
            message: Assistant message from SDK

        Returns:
            Processed result
        """
        pass

    async def start(self):
        """Initialize SDK client and start conversation"""
        if not self.client:
            self.client = ClaudeSDKClient(self.options)
            await self.client.connect()
            self._conversation_active = True

    async def stop(self):
        """Clean up SDK client connection"""
        if self.client:
            await self.client.disconnect()
            self._conversation_active = False

    async def run(self, prompt: str) -> Any:
        """
        Execute agent with a prompt

        Args:
            prompt: User prompt

        Returns:
            Processed result from process_response()
        """
        await self.start()

        try:
            await self.client.query(prompt)

            result = None
            async for message in self.client.receive_response():
                if isinstance(message, AssistantMessage):
                    result = await self.process_response(message)

            return result

        finally:
            await self.stop()

    async def chat(self, prompt: str) -> str:
        """
        Simplified chat interface that extracts text response

        Args:
            prompt: User message

        Returns:
            Text response from Claude
        """
        if not self._conversation_active:
            await self.start()

        await self.client.query(prompt)

        response_text = []
        async for message in self.client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text.append(block.text)

        return "\n".join(response_text)

    async def __aenter__(self):
        """Async context manager support"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup"""
        await self.stop()
