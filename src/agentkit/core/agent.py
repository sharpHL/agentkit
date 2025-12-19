from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import os


def configure_deepseek(api_key: Optional[str] = None) -> None:
    """
    Configure environment for DeepSeek with Claude Agent SDK.

    DeepSeek provides an Anthropic-compatible endpoint at /anthropic.
    This function sets up all required environment variables.

    Args:
        api_key: DeepSeek API key (or uses DEEPSEEK_API_KEY / ANTHROPIC_API_KEY env var)
    """
    key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY or pass api_key.")

    os.environ["ANTHROPIC_BASE_URL"] = "https://api.deepseek.com/anthropic"
    os.environ["ANTHROPIC_AUTH_TOKEN"] = key
    os.environ["ANTHROPIC_MODEL"] = "deepseek-chat"
    os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = "deepseek-chat"
    os.environ["API_TIMEOUT_MS"] = "600000"
    os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"


class Agent(ABC):
    """
    High-level agent abstraction built on Claude Agents SDK

    Supports any model via environment configuration:
    - Claude models (default SDK behavior)
    - DeepSeek via configure_deepseek() or ANTHROPIC_BASE_URL
    - Other Anthropic-compatible endpoints

    Example with DeepSeek:
        from agentkit.core.agent import configure_deepseek
        configure_deepseek(api_key="sk-xxx")
        agent = MyAgent(model="deepseek-chat")
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
        use_deepseek: bool = False,
        **kwargs
    ):
        """
        Initialize agent with flexible model configuration

        Args:
            model: Model name (defaults to env AGENTKIT_MODEL or ANTHROPIC_MODEL)
            api_base: API base URL (defaults to env ANTHROPIC_BASE_URL)
            api_key: API key (defaults to env ANTHROPIC_AUTH_TOKEN)
            system_prompt: Custom system prompt
            tools: Tool configuration - list of tool names or preset dict
                   e.g., {"type": "preset", "preset": "claude_code"}
            allowed_tools: List of allowed tool names (legacy, use tools instead)
            permission_mode: Permission mode (default, acceptEdits, plan, bypassPermissions)
            use_deepseek: If True, auto-configure for DeepSeek using api_key
            **kwargs: Additional ClaudeAgentOptions
        """
        # Auto-configure DeepSeek if requested
        if use_deepseek:
            configure_deepseek(api_key)

        # Default model priority: arg > env AGENTKIT_MODEL > env ANTHROPIC_MODEL > deepseek-chat
        self.model = (
            model
            or os.getenv("AGENTKIT_MODEL")
            or os.getenv("ANTHROPIC_MODEL")
            or "deepseek-chat"
        )

        # Setup environment for SDK (using correct variable names)
        if api_base:
            os.environ["ANTHROPIC_BASE_URL"] = api_base
        if api_key and not use_deepseek:
            os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key

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
