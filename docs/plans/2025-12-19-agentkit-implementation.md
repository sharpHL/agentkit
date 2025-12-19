# AgentKit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a universal agent development framework on top of Claude Agents SDK supporting DeepSeek, Claude, and OpenAI models.

**Architecture:** Wrapper around Claude Agents SDK providing high-level Agent base class, tool collection helpers, subagent orchestration patterns, and CLI interface. Uses uv for package management, supports multiple LLM providers via environment configuration.

**Tech Stack:** Python 3.10+, claude-agent-sdk, click, pydantic-settings, pytest, uv

---

## Task 1: Project Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/agentkit/__init__.py`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```bash
cat > pyproject.toml << 'EOF'
[project]
name = "agentkit"
version = "0.1.0"
description = "Universal agent development framework built on Claude Agents SDK"
requires-python = ">=3.10"
dependencies = [
    "claude-agent-sdk>=0.1.0",
    "click>=8.1.0",
    "python-dotenv>=1.0.0",
    "pydantic-settings>=2.0.0",
    "httpx>=0.25.0",
]

[project.scripts]
agentkit = "agentkit.cli.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--verbose",
    "--strict-markers",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests that call real APIs",
]
EOF
```

**Step 2: Create .env.example**

```bash
cat > .env.example << 'EOF'
# AgentKit Configuration

# Model settings
AGENTKIT_MODEL=deepseek-chat

# API settings (DeepSeek)
ANTHROPIC_API_BASE=https://api.deepseek.com
ANTHROPIC_API_KEY=sk-your-key-here

# For Claude (uncomment and set)
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# For OpenAI (uncomment and set)
# ANTHROPIC_API_BASE=https://api.openai.com/v1
# ANTHROPIC_API_KEY=sk-your-key-here
EOF
```

**Step 3: Create .gitignore**

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
env/

# uv
.uv/
uv.lock

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db
EOF
```

**Step 4: Create src structure**

```bash
mkdir -p src/agentkit
touch src/agentkit/__init__.py
```

**Step 5: Initialize with uv**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 6: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/agentkit/__init__.py
git commit -m "feat: initialize project structure with uv

- Add pyproject.toml with dependencies
- Add .env.example for configuration
- Add .gitignore for Python project
- Create src/agentkit package

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Configuration Management

**Files:**
- Create: `src/agentkit/config/__init__.py`
- Create: `src/agentkit/config/settings.py`
- Create: `tests/unit/test_config.py`

**Step 1: Write the failing test**

```bash
mkdir -p tests/unit
cat > tests/unit/test_config.py << 'EOF'
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
EOF
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL with "No module named 'agentkit.config'"

**Step 3: Write minimal implementation**

```bash
mkdir -p src/agentkit/config
cat > src/agentkit/config/__init__.py << 'EOF'
from .settings import AgentKitSettings, settings

__all__ = ["AgentKitSettings", "settings"]
EOF

cat > src/agentkit/config/settings.py << 'EOF'
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


# Singleton instance
settings = AgentKitSettings(anthropic_api_key="not-set")  # Will be overridden by env
EOF
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agentkit/config/ tests/unit/test_config.py
git commit -m "feat: add configuration management

- Add AgentKitSettings with pydantic-settings
- Support AGENTKIT_MODEL, ANTHROPIC_API_BASE, ANTHROPIC_API_KEY
- Add API type detection (DeepSeek/Claude/OpenAI)
- Add comprehensive unit tests

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Core Agent Base Class

**Files:**
- Create: `src/agentkit/core/__init__.py`
- Create: `src/agentkit/core/agent.py`
- Create: `tests/unit/test_agent.py`
- Create: `tests/conftest.py`

**Step 1: Create test fixtures**

```bash
cat > tests/conftest.py << 'EOF'
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


@pytest.fixture
def mock_sdk_client(mock_llm_response):
    """Mock ClaudeSDKClient for testing without API calls"""
    client = AsyncMock()

    async def mock_receive_response():
        yield mock_llm_response("Test response")

    client.receive_response.return_value = mock_receive_response()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.query = AsyncMock()

    return client
EOF
```

**Step 2: Write the failing test**

```bash
cat > tests/unit/test_agent.py << 'EOF'
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
EOF
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_agent.py -v`
Expected: FAIL with "No module named 'agentkit.core'"

**Step 4: Write minimal implementation**

```bash
mkdir -p src/agentkit/core
cat > src/agentkit/core/__init__.py << 'EOF'
from .agent import Agent

__all__ = ["Agent"]
EOF

cat > src/agentkit/core/agent.py << 'EOF'
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
from abc import ABC, abstractmethod
from typing import Any, Optional
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
        tools: Optional[list[str]] = None,
        **kwargs
    ):
        """
        Initialize agent with flexible model configuration

        Args:
            model: Model name (defaults to env AGENTKIT_MODEL or "deepseek-chat")
            api_base: API base URL (defaults to env ANTHROPIC_API_BASE)
            api_key: API key (defaults to env ANTHROPIC_API_KEY)
            system_prompt: Custom system prompt
            tools: List of allowed tool names
            **kwargs: Additional ClaudeAgentOptions
        """
        # Default model priority: arg > env > deepseek-chat
        self.model = model or os.getenv("AGENTKIT_MODEL", "deepseek-chat")

        # Setup environment for SDK (it reads these variables)
        if api_base:
            os.environ["ANTHROPIC_API_BASE"] = api_base
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        self.options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt,
            allowed_tools=tools or [],
            permission_mode="acceptEdits",
            **kwargs
        )
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
EOF
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_agent.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/agentkit/core/ tests/unit/test_agent.py tests/conftest.py
git commit -m "feat: add core Agent base class

- Abstract Agent class wrapping ClaudeSDKClient
- Support for multi-model configuration via env vars
- run() for one-shot execution, chat() for interactive
- Async context manager support
- Comprehensive unit tests with mocks

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Tool System

**Files:**
- Create: `src/agentkit/tools/__init__.py`
- Create: `src/agentkit/tools/registry.py`
- Create: `src/agentkit/tools/builtin/__init__.py`
- Create: `src/agentkit/tools/builtin/file_tools.py`
- Create: `tests/unit/test_tools.py`

**Step 1: Write the failing test**

```bash
cat > tests/unit/test_tools.py << 'EOF'
import pytest
import json
from pathlib import Path
from agentkit.tools.registry import ToolCollection
from agentkit.tools.builtin.file_tools import read_json, write_json


@pytest.mark.unit
class TestToolSystem:
    def test_tool_collection_creation(self):
        """Test creating tool collection"""
        collection = ToolCollection("test_tools")
        assert collection.name == "test_tools"
        assert collection.version == "1.0.0"
        assert len(collection.tools) == 0

    def test_tool_collection_add(self):
        """Test adding tools to collection"""
        collection = ToolCollection("test_tools")
        collection.add(read_json, write_json)

        assert len(collection.tools) == 2

    def test_tool_collection_chaining(self):
        """Test chainable tool addition"""
        collection = (
            ToolCollection("test_tools")
            .add(read_json)
            .add(write_json)
        )

        assert len(collection.tools) == 2

    def test_tool_names_generation(self):
        """Test tool name generation for allowed_tools"""
        collection = ToolCollection("files")
        collection.add(read_json, write_json)

        names = collection.tool_names()
        assert "mcp__files__read_json" in names
        assert "mcp__files__write_json" in names

    @pytest.mark.asyncio
    async def test_read_json_tool(self, tmp_path):
        """Test read_json tool execution"""
        test_file = tmp_path / "test.json"
        test_data = {"name": "test", "value": 123}
        test_file.write_text(json.dumps(test_data))

        result = await read_json({"file_path": str(test_file)})

        assert "content" in result
        assert len(result["content"]) > 0
        assert "test" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_write_json_tool(self, tmp_path):
        """Test write_json tool execution"""
        test_file = tmp_path / "output.json"
        test_data = {"key": "value"}

        result = await write_json({
            "file_path": str(test_file),
            "data": test_data
        })

        assert test_file.exists()
        assert json.loads(test_file.read_text()) == test_data
EOF
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tools.py -v`
Expected: FAIL with "No module named 'agentkit.tools'"

**Step 3: Write minimal implementation**

```bash
mkdir -p src/agentkit/tools/builtin
cat > src/agentkit/tools/__init__.py << 'EOF'
from .registry import ToolCollection

__all__ = ["ToolCollection"]
EOF

cat > src/agentkit/tools/registry.py << 'EOF'
from claude_agent_sdk import create_sdk_mcp_server, SdkMcpTool
from typing import Any


class ToolCollection:
    """
    Manage collections of tools for easy registration

    Usage:
        toolkit = ToolCollection("my_tools")
        toolkit.add(tool1, tool2)
        agent = Agent(toolkits=[toolkit])
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: list[SdkMcpTool[Any]] = []

    def add(self, *tools: SdkMcpTool[Any]) -> "ToolCollection":
        """Add tools to collection (chainable)"""
        self.tools.extend(tools)
        return self

    def as_mcp_server(self):
        """Convert to MCP server config for SDK"""
        return create_sdk_mcp_server(
            name=self.name,
            version=self.version,
            tools=self.tools
        )

    def tool_names(self) -> list[str]:
        """Get list of tool names for allowed_tools"""
        return [f"mcp__{self.name}__{tool.name}" for tool in self.tools]
EOF

cat > src/agentkit/tools/builtin/__init__.py << 'EOF'
from .file_tools import read_json, write_json

__all__ = ["read_json", "write_json"]
EOF

cat > src/agentkit/tools/builtin/file_tools.py << 'EOF'
from claude_agent_sdk import tool
from typing import Any
import json


@tool(
    "read_json",
    "Read and parse JSON file",
    {"file_path": str}
)
async def read_json(args: dict[str, Any]) -> dict[str, Any]:
    """Read JSON file and return parsed content"""
    try:
        with open(args["file_path"], 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(data, indent=2, ensure_ascii=False)
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error reading JSON: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "write_json",
    "Write data to JSON file with formatting",
    {"file_path": str, "data": dict}
)
async def write_json(args: dict[str, Any]) -> dict[str, Any]:
    """Write formatted JSON to file"""
    try:
        with open(args["file_path"], 'w', encoding='utf-8') as f:
            json.dump(args["data"], f, indent=2, ensure_ascii=False)

        return {
            "content": [{
                "type": "text",
                "text": f"Successfully wrote JSON to {args['file_path']}"
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error writing JSON: {str(e)}"
            }],
            "is_error": True
        }
EOF
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tools.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agentkit/tools/ tests/unit/test_tools.py
git commit -m "feat: add tool system with registry and built-in tools

- ToolCollection for managing tool groups
- Built-in file tools (read_json, write_json)
- MCP server integration via SDK
- Comprehensive unit tests

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: CLI Interface

**Files:**
- Create: `src/agentkit/cli/__init__.py`
- Create: `src/agentkit/cli/main.py`
- Create: `tests/integration/test_cli.py`

**Step 1: Write the failing test**

```bash
mkdir -p tests/integration
cat > tests/integration/test_cli.py << 'EOF'
import pytest
from click.testing import CliRunner
from pathlib import Path
from agentkit.cli.main import cli


@pytest.mark.integration
class TestCLI:
    def test_cli_help(self):
        """Test CLI help command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "AgentKit" in result.output

    def test_init_command(self, tmp_path):
        """Test agentkit init command"""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert Path(".env").exists()
            assert Path("examples/simple_agent.py").exists()

    def test_create_command(self, tmp_path):
        """Test agentkit create command"""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["create", "my_agent"])

            assert result.exit_code == 0
            assert Path("my_agent.py").exists()

            content = Path("my_agent.py").read_text()
            assert "MyAgentAgent" in content
EOF
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_cli.py -v`
Expected: FAIL with "No module named 'agentkit.cli'"

**Step 3: Write minimal implementation**

```bash
mkdir -p src/agentkit/cli
cat > src/agentkit/cli/__init__.py << 'EOF'
EOF

cat > src/agentkit/cli/main.py << 'EOF'
import click
from pathlib import Path


@click.group()
@click.version_option(version="0.1.0", prog_name="agentkit")
def cli():
    """
    AgentKit - Universal agent development framework

    Built on Claude Agents SDK, supports DeepSeek, Claude, and OpenAI models.
    """
    pass


@cli.command()
def init():
    """
    Initialize a new agentkit project

    Creates:
        - .env file with configuration
        - examples/ directory with sample agents
    """
    click.echo("Initializing agentkit project...")

    # Create .env
    env_content = """# AgentKit Configuration

# Model settings
AGENTKIT_MODEL=deepseek-chat

# API settings (DeepSeek)
ANTHROPIC_API_BASE=https://api.deepseek.com
ANTHROPIC_API_KEY=sk-your-key-here

# For Claude (uncomment and set)
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# For OpenAI (uncomment and set)
# ANTHROPIC_API_BASE=https://api.openai.com/v1
# ANTHROPIC_API_KEY=sk-your-key-here
"""

    if not Path(".env").exists():
        Path(".env").write_text(env_content)
        click.echo("✓ Created .env file")
    else:
        click.echo("⊗ .env already exists, skipping")

    # Create examples/
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    simple_example = '''"""Simple agent example"""
import asyncio
from agentkit.core.agent import Agent
from claude_agent_sdk import AssistantMessage, TextBlock

class MyAgent(Agent):
    async def process_response(self, message: AssistantMessage) -> str:
        return "".join(
            block.text for block in message.content
            if isinstance(block, TextBlock)
        )

async def main():
    agent = MyAgent()
    result = await agent.run("Hello, explain what you can do")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
'''

    (examples_dir / "simple_agent.py").write_text(simple_example)
    click.echo("✓ Created examples/simple_agent.py")

    click.echo("\n✓ Project initialized!")
    click.echo("\nNext steps:")
    click.echo("  1. Edit .env and add your API key")
    click.echo("  2. Run: uv sync")
    click.echo("  3. Test: uv run python examples/simple_agent.py")


@cli.command()
@click.argument("agent_name")
def create(agent_name: str):
    """
    Create a new agent from template

    Example:
        agentkit create my_agent
    """
    agent_file = Path(f"{agent_name}.py")

    if agent_file.exists():
        click.echo(f"Error: {agent_file} already exists", err=True)
        return

    template = f'''"""
{agent_name} - Custom agent
"""
import asyncio
from agentkit.core.agent import Agent
from claude_agent_sdk import AssistantMessage, TextBlock

class {agent_name.title().replace("_", "")}Agent(Agent):
    """Custom agent implementation"""

    def __init__(self, **kwargs):
        super().__init__(
            system_prompt="Your custom system prompt here",
            tools=["Read", "Write", "Bash"],
            **kwargs
        )

    async def process_response(self, message: AssistantMessage) -> str:
        return "".join(
            block.text for block in message.content
            if isinstance(block, TextBlock)
        )

async def main():
    async with {agent_name.title().replace("_", "")}Agent() as agent:
        result = await agent.chat("Your task here")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
'''

    agent_file.write_text(template)
    click.echo(f"✓ Created {agent_file}")
    click.echo(f"\nEdit {agent_file} and run:")
    click.echo(f"  uv run python {agent_file}")


def main():
    """Entry point for CLI"""
    cli()


if __name__ == "__main__":
    main()
EOF
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_cli.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agentkit/cli/ tests/integration/test_cli.py
git commit -m "feat: add CLI interface with init and create commands

- agentkit init - initialize new project
- agentkit create <name> - generate agent template
- Integration tests with Click testing

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Examples and README

**Files:**
- Create: `examples/simple_agent.py`
- Create: `examples/tool_agent.py`
- Create: `README.md`

**Step 1: Create simple example**

```bash
mkdir -p examples
cat > examples/simple_agent.py << 'EOF'
"""Simple agent example"""
import asyncio
from agentkit.core.agent import Agent
from claude_agent_sdk import AssistantMessage, TextBlock


class SimpleAgent(Agent):
    async def process_response(self, message: AssistantMessage) -> str:
        return "".join(
            block.text for block in message.content
            if isinstance(block, TextBlock)
        )


async def main():
    agent = SimpleAgent()
    result = await agent.run("Explain Python async/await in simple terms")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
EOF
```

**Step 2: Create tool agent example**

```bash
cat > examples/tool_agent.py << 'EOF'
"""Agent with custom tools example"""
import asyncio
from agentkit.core.agent import Agent
from agentkit.tools.registry import ToolCollection
from agentkit.tools.builtin.file_tools import read_json, write_json
from claude_agent_sdk import AssistantMessage, TextBlock


class ToolAgent(Agent):
    def __init__(self):
        # Create toolkit with file tools
        toolkit = ToolCollection("files").add(read_json, write_json)

        super().__init__(
            toolkits=[toolkit],
            system_prompt="You are a file management assistant"
        )

    async def process_response(self, message: AssistantMessage) -> str:
        return "".join(
            block.text for block in message.content
            if isinstance(block, TextBlock)
        )


async def main():
    async with ToolAgent() as agent:
        # Create a JSON file
        result = await agent.chat(
            'Create a file called data.json with {"name": "agentkit", "version": "0.1.0"}'
        )
        print(result)

        # Read it back
        result = await agent.chat("Read the data.json file")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
EOF
```

**Step 3: Create README**

```bash
cat > README.md << 'EOF'
# AgentKit

Universal agent development framework built on Claude Agents SDK.

## Features

- 🚀 **Multi-Model Support**: DeepSeek, Claude, OpenAI, and any OpenAI-compatible API
- 🛠️ **Flexible Tools**: Built-in tools + easy custom tool creation
- 💻 **CLI Interface**: Command-line tools for quick development
- 🧪 **Well-Tested**: Comprehensive unit and integration tests
- 📦 **Modern Stack**: Built with uv, async/await, type hints

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agentkit.git
cd agentkit

# Install dependencies
uv sync
```

### Initialize Project

```bash
uv run agentkit init
```

### Configure API

Edit `.env`:

```bash
# For DeepSeek (default)
AGENTKIT_MODEL=deepseek-chat
ANTHROPIC_API_BASE=https://api.deepseek.com
ANTHROPIC_API_KEY=sk-your-deepseek-key
```

### Run Examples

```bash
uv run python examples/simple_agent.py
```

## Usage

### Simple Agent

```python
import asyncio
from agentkit.core.agent import Agent
from claude_agent_sdk import AssistantMessage, TextBlock

class SimpleAgent(Agent):
    async def process_response(self, message: AssistantMessage) -> str:
        return "".join(
            block.text for block in message.content
            if isinstance(block, TextBlock)
        )

async def main():
    agent = SimpleAgent()
    result = await agent.run("Explain Python decorators")
    print(result)

asyncio.run(main())
```

### CLI Usage

```bash
# Initialize project
uv run agentkit init

# Create new agent
uv run agentkit create my_agent

# Run example
uv run python my_agent.py
```

## Development

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_agent.py -v

# Check coverage
uv run pytest --cov=agentkit
```

## License

MIT License

## Acknowledgments

Built on top of [Claude Agents SDK](https://github.com/anthropics/claude-agent-sdk-python).
EOF
```

**Step 4: Commit**

```bash
git add examples/ README.md
git commit -m "docs: add examples and README

- Simple agent example
- Tool agent example with file operations
- Comprehensive README with quick start

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Final Package Setup

**Files:**
- Modify: `src/agentkit/__init__.py`

**Step 1: Update package __init__**

```bash
cat > src/agentkit/__init__.py << 'EOF'
"""
AgentKit - Universal agent development framework

Built on Claude Agents SDK, supporting DeepSeek, Claude, and OpenAI models.
"""

from .core.agent import Agent
from .tools.registry import ToolCollection
from .config.settings import AgentKitSettings, settings

__version__ = "0.1.0"
__all__ = ["Agent", "ToolCollection", "AgentKitSettings", "settings"]
EOF
```

**Step 2: Verify installation**

Run: `uv run python -c "import agentkit; print(agentkit.__version__)"`
Expected: Output "0.1.0"

**Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/agentkit/__init__.py
git commit -m "feat: finalize package exports and version

- Export main classes from package root
- Set version to 0.1.0
- Ready for initial release

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Testing the Complete System

**Step 1: Create test environment**

```bash
cp .env.example .env
# Edit .env and add real API key
```

**Step 2: Test CLI init**

Run: `cd /tmp && mkdir test-agentkit && cd test-agentkit && uv run agentkit init`
Expected: Project initialized with .env and examples/

**Step 3: Test simple agent**

Run: `uv run python examples/simple_agent.py`
Expected: Agent responds with explanation of async/await

**Step 4: Final commit**

```bash
git add .
git commit -m "chore: complete initial implementation

All core features implemented and tested:
- Core Agent class with multi-model support
- Configuration management
- Tool system with registry
- CLI interface (init, create)
- Examples and documentation
- Comprehensive test coverage

Ready for v0.1.0 release

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ CLI commands work (init, create)
- ✅ Examples run successfully
- ✅ Package imports correctly
- ✅ Documentation is complete

## Next Steps (Future Versions)

1. **v0.2.0**: Add subagent orchestration (Reader/Writer/Query pattern)
2. **v0.3.0**: Add more built-in tools (web_tools, etc.)
3. **v1.0.0**: Complete documentation, publish to PyPI

---

**Plan Complete**: Ready for implementation
