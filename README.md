# AgentKit

Universal agent development framework built on Claude Agents SDK.

## Features

- **Multi-Model Support**: DeepSeek, Claude, OpenAI, and any OpenAI-compatible API
- **Flexible Tools**: Built-in tools + easy custom tool creation
- **CLI Interface**: Command-line tools for quick development
- **Well-Tested**: Comprehensive unit and integration tests
- **Modern Stack**: Built with uv, async/await, type hints

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/agentkit.git
cd agentkit
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

# For Claude
# AGENTKIT_MODEL=claude-sonnet-4-5
# ANTHROPIC_API_KEY=sk-ant-your-key

# For OpenAI
# AGENTKIT_MODEL=gpt-4
# ANTHROPIC_API_BASE=https://api.openai.com/v1
# ANTHROPIC_API_KEY=sk-your-key
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

### Interactive Chat

```python
async def chat():
    async with SimpleAgent() as agent:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = await agent.chat(user_input)
            print(f"Agent: {response}")

asyncio.run(chat())
```

### CLI Usage

```bash
# Initialize project
uv run agentkit init

# Create new agent
uv run agentkit create my_agent

# Run your agent
uv run python my_agent.py
```

## Development

```bash
# Run all tests
uv run pytest

# Run specific tests
uv run pytest tests/unit/test_agent.py -v

# Check coverage
uv run pytest --cov=agentkit
```

## Project Structure

```
agentkit/
├── src/agentkit/
│   ├── core/           # Agent base class
│   ├── config/         # Configuration management
│   ├── tools/          # Tool system
│   └── cli/            # CLI commands
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
└── examples/           # Usage examples
```

## License

MIT License

## Acknowledgments

Built on [Claude Agents SDK](https://github.com/anthropics/claude-agent-sdk-python).
