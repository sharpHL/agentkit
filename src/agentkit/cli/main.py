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
        click.echo("Created .env file")
    else:
        click.echo(".env already exists, skipping")

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
    click.echo("Created examples/simple_agent.py")

    click.echo("\nProject initialized!")
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
    click.echo(f"Created {agent_file}")
    click.echo(f"\nEdit {agent_file} and run:")
    click.echo(f"  uv run python {agent_file}")


def main():
    """Entry point for CLI"""
    cli()


if __name__ == "__main__":
    main()
