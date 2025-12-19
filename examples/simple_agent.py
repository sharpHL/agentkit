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
