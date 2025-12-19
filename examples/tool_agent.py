"""Agent with custom tools example"""
import asyncio
from agentkit.core.agent import Agent
from agentkit.tools.registry import ToolCollection
from agentkit.tools.builtin.file_tools import read_json, write_json
from claude_agent_sdk import AssistantMessage, TextBlock


class ToolAgent(Agent):
    def __init__(self):
        toolkit = ToolCollection("files").add(read_json, write_json)

        super().__init__(
            system_prompt="You are a file management assistant",
        )
        # Note: In production, pass toolkits via mcp_servers parameter

    async def process_response(self, message: AssistantMessage) -> str:
        return "".join(
            block.text for block in message.content
            if isinstance(block, TextBlock)
        )


async def main():
    async with ToolAgent() as agent:
        result = await agent.chat(
            'Create a file called data.json with {"name": "agentkit", "version": "0.1.0"}'
        )
        print(result)

        result = await agent.chat("Read the data.json file")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
