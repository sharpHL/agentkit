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
