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

        # Call the handler directly (SdkMcpTool wraps the function)
        result = await read_json.handler({"file_path": str(test_file)})

        assert "content" in result
        assert len(result["content"]) > 0
        assert "test" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_write_json_tool(self, tmp_path):
        """Test write_json tool execution"""
        test_file = tmp_path / "output.json"
        test_data = {"key": "value"}

        # Call the handler directly (SdkMcpTool wraps the function)
        result = await write_json.handler({
            "file_path": str(test_file),
            "data": test_data
        })

        assert test_file.exists()
        assert json.loads(test_file.read_text()) == test_data
