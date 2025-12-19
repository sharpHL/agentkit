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
