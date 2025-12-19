# AgentKit - Universal Agent Development Framework

**Design Document**
**Date:** 2025-12-19
**Status:** Approved

## Overview

AgentKit is a universal agent development framework built on top of Claude Agents SDK, designed to support multiple LLM models (DeepSeek, Claude, OpenAI) and provide high-level abstractions for rapid agent development.

## Design Goals

1. **Multi-Model Support**: Work with DeepSeek, Claude, OpenAI, and any OpenAI-compatible API
2. **Developer Friendly**: Simple API for common use cases, powerful API for advanced scenarios
3. **Well-Tested**: Comprehensive test coverage (unit + integration)
4. **Production Ready**: Complete documentation, CLI tools, deployment configs
5. **SDK-Based**: Built on Claude Agents SDK, not reinventing the wheel

## Architecture

### Technology Stack

- **Package Manager**: uv (modern, fast Python package manager)
- **Core Framework**: Claude Agents SDK
- **Python Version**: 3.10+
- **Key Dependencies**:
  - `claude-agent-sdk` - Core agent framework
  - `click` - CLI interface
  - `pydantic-settings` - Configuration management
  - `python-dotenv` - Environment variable loading
  - `httpx` - Async HTTP client

### Project Structure

```
agentkit/
├── src/
│   └── agentkit/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent.py          # Base Agent class
│       │   ├── context.py        # Context management (session-scoped)
│       │   └── subagent.py       # Subagent orchestration
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py       # ToolCollection helper
│       │   └── builtin/
│       │       ├── file_tools.py
│       │       └── web_tools.py
│       ├── patterns/
│       │   ├── __init__.py
│       │   └── data_agents.py   # Reader/Writer/Query pattern
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py          # CLI commands
│       └── config/
│           ├── __init__.py
│           └── settings.py      # Configuration management
├── examples/
│   ├── simple_agent.py
│   ├── tool_agent.py
│   └── subagent_orchestrator.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_agent.py
│   │   ├── test_tools.py
│   │   └── test_subagent.py
│   └── integration/
│       ├── test_workflows.py
│       └── test_cli.py
├── docs/
│   ├── getting-started.md
│   ├── api-reference.md
│   ├── examples.md
│   └── plans/
│       └── 2025-12-19-agentkit-design.md (this file)
├── pyproject.toml
├── .env.example
└── README.md
```

## Core Components

### 1. Agent Base Class

**Location**: `src/agentkit/core/agent.py`

**Design**:
- Wraps `ClaudeSDKClient` from Claude Agents SDK
- Provides simplified interface for common operations
- Supports async context manager protocol
- Handles model configuration (DeepSeek/Claude/OpenAI via env vars)

**Key Features**:
- `run(prompt)` - One-shot execution
- `chat(prompt)` - Interactive conversation
- `start()/stop()` - Lifecycle management
- `process_response(message)` - Abstract method for response handling

**Configuration**:
- Model name from `AGENTKIT_MODEL` env var (default: `deepseek-chat`)
- API base URL from `ANTHROPIC_API_BASE` (optional, for non-Claude APIs)
- API key from `ANTHROPIC_API_KEY`

### 2. Context Management

**Location**: `src/agentkit/core/context.py`

**Design**:
- Three-layer context system:
  - `SESSION`: Shared across entire conversation
  - `AGENT`: Agent-specific private data
  - `TOOL`: Tool call stack
- Support for context inheritance (subagents can share or isolate)
- Automatic conversation history tracking
- Token count estimation for summarization triggers

**Key Features**:
- `create_child(isolated=False)` - Create child context for subagents
- `get/set(key, layer)` - Layer-aware data access
- `add_message(role, content)` - History tracking
- `should_summarize()` - Context overflow detection

**Note**: Cross-session persistence is designed but NOT implemented (YAGNI)

### 3. Tool System

**Location**: `src/agentkit/tools/`

**Design**:
- Uses Claude SDK's `@tool` decorator (no custom implementation)
- `ToolCollection` helper for grouping and registration
- Predefined tool collections: `file_toolkit`, `web_toolkit`, `standard_toolkit`

**Built-in Tools**:
- `read_json`, `write_json` - JSON file operations
- `list_directory` - Directory listing with glob patterns
- `http_request` - HTTP API calls

**Custom Tool Creation**:
```python
from claude_agent_sdk import tool

@tool("my_tool", "Description", {"param": str})
async def my_tool(args):
    return {"content": [{"type": "text", "text": "result"}]}
```

**Integration with Agents**:
```python
agent = Agent(toolkits=[file_toolkit, web_toolkit])
```

### 4. Subagent System

**Location**: `src/agentkit/core/subagent.py`

**Design**:
- `OrchestratorAgent` - Base class for multi-agent coordination
- `SubagentConfig` - Configuration for each subagent type
- Support for both pre-registered and dynamic subagents

**Predefined Patterns** (`src/agentkit/patterns/data_agents.py`):
- `ReaderAgent` - Read-only operations (Read, Glob, Grep)
- `WriterAgent` - Write operations (Write, Edit) - isolated context
- `QueryAgent` - Search operations (Grep, WebSearch)
- `DataOrchestrator` - Combines all three patterns

**Key Features**:
- `spawn_subagent(name, prompt)` - Execute registered subagent
- `delegate(task)` - Auto-select appropriate subagent
- `create_specialist(...)` - Dynamic subagent creation

### 5. CLI Interface

**Location**: `src/agentkit/cli/main.py`

**Commands**:
- `agentkit init` - Initialize new project (`.env`, examples, `pyproject.toml`)
- `agentkit run -p "prompt"` - One-shot execution
- `agentkit chat` - Interactive session
- `agentkit create <name>` - Generate agent template

**Options**:
- `--model` / `-m` - Model selection
- `--agent` / `-a` - Agent type (simple/data/orchestrator)
- `--tools` / `-t` - Additional tools
- `--env-file` - Custom env file path
- `--verbose` / `-v` - Verbose output

### 6. Configuration Management

**Location**: `src/agentkit/config/settings.py`

**Design**:
- Pydantic Settings for type-safe configuration
- Environment variable loading with `.env` file support
- Helper properties: `is_deepseek`, `is_openai`, `is_claude`

**Configuration Variables**:
```bash
AGENTKIT_MODEL=deepseek-chat
ANTHROPIC_API_BASE=https://api.deepseek.com
ANTHROPIC_API_KEY=sk-xxxxx
```

## Testing Strategy

### Unit Tests (`tests/unit/`)

**Coverage**:
- Agent initialization and lifecycle
- Context management operations
- Tool registration and execution
- Subagent configuration

**Approach**:
- Mock `ClaudeSDKClient` to avoid API calls
- Test business logic in isolation
- Fast execution (< 1 second per test)

### Integration Tests (`tests/integration/`)

**Coverage**:
- End-to-end workflows
- CLI command execution
- Multi-agent orchestration

**Approach**:
- Use mocks for most scenarios
- `@pytest.mark.slow` for real API tests (opt-in)
- Test complete user journeys

### Test Configuration

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests (require --run-slow)"
]
```

## Documentation

### User Documentation

1. **README.md** - Quick start, features, basic examples
2. **docs/getting-started.md** - Step-by-step setup guide
3. **docs/api-reference.md** - Complete API documentation
4. **docs/examples.md** - Advanced usage examples
5. **docs/tools.md** - Tool development guide
6. **docs/subagents.md** - Subagent patterns

### Developer Documentation

1. **docs/architecture.md** - Design decisions
2. **docs/plans/** - Design documents (this file)
3. **CONTRIBUTING.md** - Contribution guidelines

## Deployment

### Docker Support

```dockerfile
FROM python:3.11-slim
RUN pip install uv
COPY . /app
WORKDIR /app
RUN uv sync
ENTRYPOINT ["uv", "run", "agentkit"]
```

### CI/CD (GitHub Actions)

- Test on Python 3.10, 3.11, 3.12
- Run pytest with coverage
- Upload to codecov
- Auto-publish to PyPI on tag

### Package Distribution

- Publish to PyPI via `uv publish`
- Semantic versioning
- Changelog maintenance

## Design Decisions

### 1. Why Build on Claude Agents SDK?

**Decision**: Use Claude Agents SDK as foundation instead of building from scratch.

**Rationale**:
- SDK provides robust agent loop, context management, tool system
- MCP server integration already implemented
- Maintained by Anthropic
- Focus on higher-level abstractions, not reinventing

**Trade-offs**:
- Dependency on SDK (but it's well-maintained)
- Need to understand SDK internals for advanced use

### 2. Why Support Multiple Models?

**Decision**: Support DeepSeek, Claude, OpenAI via configurable API endpoints.

**Rationale**:
- Cost considerations (DeepSeek cheaper for development)
- Quality differences for production
- Flexibility for users
- Future-proof (new models appear constantly)

**Implementation**: Claude SDK uses OpenAI-compatible API, so any provider works via `ANTHROPIC_API_BASE`.

### 3. Why uv for Package Management?

**Decision**: Use uv instead of pip/poetry.

**Rationale**:
- Faster dependency resolution (10-100x)
- Modern tooling
- Built-in virtual environment management
- Growing adoption in Python community

**Trade-offs**: Requires users to install uv first.

### 4. Why Skip Cross-Session Context?

**Decision**: Session-scoped context only, no cross-session persistence.

**Rationale**:
- YAGNI - most use cases don't need it
- Adds complexity (storage, retrieval, cleanup)
- Can add later if needed
- Interface designed for future extension

### 5. Why Three Context Layers?

**Decision**: SESSION, AGENT, TOOL context layers.

**Rationale**:
- SESSION: Shared state across conversation
- AGENT: Private agent state (supports subagents)
- TOOL: Call stack for debugging

**Alternative Considered**: Single flat context - rejected due to lack of isolation.

### 6. Why Predefined Subagent Patterns?

**Decision**: Provide Reader/Writer/Query pattern out of the box.

**Rationale**:
- Common pattern from DeepSeek tutorial
- Demonstrates best practices
- Helps avoid context overload (26+ tools)
- Users can still create custom patterns

## Success Criteria

### Must Have (v0.1.0)

- ✅ Core Agent class with multi-model support
- ✅ Tool system integration (SDK-based)
- ✅ Subagent orchestration
- ✅ CLI interface (init, run, chat, create)
- ✅ Unit tests (>80% coverage)
- ✅ Documentation (README + getting-started)
- ✅ Example agents

### Should Have (v0.2.0)

- Integration tests
- More built-in tools
- API reference docs
- PyPI package

### Nice to Have (v1.0.0)

- Web UI
- Agent marketplace
- Cross-session context
- Advanced monitoring

## Implementation Plan

See separate implementation plan document (to be created via superpowers:writing-plans).

## Open Questions

None - design approved and ready for implementation.

## References

- Claude Agents SDK: https://github.com/anthropics/claude-agent-sdk-python
- Original DeepSeek Tutorial: https://github.com/NielsRogge/tutorials/tree/main/deepseekv3.2-mongodb
- uv Documentation: https://github.com/astral-sh/uv

---

**Design Approved**: 2025-12-19
**Next Step**: Create implementation plan and begin development
