# Agent Backend Configuration Guide

This document explains how to use the new configurable agent backend system that allows you to choose between pydantic.ai and "baremetal" OpenAI API approaches.

## Overview

The system now supports two different agent backends:

1. **Pydantic.ai Backend** - Uses the pydantic.ai framework with structured outputs, automatic retries, and rich debugging
2. **Baremetal Backend** - Direct OpenAI API calls with manual JSON parsing, similar to your `agent.py` pattern

Both backends maintain the **same interface**, so you can switch between them without changing your code.

## Quick Start

### Option 1: Environment Variable Configuration

Set the `AGENT_BACKEND` environment variable:

```bash
# Use pydantic.ai backend (default)
export AGENT_BACKEND=pydantic_ai

# Use baremetal backend
export AGENT_BACKEND=baremetal
```

### Option 2: .env File Configuration

Add to your `.env` file:

```env
# Agent Backend Configuration
AGENT_BACKEND=baremetal  # or "pydantic_ai"
```

### Option 3: Programmatic Configuration

```python
from agents.factory import AgentFactory, BackendType
from agents.source_evaluator_new import SourceEvaluatorAgent

# Create specific backend
pydantic_backend = AgentFactory.create_backend(
    backend_type=BackendType.PYDANTIC_AI,
    system_prompt="Your system prompt here"
)

baremetal_backend = AgentFactory.create_backend(
    backend_type=BackendType.BAREMETAL,
    system_prompt="Your system prompt here"
)

# Use with agents
evaluator = SourceEvaluatorAgent(backend=pydantic_backend)
```

## Backend Comparison

| Feature | Pydantic.ai Backend | Baremetal Backend |
|---------|-------------------|-------------------|
| **Structured Output** | ✅ Automatic validation | ✅ Manual JSON parsing |
| **Error Handling** | ✅ Built-in retries | ✅ Custom retry logic |
| **Debugging** | ✅ Rich logfire integration | ✅ Raw response access |
| **Performance** | 🟡 Framework overhead | ✅ Direct API calls |
| **Simplicity** | 🟡 Abstracted complexity | ✅ Transparent operation |
| **Reliability** | ✅ Battle-tested framework | 🟡 Manual implementation |

## When to Use Each Backend

### Choose **Pydantic.ai** when:
- You want robust, production-ready agents
- You need advanced debugging capabilities
- You're building complex multi-agent systems
- You prefer framework-managed error handling

### Choose **Baremetal** when:
- You want maximum control over API calls
- You're debugging LLM response issues
- You prefer simple, transparent code
- You're building lightweight applications

## Migration from Existing Agents

### Current Agents (Backward Compatible)

Your existing code continues to work unchanged:

```python
# This still works exactly as before
from agents.source_evaluator import evaluate_sources
from agents.sheet_generator import generate_learning_sheet_from_snippets

evaluations = evaluate_sources(search_results, topic)
sheet = generate_learning_sheet_from_snippets(topic, results)
```

### New Configurable Agents

Switch to the new backend-aware agents:

```python
# New backend-configurable agents
from agents.source_evaluator_new import evaluate_sources
from agents.sheet_generator_new import generate_learning_sheet_from_snippets

# Same interface, configurable backend
evaluations = evaluate_sources(search_results, topic)
sheet = generate_learning_sheet_from_snippets(topic, results)
```

## Configuration Examples

### Example 1: Development with Baremetal Backend

```env
# .env file for development
AGENT_BACKEND=baremetal
LLM_MODEL=llama3.2
LLM_BASE_URL=http://127.0.0.1:11434/v1
VERBOSE_OUTPUT=true
```

Benefits: Easy debugging, transparent API calls, simple troubleshooting

### Example 2: Production with Pydantic.ai Backend

```env
# .env file for production
AGENT_BACKEND=pydantic_ai
LLM_MODEL=gpt-4
LLM_BASE_URL=https://api.openai.com/v1
LOG_LEVEL=INFO
VERBOSE_OUTPUT=false
```

Benefits: Robust error handling, structured logging, automatic retries

## Testing the Backends

Run the comprehensive test suite:

```bash
python test_agent_backends.py
```

This will:
- Test both backends with the same inputs
- Compare performance and reliability
- Demonstrate environment configuration
- Show detailed output for debugging

## Troubleshooting

### Common Issues

**1. Baremetal Backend JSON Parsing Errors**
```
Solution: Check LLM response format. The baremetal backend includes enhanced
JSON cleaning and validation. Enable verbose output to see raw responses.
```

**2. Pydantic.ai Backend Validation Errors**
```
Solution: The framework handles most validation automatically. Check your
system prompts and ensure they request proper JSON structure.
```

**3. Backend Selection Not Working**
```
Solution: Ensure environment variable is set correctly and restart your
application. Check settings.agent.backend_type value.
```

### Debug Mode

Enable detailed debugging:

```python
from config.settings import settings
settings.logging.verbose_output = True
```

This will show:
- Backend selection process
- Raw LLM responses (baremetal backend)
- Validation steps and errors
- Performance timing information

## Advanced Usage

### Custom Backend Implementation

You can create your own backend by implementing the `AgentBackend` interface:

```python
from agents.base import AgentBackend
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class CustomBackend(AgentBackend):
    def generate_response(self, prompt: str, output_type: Type[T]) -> T:
        # Your custom implementation
        pass
```

### Mixed Backend Usage

Use different backends for different agents:

```python
from agents.factory import AgentFactory, BackendType
from agents.source_evaluator_new import SourceEvaluatorAgent
from agents.sheet_generator_new import SheetGeneratorAgent

# Use baremetal for evaluation (faster, simpler)
eval_backend = AgentFactory.create_backend(
    BackendType.BAREMETAL,
    "Source evaluation prompt"
)
evaluator = SourceEvaluatorAgent(backend=eval_backend)

# Use pydantic.ai for generation (more robust)
gen_backend = AgentFactory.create_backend(
    BackendType.PYDANTIC_AI,
    "Sheet generation prompt"
)
generator = SheetGeneratorAgent(backend=gen_backend)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│     SourceEvaluatorAgent     │     SheetGeneratorAgent      │
├─────────────────────────────────────────────────────────────┤
│                  AgentBackend Interface                     │
├─────────────────────────────┬───────────────────────────────┤
│      PydanticAIBackend      │      BaremetalBackend         │
├─────────────────────────────┼───────────────────────────────┤
│        pydantic.ai          │      Direct OpenAI API        │
│    - Structured output      │    - Manual JSON parsing      │
│    - Auto retries          │    - Custom retry logic       │
│    - Rich debugging        │    - Raw response access      │
└─────────────────────────────┴───────────────────────────────┘
```

The factory pattern ensures a consistent interface while allowing backend flexibility.

## Conclusion

The new agent backend system provides the best of both worlds:

- **Flexibility**: Choose the right backend for your needs
- **Compatibility**: Existing code continues to work
- **Transparency**: Full control when needed
- **Robustness**: Framework reliability when desired

Start with the default pydantic.ai backend for most use cases, and switch to baremetal when you need more control or simpler debugging.
