# Learning Sheet Generator

Where LLMs meet melody -- turning knowledge into something you can sing

## 🎯 Overview

An intelligent learning sheet generator that combines web search with AI to create comprehensive educational content. The system searches the web for current information about any topic and uses a local LLM to synthesize it into well-structured learning materials.

## ✨ Features

- **🔍 Intelligent Web Search**: Multi-query web search using DuckDuckGo for comprehensive topic coverage
- **🤖 AI-Powered Generation**: Local LLM integration via Ollama for content synthesis
- **🔄 Configurable Agent Backends**: Choose between pydantic.ai framework or direct OpenAI API calls
- **⚙️ Configurable Pipeline**: Environment-based configuration for search parameters, timeouts, and output settings
- **📊 Verbose Logging**: Detailed progress tracking showing search queries, results, and processing steps
- **🏗️ Modular Architecture**: Clean separation of concerns with dedicated modules for search, content processing, and generation
- **🔧 Type Safety**: Pydantic models for robust data validation and structure
- **🛡️ Robust Error Handling**: Graceful fallbacks and retry mechanisms for reliable operation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) running locally with llama3.2 model
- Virtual environment (recommended)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd sing-about-it
   python -m venv venv
   ```

2. **Activate virtual environment:**
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1

   # Windows Command Prompt
   venv\Scripts\activate.bat

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   # Copy and customize the configuration
   cp .env.example .env
   # Edit .env file with your preferred settings
   ```

### Usage

**Run the enhanced modular version:**
```bash
python src/main.py
```

**Run the original single-file version:**
```bash
python src/sheet_generate.py
```

## 📋 Configuration

The system uses environment variables for configuration. Key settings include:

- **LLM Configuration**: Model selection, API endpoints, timeouts
- **Agent Backend Selection**: Choose between `pydantic_ai` or `baremetal` backends
- **Search Parameters**: Number of queries, results per query, search depth
- **Content Processing**: Chunking settings, content limits, quality thresholds
- **Logging**: Verbosity levels, debug options, output formatting

### Agent Backend Configuration

Choose your preferred agent backend approach:

```bash
# Use pydantic.ai framework (default) - robust, production-ready
export AGENT_BACKEND=pydantic_ai

# Use baremetal OpenAI API - direct control, simple debugging
export AGENT_BACKEND=baremetal
```

**Pydantic.ai Backend**: Provides structured output validation, automatic retries, and rich debugging capabilities.

**Baremetal Backend**: Direct OpenAI API calls with manual JSON parsing and transparent operation, ideal for debugging and simple deployments.

See `.env.example` for all available configuration options with optimized defaults.

## 🏗️ Architecture

```
src/
├── main.py                 # Main orchestrator
├── config/                 # Configuration management
│   ├── settings.py        # Environment-based settings
│   └── models.py          # Pydantic data models
├── search/                 # Web search functionality
│   ├── web_search.py      # DuckDuckGo search implementation
│   └── search_models.py   # Search-related models
├── agents/                 # AI agents with configurable backends
│   ├── base.py            # Abstract backend interface
│   ├── factory.py         # Backend factory pattern
│   ├── backends/          # Backend implementations
│   │   ├── pydantic_backend.py  # Pydantic.ai integration
│   │   └── baremetal_backend.py # Direct OpenAI API calls
│   ├── source_evaluator.py     # Original implementation
│   ├── source_evaluator_new.py # Configurable backend version
│   ├── sheet_generator.py      # Original implementation
│   └── sheet_generator_new.py  # Configurable backend version
├── scraping/              # Web content extraction (planned)
└── utils/                 # Utility functions
    ├── text_utils.py      # Text processing
    └── logging_utils.py   # Enhanced logging
```

### Agent Backend System

The system uses a factory pattern to provide flexible agent backends:

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
│    - Auto retries           │    - Custom retry logic       │
│    - Rich debugging         │    - Raw response access      │
└─────────────────────────────┴───────────────────────────────┘
```

This architecture allows you to:
- **Switch backends** without changing application code
- **Mix backends** for different agents based on needs
- **Add custom backends** by implementing the AgentBackend interface
- **Maintain compatibility** with existing implementations

## 📊 Example Output

The system generates comprehensive learning sheets with:
- **Structured Content**: Clear headings, bullet points, and examples
- **Current Information**: Up-to-date web search results incorporated
- **Source Attribution**: Links to all referenced sources
- **Comprehensive Coverage**: Multiple perspectives and use cases

## 🔧 Dependencies

- **pydantic-ai**: AI agent framework (optional - for pydantic.ai backend)
- **openai**: OpenAI API client (for both backend types)
- **ollama**: Local LLM integration
- **ddgs**: DuckDuckGo search functionality
- **python-dotenv**: Environment configuration
- **requests**: HTTP client for direct API calls
- **Additional libraries**: BeautifulSoup4 and others for future web scraping capabilities

## 🧪 Testing

Test the agent backend system with the comprehensive test suite:

```bash
python test_agent_backends.py
```

This will:
- Test both pydantic.ai and baremetal backends
- Compare performance and reliability
- Demonstrate environment configuration
- Show detailed output for debugging

For detailed usage instructions and examples, see [BACKEND_USAGE.md](BACKEND_USAGE.md).
