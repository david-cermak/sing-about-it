# Learning Sheet Generator

Where LLMs meet melody -- turning knowledge into something you can sing

## 🎯 Overview

An intelligent learning sheet generator that combines web search with AI to create comprehensive educational content. The system searches the web for current information about any topic, intelligently evaluates sources, scrapes full-text content, and uses a local LLM to synthesize it into well-structured learning materials.

## ✨ Features

- **🔍 Intelligent Web Search**: Multi-query web search using DuckDuckGo for comprehensive topic coverage
- **🧠 Smart Source Evaluation**: AI-powered source evaluation and selection based on relevance and authority
- **🌐 Robust Web Scraping**: Full-text content extraction using multiple methods (newspaper3k, readability, BeautifulSoup)
- **📄 Content Processing**: Intelligent cleaning, chunking, and quality validation of scraped content
- **🤖 AI-Powered Generation**: Local LLM integration via Ollama for content synthesis
- **🔄 Configurable Agent Backends**: Choose between pydantic.ai framework or direct OpenAI API calls
- **⚙️ Configurable Pipeline**: Environment-based configuration for search parameters, timeouts, and output settings
- **📊 Verbose Logging**: Detailed progress tracking showing search queries, results, and processing steps
- **🏗️ Modular Architecture**: Clean separation of concerns with dedicated modules for search, content processing, and generation
- **🔧 Type Safety**: Pydantic models for robust data validation and structure
- **🛡️ Robust Error Handling**: Graceful fallbacks and retry mechanisms for reliable operation
- **📂 Phase Management**: Run individual phases or the complete pipeline with result persistence

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

**Run the complete pipeline:**
```bash
python src/main.py --topic "machine learning fundamentals"
```

**Run individual phases:**
```bash
# Phase 1: Search for sources
python src/main.py --phase search --topic "blockchain technology"

# Phase 2: Evaluate and select sources
python src/main.py --phase eval --topic "blockchain technology"

# Phase 3: Scrape content from selected sources
python src/main.py --phase scrape --topic "blockchain technology"

# Phase 4: Generate learning sheet
python src/main.py --phase generate --topic "blockchain technology"
```

**Resume from saved results:**
```bash
# Continue from specific phase using saved results
python src/main.py --phase scrape --file results_evaluation_blockchain_technology.json

# List all saved result files
python src/main.py --list-saves
```

**Run the original single-file version:**
```bash
python src/sheet_generate.py
```

## 🏗️ Pipeline Architecture

The system operates through four distinct phases:

### Phase 1: 🔍 **Web Search**
- Executes multiple targeted search queries per topic
- Uses DuckDuckGo for current, relevant results
- Aggregates and deduplicates sources across queries
- **Output:** `results_search_[topic].json`

### Phase 2: 🧠 **Source Evaluation**
- AI-powered evaluation of each source for relevance and authority
- Scores sources based on content type, domain credibility, and topic alignment
- Selects top N sources for scraping (default: 8)
- **Output:** `results_evaluation_[topic].json`

### Phase 3: 🌐 **Content Scraping** ✅ **NEW**
- **Multi-method extraction**: newspaper3k → readability → BeautifulSoup fallback
- **Rate limiting**: Respectful delays between requests
- **Content processing**: Cleaning, chunking, and quality validation
- **Quality filtering**: Validates content length, structure, and coherence
- **Output:** `results_scraping_[topic].json`

### Phase 4: 📝 **Learning Sheet Generation**
- Synthesizes collected content into comprehensive learning materials
- Uses full scraped content instead of just snippets (50-100x more information)
- Creates structured, well-sourced educational content
- **Output:** `results_generation_[topic].json`

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🔍 Search     │───▶│  🧠 Evaluate    │───▶│  🌐 Scrape      │───▶│  📝 Generate    │
│   Multi-query   │    │  AI Selection   │    │  Full Content   │    │  Learning Sheet │
│   Web Search    │    │  Top Sources    │    │  Multi-method   │    │  Synthesis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
       📄                      📋                      📖                      📚
  Search Results         Selected Sources        Content Chunks           Learning Sheet
```

## 📋 Configuration

The system uses environment variables for configuration. Key settings include:

- **LLM Configuration**: Model selection, API endpoints, timeouts
- **Agent Backend Selection**: Choose between `pydantic_ai` or `baremetal` backends
- **Search Parameters**: Number of queries, results per query, search depth
- **Source Selection**: Max sources to scrape, minimum relevance score, authority weighting
- **Scraping Configuration**: Timeouts, delays, content limits, extraction methods
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
├── main.py                 # Main orchestrator with phase management
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
├── scraping/              # Web content extraction ✅ NEW
│   ├── web_scraper.py     # Multi-method content scraping
│   ├── content_processor.py # Content cleaning and chunking
│   └── scraper_models.py  # Scraping-related models
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

## 📊 Example Output

The enhanced system generates comprehensive learning sheets with:
- **Rich Content**: Full-text information from 50-100x more source material
- **Structured Format**: Clear headings, bullet points, and examples
- **Current Information**: Up-to-date web search results incorporated
- **Source Attribution**: Links to all referenced sources
- **Comprehensive Coverage**: Multiple perspectives and detailed explanations
- **Quality Assurance**: Content validation and intelligent filtering

### Content Volume Improvement
- **Before**: ~100 characters per source (snippets only)
- **After**: ~5,000+ characters per source (full article content)
- **Result**: 50-100x more information for LLM synthesis

## 🧪 Testing

**Test the web scraping functionality:**
```bash
python test_web_scraping.py
```

**Test the agent backend system:**
```bash
python test_agent_backends.py
```

These tests will:
- Demonstrate scraping with multiple extraction methods
- Show content processing and quality validation
- Test both pydantic.ai and baremetal backends
- Compare performance and reliability
- Show detailed output for debugging

For detailed usage instructions and examples, see [BACKEND_USAGE.md](BACKEND_USAGE.md).

## 🔧 Dependencies

### Core Dependencies
- **pydantic-ai**: AI agent framework (optional - for pydantic.ai backend)
- **openai**: OpenAI API client (for both backend types)
- **ollama**: Local LLM integration
- **ddgs**: DuckDuckGo search functionality
- **python-dotenv**: Environment configuration

### Web Scraping Dependencies ✅ NEW
- **newspaper3k**: Primary article extraction
- **readability-lxml**: Content extraction fallback
- **beautifulsoup4**: HTML parsing and cleaning
- **requests**: HTTP client for scraping
- **fake-useragent**: Rotating user agents for respectful scraping
- **markdownify**: HTML to Markdown conversion
- **validators**: URL validation

### Additional Libraries
- **aiohttp**: Async HTTP capabilities (future enhancement)
- **python-magic**: File type detection
- **urllib3**: Advanced URL handling

## 🎯 Implementation Status

- ✅ **Phase 1**: Modular architecture with configuration management
- ✅ **Phase 2**: Intelligent source evaluation with LLM agents
- ✅ **Phase 3**: Robust web scraping with multiple extraction methods
- ⏳ **Phase 4**: Enhanced generation with full content integration

**Next:** Integrate scraped content into learning sheet generation for dramatically improved quality and comprehensiveness.
