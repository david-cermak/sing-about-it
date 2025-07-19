# Learning Sheet Generator

Where LLMs meet melody -- turning knowledge into something you can sing

## 🎯 Overview

An intelligent learning sheet generator that combines web search with AI to create comprehensive educational content. The system searches the web for current information about any topic and uses a local LLM to synthesize it into well-structured learning materials.

## ✨ Features

- **🔍 Intelligent Web Search**: Multi-query web search using DuckDuckGo for comprehensive topic coverage
- **🤖 AI-Powered Generation**: Local LLM integration via Ollama for content synthesis
- **⚙️ Configurable Pipeline**: Environment-based configuration for search parameters, timeouts, and output settings
- **📊 Verbose Logging**: Detailed progress tracking showing search queries, results, and processing steps
- **🏗️ Modular Architecture**: Clean separation of concerns with dedicated modules for search, content processing, and generation
- **🔧 Type Safety**: Pydantic models for robust data validation and structure

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
- **Search Parameters**: Number of queries, results per query, search depth
- **Content Processing**: Chunking settings, content limits, quality thresholds
- **Logging**: Verbosity levels, debug options, output formatting

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
├── agents/                 # AI agents
│   ├── source_evaluator.py # Source quality evaluation (planned)
│   └── sheet_generator.py  # Learning sheet generation
├── scraping/              # Web content extraction (planned)
└── utils/                 # Utility functions
    ├── text_utils.py      # Text processing
    └── logging_utils.py   # Enhanced logging
```

## 📊 Example Output

The system generates comprehensive learning sheets with:
- **Structured Content**: Clear headings, bullet points, and examples
- **Current Information**: Up-to-date web search results incorporated
- **Source Attribution**: Links to all referenced sources
- **Comprehensive Coverage**: Multiple perspectives and use cases

## 🔧 Dependencies

- **pydantic-ai**: AI agent framework
- **ollama**: Local LLM integration
- **ddgs**: DuckDuckGo search functionality
- **python-dotenv**: Environment configuration
- **Additional libraries**: BeautifulSoup4, requests, and others for future web scraping capabilities
