# NextDawnAI Chatbot Server

This document outlines the architecture, workflow, and components of the NextDawnAI financial assistant chatbot.

## Overview

The NextDawnAI chatbot is a financial assistant that specializes in stocks, cryptocurrency, and trading information. It provides real-time financial data, performs market analysis, and responds to user queries using a combination of AI language models, specialized tools, and external data sources.

## System Architecture

The application has been refactored into a modular, maintainable structure with clear separation of concerns:

```
server/
├── app/
│   ├── config.py           # Configuration settings
│   ├── lifecycle.py        # Application lifecycle management
│   ├── main.py             # FastAPI application initialization
│   ├── endpoints/          # API endpoints
│   │   ├── chat.py         # Main chat endpoint
│   │   └── health.py       # Health check endpoint
│   ├── models/             # Data models
│   │   └── data_models.py  # Pydantic models
│   ├── services/           # Business logic
│   │   ├── classification.py   # Query classification
│   │   ├── market_data.py      # Stock market data services
│   │   ├── price_handler.py    # Stock price query handling
│   │   ├── qa_service.py       # Q&A file operations
│   │   ├── tool_handler.py     # OpenAI tool call handling
│   │   └── web_search.py       # Web search services
│   └── utils/              # Utility functions
│       └── text_processing.py  # Text processing utilities
└── main.py                 # Entry point
```

## Request Workflow

When a user sends a message to the chatbot, the following flow occurs:

1. **Request Initialization**
   - The request is received via the `/api/chat` endpoint in `app/endpoints/chat.py`
   - Session management is handled through cookies and Redis
   - Conversation history is retrieved if available

2. **Query Classification**
   - Query is analyzed to determine its type and required processing
   - Classification services determine if the query:
     - Is related to stocks/crypto
     - Is specifically a stock price query
     - Needs web search for current information

3. **Processing Path Selection**
   - **Stock Price Queries**: Direct lookup via Yahoo Finance
   - **Web Search Needed**: DuckDuckGo search is performed for current data
   - **General Financial Questions**: Answered using the LLM with context

4. **Tool Execution**
   - If the LLM determines a tool is needed, the query is delegated to:
     - `get_stock_price`: Retrieves real-time financial data
     - `web_search`: Searches the web for current information

5. **Response Generation**
   - Final response is composed based on the processing path and tool results
   - Text is formatted for consistency and clarity
   - Response is returned to the user

6. **Session Management**
   - Conversation history is updated in Redis
   - Session cookies are maintained for continuity

## Key Components

### 1. Request Handling
- FastAPI framework for API endpoints
- Redis for session and conversation history storage
- Cookie-based session tracking

### 2. Query Classification
- AI-based classification for query intent determination
- Pattern matching for simple queries

### 3. Data Sources
- **Yahoo Finance API**: Real-time stock and financial instrument data
- **DuckDuckGo Search**: Web search for current events and data
- **Stored Q&A Pairs**: Fast responses for common questions

### 4. AI Integration
- OpenAI GPT models for:
  - Query understanding
  - Response generation
  - Tool selection and execution
- Langfuse for tracing and monitoring AI operations

### 5. Response Processing
- Text normalization and formatting
- Currency standardization (USD)
- Context-aware responses

## Monitoring & Observability

- Detailed logging throughout the request lifecycle
- Langfuse integration for AI operation tracing
- Health check endpoint for system status monitoring

## Environment Configuration

The application relies on environment variables for configuration:
- OpenAI API keys
- Redis connection details
- Langfuse monitoring settings
- CORS settings for frontend integration

## Deployment

The application is designed to run with:
- Uvicorn ASGI server
- Multiple workers for concurrent processing
- Nginx as a reverse proxy
- Redis for session storage 