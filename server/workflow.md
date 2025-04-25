# Naor Chatbot Workflow

Below is a visual representation of the Naor chatbot's processing flow, from receiving a user query to delivering the response.

If you the text is not visible when exporting it as svg try using this: https://mermaid.live/
you need to copy each graph code by itself, from the part flowchart TD to the end of that graph (until you see a new flowchart text)

## Request Processing Flow

```mermaid
flowchart TD
    A[User Query] --> B[API Endpoint: /api/chat]
    B --> C{Session Exists?}
    C -->|Yes| D[Load Conversation History]
    C -->|No| E[Create New Session]
    D --> F{Query Type?}
    E --> F
    
    F -->|QA Match| G[Direct Answer from QA Database]
    F -->|Stock Related?| S{Stock Related Check}
    S -->|No| T[Return Restricted Response]
    S -->|Yes| U{Stock Price Query?}
    
    U -->|Yes| H[Get Real-time Stock Data]
    U -->|No| V{Needs Web Search?}
    
    V -->|Yes| I[Perform Web Search]
    V -->|No| J[Process with LLM]
    
    J --> K{LLM Needs Tools?}
    K -->|Yes| L[Execute Tool Calls]
    K -->|No| M[Generate Direct Response]
    
    L --> N[Process Tool Results]
    N --> O[Generate Final Response with Context]
    
    G --> P[Format & Apply Guardrails]
    H --> P
    I --> P
    M --> P
    O --> P
    T --> P
    
    P --> Q[Update Conversation History]
    Q --> R[Return Response to User]

    subgraph "Tool Execution"
        L1[Tool Handler] --> L2{Tool Type?}
        L2 -->|Stock Price| L3[Yahoo Finance API]
        L2 -->|Web Search| L4[DuckDuckGo Search]
        L3 --> L5[Return Price Data]
        L4 --> L6[Return Search Results]
    end

    subgraph "Price Query Handling"
        H1[Extract Ticker] --> H2[Fetch Yahoo Finance Data]
        H2 --> H3{Ask for Reason?}
        H3 -->|Yes| H4[Add Web Search Data]
        H3 -->|No| H5[Format Price Response]
        H4 --> H5
    end

    L -.-> L1
    L5 -.-> N
    L6 -.-> N
    H -.-> H1
```

## Classification Logic

```mermaid
flowchart LR
    A[User Query] --> B{Is Stock Related?}
    B -->|No| C[Return Restricted Response]
    B -->|Yes| D{Is Price Query?}
    
    D -->|Yes| E[Extract Ticker Symbol]
    D -->|No| F{Needs Web Search?}
    
    F -->|Yes| G[Web Search Query]
    F -->|No| H[General Financial Query]
    
    E --> I[Stock Price Query]
    I --> J1{Is Reason Query?}
    J1 -->|Yes| J2[Add Market Data Context]
    J1 -->|No| J3[Simple Price Response]
    
    I --> J[Yahoo Finance]
    G --> K[DuckDuckGo]
    H --> L[GPT Model with Tools]
    J2 --> M[Web Search for Reasons]
```

## Session Management

```mermaid
flowchart TD
    A[Start] --> B{Cookie Present?}
    B -->|Yes| C[Load Session from Redis]
    B -->|No| D[Generate New Session ID]
    D --> E[Create Empty History]
    C --> F{Valid Session?}
    F -->|Yes| G[Load Conversation History]
    F -->|No| E
    
    G --> H[Process Query]
    E --> H
    
    H --> I[Update History in Redis]
    I --> J[Set/Update Session Cookie]
    J --> K[Return Response]
    
    subgraph "History Management"
        I1[Update Conversation List] --> I2[Trim to Max Messages]
        I2 --> I3[Set TTL for Session]
    end
    
    I -.-> I1
```

## Component Interaction

```mermaid
flowchart TD
    A[FastAPI App] --> B[Endpoints]
    B --> C[Services]
    C --> D[Models]
    C --> E[Utils]
    
    B -->|chat.py| B1[Chat Endpoint]
    B -->|chat_handler.py| B3[Chat Processing]
    B -->|health.py| B2[Health Endpoint]
    
    C -->|query_classification.py| C1[Stock Relevance Check]
    C -->|price_query_detector.py| C2[Price Query Detection]
    C -->|web_search_classifier.py| C3[Web Search Need Check]
    C -->|market_data.py| C4[Market Data Service]
    C -->|price_handler.py| C5[Price Handling]
    C -->|qa_service.py| C6[QA Matching Service]
    C -->|tool_handler.py| C7[Tool Handling]
    C -->|web_search.py| C8[Web Search]
    
    E -->|text_processing.py| E1[Text Processing]
    E -->|response_processor.py| E2[Response Processing]
    E -->|session_manager.py| E3[Session Management]
    
    F[Redis] <-.-> B1
    G[OpenAI] <-.-> C1
    G <-.-> C2
    G <-.-> C3
    G <-.-> C7
    H[Yahoo Finance] <-.-> C4
    I[DuckDuckGo] <-.-> C8
    J[Langfuse] <-.-> B1
```

## Langfuse Tracing Integration

```mermaid
flowchart TD
    A[Chat Request] --> B[Create Trace]
    B --> C[Span: Stock Relevance]
    C --> D{Stock Related?}
    D -->|No| E[End: Restricted]
    D -->|Yes| F[Span: Price Query Check]
    F --> G{Is Price Query?}
    G -->|Yes| H[Span: Price Retrieval]
    G -->|No| I[Span: Web Search Check]
    I --> J{Need Search?}
    J -->|Yes| K[Span: Web Search]
    J -->|No| L[Span: LLM Call]
    
    H --> M[End: Success]
    K --> L
    L --> M
    
    subgraph "Trace Metrics Captured"
        N1[Query Type]
        N2[Response Type]
        N3[Tool Usage]
        N4[Response Time]
        N5[User Session]
    end
``` 