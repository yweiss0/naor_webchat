# NextDawnAI Chatbot Workflow

Below is a visual representation of the NextDawnAI chatbot's processing flow, from receiving a user query to delivering the response.

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
    F -->|Stock Price Query| H[Get Real-time Stock Data]
    F -->|Web Search Needed| I[Perform Web Search]
    F -->|General Query| J[Process with LLM]
    
    J --> K{LLM Needs Tools?}
    K -->|Yes| L[Execute Tool Calls]
    K -->|No| M[Generate Direct Response]
    
    L --> N[Process Tool Results]
    N --> O[Generate Final Response with Context]
    
    G --> P[Format Response]
    H --> P
    I --> P
    M --> P
    O --> P
    
    P --> Q[Update Conversation History]
    Q --> R[Return Response to User]

    subgraph "Tool Execution"
        L1[Tool Handler] --> L2{Tool Type?}
        L2 -->|Stock Price| L3[Yahoo Finance API]
        L2 -->|Web Search| L4[DuckDuckGo Search]
        L3 --> L5[Return Price Data]
        L4 --> L6[Return Search Results]
    end

    L -.-> L1
    L5 -.-> N
    L6 -.-> N
```

## Classification Logic

```mermaid
flowchart LR
    A[User Query] --> B{Is Stock Related?}
    B -->|Yes| C{Is Price Query?}
    B -->|No| D[General Query]
    
    C -->|Yes| E[Extract Ticker Symbol]
    C -->|No| F{Needs Web Search?}
    
    F -->|Yes| G[Web Search Query]
    F -->|No| H[General Financial Query]
    
    E --> I[Stock Price Query]
    
    I --> J[Yahoo Finance]
    G --> K[DuckDuckGo]
    H --> L[GPT Model]
    D --> L
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
```

## Component Interaction

```mermaid
flowchart TD
    A[FastAPI App] --> B[Endpoints]
    B --> C[Services]
    C --> D[Models]
    C --> E[Utils]
    
    B -->|chat.py| B1[Chat Endpoint]
    B -->|health.py| B2[Health Endpoint]
    
    C -->|classification.py| C1[Query Classification]
    C -->|market_data.py| C2[Market Data]
    C -->|price_handler.py| C3[Price Handling]
    C -->|qa_service.py| C4[QA Service]
    C -->|tool_handler.py| C5[Tool Handling]
    C -->|web_search.py| C6[Web Search]
    
    E -->|text_processing.py| E1[Text Processing]
    
    F[Redis] <-.-> B1
    G[OpenAI] <-.-> C1
    G <-.-> C5
    H[Yahoo Finance] <-.-> C2
    I[DuckDuckGo] <-.-> C6
    J[Langfuse] <-.-> B1
``` 