# Refactor with classify call
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
import os
import json
from duckduckgo_search import DDGS
from datetime import datetime, timedelta, timezone
import re
import uuid
import redis.asyncio as redis
from contextlib import asynccontextmanager
import traceback

# --- Configuration & Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Redis Configuration (Reads from .env) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
    print("Config: Redis URL uses password.")
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    print("Config: Redis URL does not use password.")

SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours
MAX_HISTORY_PAIRS = 10
MAX_HISTORY_MESSAGES = MAX_HISTORY_PAIRS * 2


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    message: str


# --- Lifespan Context Manager (Handles ALL Client Initialization) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_conn = None
    app.state.openai_client = None
    print("Lifespan: Initializing resources...")
    try:
        redis_connection = redis.Redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True
        )
        await redis_connection.ping()
        app.state.redis_conn = redis_connection
        print("Lifespan: Redis client created and connected.")
    except Exception as e:
        print(f"Lifespan Startup Error: Could not connect to Redis: {e}")
        app.state.redis_conn = None
    if OPENAI_API_KEY:
        try:
            app.state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("Lifespan: OpenAI client created.")
        except Exception as e:
            print(f"Lifespan Startup Error: Could not initialize OpenAI client: {e}")
            app.state.openai_client = None
    else:
        print("Lifespan Warning: OPENAI_API_KEY not found.")
    yield
    print("Lifespan: Shutting down resources...")
    if hasattr(app.state, "redis_conn") and app.state.redis_conn:
        await app.state.redis_conn.close()
        print("Lifespan: Redis connection closed.")
    print("Lifespan: Shutdown complete.")


app = FastAPI(lifespan=lifespan)

# --- CORS configuration (Specific origins for credentialed requests) ---
origins = [
    "http://localhost:5173",  # Local dev frontend
    "https://nextaisolutions.cloud",  # PRODUCTION FRONTEND
    # Add other specific origins if necessary
]
# Note: Avoid "*" with allow_credentials=True in production if possible

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use specific list
    allow_credentials=True,  # MUST be True for cookies
    allow_methods=["GET", "POST", "OPTIONS"],  # Be specific
    allow_headers=["*"],  # Or specify needed headers like Content-Type
    expose_headers=["Content-Length", "X-Request-ID"],
    max_age=600,
)


# --- Helper Functions --- (Keep as they are)
def is_related_to_stocks_crypto(query: str) -> bool:
    keywords = [
        "stock",
        "shares",
        "equity",
        "ticker",
        "crypto",
        "cryptocurrency",
        "bitcoin",
        "ethereum",
        "coin",
        "token",
        "trading",
        "invest",
        "investment",
        "buy",
        "sell",
        "hold",
        "market",
        "exchange",
        "nasdaq",
        "nyse",
        "price",
        "value",
        "valuation",
        "portfolio",
        "asset",
        "dividend",
        "earnings",
        "revenue",
        "profit",
        "loss",
        "ipo",
        "etf",
        "mutual fund",
        "bond",
        "forex",
        "commodity",
        "gold",
        "silver",
        "analysis",
        "forecast",
        "trend",
        "outlook",
        "recommendation",
        "company",
        "companies",
        "business",
        "corporation",
        "industry",
        "sector",
        "operations",
        "tesla",
        "tsla",
        "apple",
        "aapl",
        "microsoft",
        "msft",
        "google",
        "googl",
        "alphabet",
        "amazon",
        "amzn",
        "meta",
        "facebook",
        "nvidia",
        "nvda",
        "coinbase",
        "coin",
        "binance",
        "netflix",
        "nflx",
        "ford",
        "gm",
        "boeing",
        "hp",
        "biotech",
        "technology",
        "energy",
        "finance",
        "healthcare",
        "pharmaceutical",
        "semiconductor",
        "retail",
        "automotive",
    ]
    query_lower = query.lower()
    if any(
        re.search(r"\b" + re.escape(keyword) + r"\b", query_lower)
        or keyword in query_lower
        for keyword in keywords
    ):
        matched_keywords = [
            k
            for k in keywords
            if re.search(r"\b" + re.escape(k) + r"\b", query_lower) or k in query_lower
        ]
        print(f"Query deemed related. Matched keywords: {matched_keywords}")
        return True
    print("Query NOT deemed related based on keywords.")
    return False


def get_stock_price(ticker: str) -> float | str:
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
        if price is not None:
            return round(float(price), 2)
        hist = stock.history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
        return f"Could not retrieve price for {ticker}."
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return f"Error retrieving price for {ticker}."


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    print(f"DDG Search: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            if not results:
                return "No relevant information found."
            return "\n".join(
                [
                    f"- {res.get('title', 'No Title')}: {res.get('body', 'No snippet.')}"
                    for res in results
                ]
            )
    except Exception as e:
        print(f"DDG Error: {e}")
        return "Error performing web search."


def validate_usd_result(text: str) -> bool:
    return "$" in text or "USD" in text.upper()


def process_text(text: str) -> str:
    text = re.sub(r"```(json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    suffix = "It’s always good to get advice from our professionals here at NRDX.com."
    text_stripped = text.strip()
    if not text_stripped.endswith(suffix):
        if text_stripped and text_stripped[-1] not in ".!?":
            text_stripped += "."
        text_stripped += f" {suffix}"
    return text_stripped


# --- OpenAI Tool Definitions --- (Keep as they are)
stock_price_function = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the latest stock price for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., AAPL, MSFT)",
                }
            },
            "required": ["ticker"],
        },
    },
}
web_search_function = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information using DuckDuckGo. Use for current events, recent data, or information not likely known by the LLM.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}
available_tools = [stock_price_function, web_search_function]

# --- Core Logic ---


async def needs_web_search(user_query: str, client: OpenAI | None) -> bool:
    print(f"Classifying web search need for: '{user_query}'")
    query_lower = user_query.lower()
    recall_keywords = [
        "remember",
        "what was",
        "talked about",
        "previous",
        "before",
        "stock name i asked",
        "which stock",
    ]
    if len(query_lower.split()) < 12 and any(
        key in query_lower for key in recall_keywords
    ):
        print("DEBUG: Query classified as recall, skipping web search.")
        return False

    if not client:
        print("DEBUG: needs_web_search - OpenAI client is None, cannot classify.")
        return False
    try:
        classification_messages = [
            {
                "role": "system",
                "content": "Analyze the user query. Does it require searching the web for current events (e.g., today's news), real-time data (like specific current stock prices not covered by tools), or very recent information published today or within the last few days? Do NOT say True if the user is asking about the conversation history or what was said before. Answer only with 'True' or 'False'.",
            },
            {
                "role": "user",
                "content": f'User Query: "{user_query}"\n\nRequires Web Search (True/False):',
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=classification_messages,
            max_tokens=5,
            temperature=0.0,
        )
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: Web search classification result from LLM: '{result_text}'")
            return "true" in result_text
        else:
            print(
                "DEBUG: Could not parse classification response. Defaulting to False."
            )
            return False
    except Exception as e:
        print(f"DEBUG: Error during classification LLM call: {e}")
        return False


async def handle_tool_calls(
    response_message, user_query: str, messages_history: list, client: OpenAI | None
) -> str:
    if not client:
        return "Error: OpenAI client not available."
    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content or "Error: No tool calls or content."

    print(f"DEBUG: Handling {len(tool_calls)} tool call(s)...")
    messages_for_follow_up = messages_history + [response_message]

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        result_content = ""
        try:
            args_dict = json.loads(function_args_str)
            if function_name == "get_stock_price":
                ticker = args_dict.get("ticker")
                price = get_stock_price(ticker)
                result_content = (
                    f"Price for {ticker}: {price}"
                    if ticker
                    else "Error: Ticker missing."
                )
            elif function_name == "web_search":
                search_query_arg = args_dict.get("query")
                result_content = (
                    duckduckgo_search(
                        f"{search_query_arg} in USD on {datetime.now():%B %d, %Y}"
                    )
                    if search_query_arg
                    else "Error: Search query missing."
                )
            else:
                result_content = f"Error: Unknown function '{function_name}'."
            print(
                f"DEBUG: Tool '{function_name}' executed. Result snippet: {result_content[:50]}..."
            )
        except Exception as e:
            print(f"DEBUG: Error executing tool {function_name}: {e}")
            result_content = f"Error executing tool: {e}"

        messages_for_follow_up.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": result_content,
            }
        )

    print("DEBUG: Making follow-up LLM call with tool results...")
    try:
        follow_up_response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages_for_follow_up
        )
        final_content = follow_up_response.choices[0].message.content
        print(
            f"DEBUG: Follow-up Response snippet: {final_content[:50] if final_content else 'None'}..."
        )
        return final_content or "Error: No content in follow-up."
    except Exception as e:
        print(f"DEBUG: Error during follow-up LLM call: {e}")
        return f"Error summarizing tool results."


# --- /api/chat Endpoint (Uses app.state) ---
@app.post("/api/chat")
async def chat(query: QueryRequest, request: Request, response: Response):
    redis_conn = request.app.state.redis_conn
    client = request.app.state.openai_client

    if not client:
        raise HTTPException(
            status_code=503, detail="OpenAI client not available (init error?)"
        )
    if not redis_conn:
        print("DEBUG: WARNING - Redis connection is NOT available for this request!")

    user_query = query.message
    session_id = request.cookies.get("chatbotSessionId")
    loaded_history = []
    is_new_session = False

    print(
        f"\n--- Request Start (Session: {session_id[-6:] if session_id else 'NEW'}) ---"
    )
    print(f"DEBUG: User Query: {user_query}")
    print(f"DEBUG: Cookie 'chatbotSessionId' value: {session_id}")

    # Load History
    if redis_conn and session_id:
        print(f"DEBUG: Attempting to load history for session {session_id[-6:]}...")
        try:
            history_json = await redis_conn.get(session_id)
            if history_json:
                loaded_history = json.loads(history_json)
                if not isinstance(loaded_history, list):
                    print(
                        f"DEBUG: ERROR - Corrupt history type: {type(loaded_history)}. Resetting."
                    )
                    loaded_history = []
                    session_id = None
                else:
                    print(f"DEBUG: Successfully loaded {len(loaded_history)} messages.")
                    if loaded_history:
                        print(f"DEBUG: Last loaded msg: {loaded_history[-1]}")
                    await redis_conn.expire(session_id, SESSION_TTL_SECONDS)
            else:
                print(f"DEBUG: Session ID {session_id[-6:]} not found in Redis.")
                session_id = None
        except json.JSONDecodeError as json_err:
            print(
                f"DEBUG: ERROR - JSON Decode failed for session {session_id}: {json_err}. Resetting."
            )
            session_id = None
        except Exception as e:
            print(
                f"DEBUG: ERROR - Redis GET/EXPIRE failed: {e}. Proceeding without history."
            )

    # Create New Session ID
    if redis_conn and not session_id:
        is_new_session = True
        session_id = str(uuid.uuid4())
        print(f"DEBUG: Generated NEW session ID: {session_id[-6:]}")
        loaded_history = []

    # Core Logic
    if not is_related_to_stocks_crypto(user_query):
        print("Query not related. Returning restricted response.")
        return {
            "response": "I can only answer questions about stocks, cryptocurrency, or trading."
        }

    final_response_content = ""
    raw_ai_response = None
    messages_sent_to_openai = []

    try:
        search_needed = await needs_web_search(user_query, client)
        system_prompt = "You are a financial assistant specializing in stocks, cryptocurrency, and trading. Use the conversation history provided. You must provide very clear and explicit answers in USD. If the user asks for a recommendation, give a direct 'You should...' statement. Use provided tools when necessary. Ensure all prices are presented in USD. Refer back to previous turns in the conversation if the user asks."

        base_messages = [{"role": "system", "content": system_prompt}] + loaded_history
        current_user_message_dict = {"role": "user", "content": user_query}

        if search_needed:
            print("DEBUG: Web search determined NEEDED.")
            search_result_text = duckduckgo_search(
                f"{user_query} price in USD on {datetime.now():%B %d, %Y}"
            )
            contextual_prompt_content = f'Based on our previous conversation history AND the following recent web search results, please answer the user\'s latest query: "{user_query}"\n\nWeb Search Results:\n---\n{search_result_text}\n---\n\nYour concise answer:'
            contextual_user_message_dict = {
                "role": "user",
                "content": contextual_prompt_content,
            }
            messages_sent_to_openai = base_messages + [contextual_user_message_dict]
            print("DEBUG: Making LLM call with History + Search Results...")
        else:
            print("DEBUG: Web search determined NOT needed.")
            messages_sent_to_openai = base_messages + [current_user_message_dict]
            print("DEBUG: Making LLM call with History + Current Query...")

        print(
            f"DEBUG: TOTAL messages being sent to OpenAI: {len(messages_sent_to_openai)}"
        )
        if messages_sent_to_openai:
            print(f"DEBUG: First message sent: {messages_sent_to_openai[0]}")
            if len(messages_sent_to_openai) > 1:
                print(f"DEBUG: Last message sent: {messages_sent_to_openai[-1]}")

        openai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_sent_to_openai,
            tools=available_tools,
            tool_choice="auto",
        )

        response_message = openai_response.choices[0].message
        if response_message.tool_calls:
            print(f"DEBUG: Tool call(s) requested...")
            raw_ai_response = await handle_tool_calls(
                response_message, user_query, messages_sent_to_openai, client
            )
        else:
            raw_ai_response = response_message.content
            print(
                f"DEBUG: Direct text response received snippet: {raw_ai_response[:50] if raw_ai_response else 'None'}..."
            )

        if not raw_ai_response:
            print("DEBUG: ERROR - No final content generated.")
            final_response_content = "I encountered an issue."
            raw_ai_response = None
        else:
            final_response_content = process_text(raw_ai_response)
            print(f"DEBUG: Returning formatted response: {final_response_content}")

    # Error Handling
    except BadRequestError as bre:
        print(f"DEBUG: ERROR - OpenAI Bad Request: {bre}")
        raw_ai_response = None
        raise HTTPException(
            status_code=400,
            detail=f"API Error: {bre.body.get('message', 'Bad Request')}",
        )
    except HTTPException as http_exc:
        raw_ai_response = None
        raise http_exc
    except Exception as e:
        print(f"DEBUG: ERROR - Critical error in chat endpoint: {e}")
        traceback.print_exc()
        raw_ai_response = None
        raise HTTPException(status_code=500, detail="Internal server error.")

    # --- Save History ---
    if redis_conn and session_id and raw_ai_response:
        try:
            new_history_entry = [
                current_user_message_dict,
                {"role": "assistant", "content": raw_ai_response},
            ]
            updated_history = loaded_history + new_history_entry
            if len(updated_history) > MAX_HISTORY_MESSAGES:
                updated_history = updated_history[-MAX_HISTORY_MESSAGES:]
                print(f"DEBUG: History truncated to {len(updated_history)} messages.")

            history_to_save_json = json.dumps(updated_history)
            print(
                f"DEBUG: Attempting to save history for session {session_id[-6:]}. Size: {len(updated_history)} messages."
            )
            await redis_conn.set(
                session_id, history_to_save_json, ex=SESSION_TTL_SECONDS
            )
            print(f"DEBUG: History save successful for session {session_id[-6:]}.")
        except Exception as e:
            print(f"DEBUG: ERROR - Redis SET failed: {e}. History not saved.")

    # --- Set Cookie ---
    if is_new_session and session_id and redis_conn:
        print(
            f"DEBUG: Setting CROSS-ORIGIN cookie for new session {session_id[-6:]}..."
        )  # Updated log message
        response.set_cookie(
            key="chatbotSessionId",
            value=session_id,
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            samesite="None",  # MUST be 'None' for cross-origin
            path="/",
            secure=True,  # MUST be True when SameSite=None
        )

    print(f"--- Request End (Session: {session_id[-6:] if session_id else 'NEW'}) ---")
    return {"response": final_response_content}


# --- Health Check Endpoint ---
@app.get("/api/health")
async def health_check(request: Request):
    redis_conn_state = request.app.state.redis_conn
    openai_client_state = request.app.state.openai_client
    redis_status = "not_initialized"
    openai_status = "not_initialized"

    if redis_conn_state:
        try:
            await redis_conn_state.ping()
            redis_status = "connected"
        except Exception as e:
            print(f"Health Check Redis Ping Error: {e}")
            redis_status = "error_connecting"
    else:
        redis_status = "conn_object_none_in_state"

    if openai_client_state:
        openai_status = "initialized"
    else:
        openai_status = "client_object_none_in_state"

    print(
        f"Health Check: Redis Status = {redis_status}, OpenAI Status = {openai_status}"
    )
    return {
        "status": "OK_V2_cross_origin",
        "redis_status": redis_status,
        "openai_status": openai_status,
    }  # Changed status


# --- Optional: To Run Directly ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4, lifespan="on")
