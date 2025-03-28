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
import traceback  # Import traceback for better error logging

# --- Configuration & Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Redis Configuration (Reads from .env) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Construct REDIS_URL dynamically
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
    print("Redis configured WITH password.")
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    print("Redis configured WITHOUT password.")

SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours
MAX_HISTORY_PAIRS = 10
MAX_HISTORY_MESSAGES = MAX_HISTORY_PAIRS * 2

# --- Initialize Redis Connection Variable ---
redis_conn = None

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables!")
    client = None
else:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    message: str


# --- Lifespan Context Manager for Redis ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_conn
    print("Application startup: Initializing Redis connection...")
    try:
        redis_conn = redis.Redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True
        )
        await redis_conn.ping()
        print("Successfully connected to Redis during startup.")
    except Exception as e:
        print(f"Startup Error: Could not connect to Redis: {e}")
        redis_conn = None

    yield  # App runs here

    print("Application shutdown: Closing Redis connection...")
    if redis_conn:
        await redis_conn.close()
        print("Redis connection closed.")
    print("Application shutdown complete.")


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)


# CORS configuration
origins = [
    "http://localhost:5173",
    "https://localhost",
    "https://nextdawnai.cloud",
    "https://nextaisolutions.cloud",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.nextaisolutions\.cloud$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    if any(keyword in query_lower for keyword in keywords):
        print(f"Query deemed related based on keywords.")
        return True
    print("Query NOT deemed related to finance based on keywords.")
    return False


def get_stock_price(ticker: str) -> float | str:
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
        if price is not None:
            print(f"Fetched price for {ticker}: ${price}")
            return round(float(price), 2)
        else:
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
                print(f"Fetched closing price for {ticker}: ${price}")
                return round(float(price), 2)
            else:
                print(f"Could not find price data for {ticker} via info or history.")
                return f"Could not retrieve price for {ticker}."
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return f"Error retrieving price for {ticker}."


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    print(f"Performing DuckDuckGo search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            if not results:
                print("No results found from DuckDuckGo.")
                return "No relevant information found."
            combined_results = "\n".join(
                [
                    f"- {res.get('title', 'No Title')}: {res.get('body', 'No snippet.')}"
                    for res in results
                ]
            )
            print(f"DuckDuckGo search returned {len(results)} results.")
            return combined_results
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return "Error performing web search."


def validate_usd_result(text: str) -> bool:
    return "$" in text or "USD" in text.upper()


def process_text(text: str) -> str:
    text = re.sub(r"```(json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    suffix = "It’s always good to get advice from our professionals here at NRDX.com."
    text_stripped = text.strip()
    if not text_stripped.endswith(suffix):
        if text_stripped and text_stripped[-1] not in [".", "!", "?"]:
            text_stripped += "."
        text_stripped += f" {suffix}"
    return text_stripped


# --- OpenAI Tool Definitions ---
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


# needs_web_search function
async def needs_web_search(user_query: str) -> bool:
    # Add more explicit logging here
    print(f"Running needs_web_search classification for: '{user_query}'")
    # Basic check: if query is short and asks about memory, likely no search needed
    query_lower = user_query.lower()
    if len(query_lower.split()) < 10 and (
        "remember" in query_lower
        or "what was" in query_lower
        or "talked about" in query_lower
    ):
        print("Query seems like a recall question, skipping web search classification.")
        return False

    if not client:
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
            print(f"Web search classification result: '{result_text}'")
            return "true" in result_text
        else:
            print(
                "Warning: Could not parse classification response. Defaulting to False."
            )
            return False
    except Exception as e:
        print(f"Error during web search classification LLM call: {e}")
        return False


# handle_tool_calls function
async def handle_tool_calls(
    response_message,
    user_query: str,  # Keep for logging/context if needed, though messages_history is primary
    messages_history: list,  # The history that *led* to the tool call
) -> str:
    if not client:
        return "Error: OpenAI client not available to handle tool calls."
    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content or "Error: No tool calls found and no content."

    print(f"Handling {len(tool_calls)} tool call(s)...")
    # Start the history for the *follow-up* call with the history *up to and including* the assistant's request for tools
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
                if ticker:
                    price = get_stock_price(ticker)
                    result_content = f"The latest price for {ticker} is: {price}"
                else:
                    result_content = "Error: Ticker symbol missing for get_stock_price."
            elif function_name == "web_search":
                search_query_arg = args_dict.get("query")
                if search_query_arg:
                    current_date = datetime.now().strftime("%B %d, %Y")
                    search_query = f"{search_query_arg} in USD on {current_date}"
                    search_result_text = duckduckgo_search(search_query)
                    result_content = f"Web search result for '{search_query_arg}': {search_result_text}"
                else:
                    result_content = "Error: Search query missing for web_search."
            else:
                result_content = f"Error: Unknown function '{function_name}' called."
            print(
                f"Tool '{function_name}' executed. Result snippet: {result_content[:100]}..."
            )
        except json.JSONDecodeError:
            print(
                f"Error decoding arguments for tool {function_name}: {function_args_str}"
            )
            result_content = "Error: Invalid arguments format provided by LLM."
        except Exception as e:
            print(f"Error executing tool {function_name}: {e}")
            result_content = f"Error executing tool: {e}"

        # Append the tool result message for the follow-up call
        messages_for_follow_up.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": result_content,
            }
        )

    print("Making follow-up LLM call with tool results...")
    try:
        # Send the history including the assistant's request AND the tool results
        follow_up_response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages_for_follow_up
        )
        final_content = follow_up_response.choices[0].message.content
        print(
            f"Follow-up Response Content snippet: {final_content[:100] if final_content else 'None'}..."
        )
        return final_content or "Error: No content in follow-up response."
    except Exception as e:
        print(f"Error during follow-up LLM call: {e}")
        raw_results_text = "\n".join(
            [msg["content"] for msg in messages_for_follow_up if msg["role"] == "tool"]
        )
        return f"Could not get final summary due to API error. Tool results:\n{raw_results_text}"


# --- /api/chat Endpoint (FIXED Logic) ---
@app.post("/api/chat")
async def chat(query: QueryRequest, request: Request, response: Response):
    global redis_conn, client

    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")
    if not redis_conn:
        print("Warning: Redis not available. Proceeding without session memory.")

    user_query = query.message
    session_id = request.cookies.get("chatbotSessionId")
    loaded_history = []
    is_new_session = False

    print(
        f"\n--- New Request (Session: {session_id[-6:] if session_id else 'New'}) ---"
    )
    print(f"Received query: {user_query}")

    # --- Load History ---
    if redis_conn and session_id:
        try:
            history_json = await redis_conn.get(session_id)
            if history_json:
                loaded_history = json.loads(history_json)
                # Ensure loaded_history is a list of dicts
                if not isinstance(loaded_history, list):
                    print(
                        f"Warning: History loaded from Redis is not a list: {type(loaded_history)}. Resetting."
                    )
                    loaded_history = []
                    session_id = None  # Treat as new if data is corrupt
                else:
                    await redis_conn.expire(session_id, SESSION_TTL_SECONDS)
                    print(
                        f"Loaded {len(loaded_history)} messages from session {session_id[-6:]}."
                    )
            else:
                print(f"Session ID {session_id[-6:]} not found in Redis.")
                session_id = None
        except json.JSONDecodeError:
            print(f"Error decoding history for session {session_id}. Starting fresh.")
            session_id = None
        except Exception as e:
            print(f"Redis GET/EXPIRE error: {e}. Proceeding without history.")
            # Keep session_id but history is empty

    # --- Create New Session ID ---
    if redis_conn and not session_id:
        is_new_session = True
        session_id = str(uuid.uuid4())
        print(f"Generated new session ID: {session_id[-6:]}")
        loaded_history = []

    # --- Core Logic ---
    if not is_related_to_stocks_crypto(user_query):
        print("Query not related to stocks/crypto, returning restricted response")
        return {
            "response": "I can only answer questions about stocks, cryptocurrency, or trading."
        }

    final_response_content = ""
    raw_ai_response = None
    messages_sent_to_openai = (
        []
    )  # Track what was actually sent for tool handling context

    try:
        # Determine if search is needed (now less likely for recall questions)
        search_needed = await needs_web_search(user_query)

        system_prompt = "You are a financial assistant specializing in stocks, cryptocurrency, and trading. Use the conversation history provided. You must provide very clear and explicit answers in USD. If the user asks for a recommendation, give a direct 'You should...' statement. Use provided tools when necessary. Ensure all prices are presented in USD."

        # Always start constructing the message list with system prompt and history
        base_messages = [{"role": "system", "content": system_prompt}] + loaded_history
        # The user's current raw message
        current_user_message_dict = {"role": "user", "content": user_query}

        if search_needed:
            print("Web search determined to be needed...")
            # ... Perform search ...
            current_date = datetime.now().strftime("%B %d, %Y")
            search_query = f"{user_query} price in USD on {current_date}"
            search_result_text = duckduckgo_search(search_query)
            print(f"Web search result snippet: {search_result_text[:200]}...")
            if (
                not validate_usd_result(search_result_text)
                and "No relevant information found" not in search_result_text
            ):
                print("Refining search for USD...")
                refined_search_query = (
                    f"{user_query} price in US dollars on {current_date}"
                )
                search_result_text = duckduckgo_search(refined_search_query)
                print(f"Refined search snippet: {search_result_text[:200]}...")

            # --- MODIFIED: Combine query, history, and search results ---
            # Create a modified user message content that includes context for the AI
            contextual_prompt_content = f"""Based on our previous conversation history AND the following recent web search results, please answer the user's latest query: "{user_query}"

Web Search Results:
---
{search_result_text}
---

Your concise answer:"""
            # Create the message dictionary with the modified content
            contextual_user_message_dict = {
                "role": "user",
                "content": contextual_prompt_content,
            }
            # Prepare the final list for the API call
            messages_sent_to_openai = base_messages + [contextual_user_message_dict]

            print("Making LLM call with history AND search results context...")
            openai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_sent_to_openai,
                tools=available_tools,
                tool_choice="auto",
            )

        else:
            # --- Web search not needed, send history + current query ---
            print(
                f"Web search not needed. Making LLM call with history ({len(loaded_history)} msgs)..."
            )
            # Prepare the final list for the API call
            messages_sent_to_openai = base_messages + [current_user_message_dict]
            openai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_sent_to_openai,
                tools=available_tools,
                tool_choice="auto",
            )

        # --- Process OpenAI response (Tools or Direct) ---
        response_message = openai_response.choices[0].message

        if response_message.tool_calls:
            print(f"Tool call(s) requested...")
            # Pass the *actual messages sent* to handle_tool_calls for context
            raw_ai_response = await handle_tool_calls(
                response_message, user_query, messages_sent_to_openai
            )
        else:
            raw_ai_response = response_message.content
            print(
                f"Direct text response extracted snippet: {raw_ai_response[:100] if raw_ai_response else 'None'}..."
            )

        # --- Format Final Response ---
        if not raw_ai_response:
            print("Error: No final content generated.")
            final_response_content = "I encountered an issue processing your request."
            raw_ai_response = None
        else:
            final_response_content = process_text(raw_ai_response)
            print(f"Returning formatted response: {final_response_content}")

    # --- Error Handling ---
    except BadRequestError as bre:
        print(f"OpenAI API Bad Request Error: {bre}")
        raw_ai_response = None
        raise HTTPException(
            status_code=400,
            detail=f"API Error: {bre.body.get('message', 'Bad Request')}",
        )
    except HTTPException as http_exc:
        raw_ai_response = None
        raise http_exc
    except Exception as e:
        print(f"An critical error occurred: {e}")
        traceback.print_exc()
        raw_ai_response = None
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )

    # --- Save History ---
    # Only save if successful and we have the necessary components
    if redis_conn and session_id and raw_ai_response:
        try:
            # Use the ORIGINAL user message dict for saving, not the contextual one
            new_history_entry = [
                current_user_message_dict,
                {"role": "assistant", "content": raw_ai_response},
            ]
            updated_history = loaded_history + new_history_entry
            # Truncate
            if len(updated_history) > MAX_HISTORY_MESSAGES:
                updated_history = updated_history[-MAX_HISTORY_MESSAGES:]
                print(f"History truncated to last {len(updated_history)} messages.")
            # Save
            history_to_save_json = json.dumps(updated_history)
            await redis_conn.set(
                session_id, history_to_save_json, ex=SESSION_TTL_SECONDS
            )
            print(
                f"Saved history ({len(updated_history)} msgs) to session {session_id[-6:]}"
            )
        except Exception as e:
            print(f"Redis SET error: {e}. History not saved.")

    # --- Set Cookie ---
    if is_new_session and session_id and redis_conn:
        response.set_cookie(
            key="chatbotSessionId",
            value=session_id,
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            samesite="Lax",
            path="/",
            # , secure=True # UNCOMMENT FOR HTTPS
        )
        print(f"Set session cookie for new session {session_id[-6:]}")

    return {"response": final_response_content}


# --- Health Check Endpoint ---
@app.get("/api/health")
async def health_check():
    global redis_conn
    redis_status = "not_initialized"
    if redis_conn:
        try:
            await redis_conn.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "error_connecting"
    return {
        "status": "OK_V2_fixed",
        "redis_status": redis_status,
    }  # Added _fixed to status for testing deploy


# --- Optional: To Run Directly ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
