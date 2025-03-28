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
from contextlib import asynccontextmanager  # Import for lifespan


# --- Configuration & Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Redis Configuration ---
REDIS_HOST = os.getenv(
    "REDIS_HOST", "localhost"
)  # Allow overriding host via .env if needed
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")  # Read password from .env

# Construct REDIS_URL dynamically
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
    print("Redis configured WITH password.")
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    print(
        "Redis configured WITHOUT password (ensure this is intentional for production)."
    )
# --- End of Redis Configuration Update ---


SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours
MAX_HISTORY_PAIRS = 10  # Keep last 10 Q&A pairs
MAX_HISTORY_MESSAGES = MAX_HISTORY_PAIRS * 2  # Total messages

# --- Initialize Redis Connection Variable ---
# We will connect/disconnect within the lifespan manager
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


# --- NEW: Lifespan Context Manager for Redis ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_conn  # Declare we are modifying the global variable
    print("Application startup: Initializing Redis connection...")
    try:
        redis_conn = redis.Redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True
        )
        # Optional: Test connection on startup
        await redis_conn.ping()
        print("Successfully connected to Redis during startup.")
    except Exception as e:
        print(f"Startup Error: Could not connect to Redis: {e}")
        redis_conn = None  # Ensure it's None if connection fails

    yield  # The application runs while yielding

    print("Application shutdown: Closing Redis connection...")
    if redis_conn:
        await redis_conn.close()
        print("Redis connection closed.")
    print("Application shutdown complete.")


# Initialize FastAPI app with the lifespan manager
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
        # Core Finance/Trading
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
        # Broader Business/Company Terms
        "company",
        "companies",
        "business",
        "corporation",
        "industry",
        "sector",
        "operations",
        # Specific Known Companies (Examples - add more as needed)
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
        # Specific Sectors (Examples - add more as needed)
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
        # Try common attributes first, then fall back to history
        price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
        if price is not None:
            print(f"Fetched price for {ticker}: ${price}")
            return round(float(price), 2)
        else:
            # If common attributes are missing, try recent history
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
            # Format results nicely
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
    # Remove markdown code blocks (json or otherwise)
    text = re.sub(r"```(json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    suffix = "It’s always good to get advice from our professionals here at NRDX.com."
    text_stripped = text.strip()
    # Add the suffix only if it's not already there
    if not text_stripped.endswith(suffix):
        # Add punctuation if needed before the suffix
        if text_stripped and text_stripped[-1] not in [".", "!", "?"]:
            text_stripped += "."
        text_stripped += f" {suffix}"
    return text_stripped


# --- OpenAI Tool Definitions --- (Included as requested)
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


# needs_web_search function (Keep as is)
async def needs_web_search(user_query: str) -> bool:
    if not client:
        print("OpenAI client not available for web search classification.")
        return False

    print(f"Classifying query for web search need: '{user_query}'")
    try:
        classification_messages = [
            {
                "role": "system",
                "content": "Analyze the user query. Does it require searching the web for current events, real-time data (like exact current prices not covered by tools), or very recent information published today or within the last few days? Answer only with 'True' or 'False'.",
            },
            {
                "role": "user",
                "content": f'User Query: "{user_query}"\n\nRequires Web Search (True/False):',
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Specify a fast, cheap model for classification
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
            # Be slightly more robust in checking the result
            return "true" in result_text
        else:
            print(
                "Warning: Could not parse classification response. Defaulting to False."
            )
            print(f"Raw classification response: {response}")
            return False
    except Exception as e:
        print(f"Error during web search classification LLM call: {e}")
        return False


# handle_tool_calls function (Keep as is)
async def handle_tool_calls(
    response_message,
    user_query: str,
    messages_history: list,
) -> str:
    if not client:
        return "Error: OpenAI client not available to handle tool calls."

    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content or "Error: No tool calls found and no content."

    print(f"Handling {len(tool_calls)} tool call(s)...")
    messages_with_tool_requests = messages_history + [response_message]
    messages_for_follow_up = (
        messages_with_tool_requests  # Start with previous history + assistant's request
    )

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

        # Append the tool result message
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
        follow_up_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_follow_up,  # Send the history including tool results
        )
        final_content = follow_up_response.choices[0].message.content
        print(
            f"Follow-up Response Content snippet: {final_content[:100] if final_content else 'None'}..."
        )
        return final_content or "Error: No content in follow-up response."
    except Exception as e:
        print(f"Error during follow-up LLM call: {e}")
        # Fallback: Provide raw tool results if summary fails
        raw_results_text = "\n".join(
            [msg["content"] for msg in messages_for_follow_up if msg["role"] == "tool"]
        )
        return f"Could not get final summary due to API error. Tool results:\n{raw_results_text}"


# --- /api/chat Endpoint (Using Redis Session) ---
@app.post("/api/chat")
async def chat(query: QueryRequest, request: Request, response: Response):
    # Use the global redis_conn initialized by lifespan
    global redis_conn, client

    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")
    if not redis_conn:
        # Decide how to handle Redis being unavailable during a request
        print(
            "Warning: Redis not available for this request. Proceeding without session memory."
        )
        # Optionally: raise HTTPException(status_code=503, detail="Session service unavailable")

    user_query = query.message
    session_id = request.cookies.get("chatbotSessionId")
    loaded_history = []
    is_new_session = False

    print(
        f"\n--- New Request (Session: {session_id[-6:] if session_id else 'New'}) ---"
    )
    print(f"Received query: {user_query}")

    # --- Load History from Redis (if available) ---
    if redis_conn and session_id:
        try:
            history_json = await redis_conn.get(session_id)
            if history_json:
                loaded_history = json.loads(history_json)
                await redis_conn.expire(session_id, SESSION_TTL_SECONDS)  # Reset TTL
                print(
                    f"Loaded {len(loaded_history)} messages from session {session_id[-6:]}. TTL reset."
                )
            else:
                print(f"Session ID {session_id[-6:]} not found in Redis.")
                session_id = None  # Treat as new
        except json.JSONDecodeError:
            print(f"Error decoding history for session {session_id}. Starting fresh.")
            session_id = None  # Treat as new
        except Exception as e:
            print(f"Redis GET/EXPIRE error: {e}. Proceeding without history.")
            # Keep session_id if it existed, but history will be empty

    # --- Create New Session ID if Needed ---
    if redis_conn and not session_id:  # Only create if Redis is available
        is_new_session = True
        session_id = str(uuid.uuid4())
        print(f"Generated new session ID: {session_id[-6:]}")
        loaded_history = []  # Ensure history is empty for new session

    # --- Core Logic ---
    if not is_related_to_stocks_crypto(user_query):
        print("Query not related to stocks/crypto, returning restricted response")
        return {
            "response": "I can only answer questions about stocks, cryptocurrency, or trading."
        }

    final_response_content = ""
    raw_ai_response = None  # Store response before formatting for history

    try:
        search_needed = await needs_web_search(user_query)
        system_prompt = "You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers in USD. If the user asks for a recommendation, give a direct 'You should...' statement. Use provided tools when necessary. Ensure all prices are presented in USD."

        openai_response = None
        current_user_message = {"role": "user", "content": user_query}
        messages_for_api = (
            [{"role": "system", "content": system_prompt}]
            + loaded_history
            + [current_user_message]
        )

        if search_needed:
            print("Web search needed. Bypassing memory for summarization call.")
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

            summarization_prompt_content = f"""Use the following web search results to answer the user's query: "{user_query}". Summarize the relevant information concisely, ensure the final answer is in USD, and respond directly to the user's question.

Web Search Results:
---
{search_result_text}
---

Your concise answer based *only* on the provided search results and user query:"""
            messages_for_summarization = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summarization_prompt_content},
            ]
            print("Making LLM call to summarize web search...")
            openai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_summarization,
                tools=available_tools,
                tool_choice="auto",
            )
            messages_for_api = (
                messages_for_summarization  # Track what was actually sent for this turn
            )

        else:
            print(
                f"Web search not needed. Making LLM call with history ({len(loaded_history)} msgs)..."
            )
            openai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_api,
                tools=available_tools,
                tool_choice="auto",
            )

        response_message = openai_response.choices[0].message

        if response_message.tool_calls:
            print(f"Tool call(s) requested...")
            raw_ai_response = await handle_tool_calls(
                response_message, user_query, messages_for_api
            )
        else:
            raw_ai_response = response_message.content
            print(
                f"Direct text response extracted snippet: {raw_ai_response[:100] if raw_ai_response else 'None'}..."
            )

        if not raw_ai_response:
            print("Error: No final content generated.")
            final_response_content = (
                "I encountered an issue processing your request. Please try again."
            )
            raw_ai_response = None  # Prevent saving this error state
        else:
            final_response_content = process_text(
                raw_ai_response
            )  # Format for user display
            print(f"Returning formatted response: {final_response_content}")

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
        print(f"An critical error occurred in the chat endpoint: {e}")
        import traceback

        traceback.print_exc()
        raw_ai_response = None
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )

    # --- Save History to Redis (if appropriate) ---
    if (
        redis_conn and session_id and raw_ai_response
    ):  # Only if Redis works, session exists, and AI succeeded
        try:
            new_history_entry = [
                current_user_message,
                {"role": "assistant", "content": raw_ai_response},
            ]
            updated_history = loaded_history + new_history_entry
            # Truncate history
            if len(updated_history) > MAX_HISTORY_MESSAGES:
                updated_history = updated_history[-MAX_HISTORY_MESSAGES:]
                print(f"History truncated to last {len(updated_history)} messages.")

            history_to_save_json = json.dumps(updated_history)
            await redis_conn.set(
                session_id, history_to_save_json, ex=SESSION_TTL_SECONDS
            )
            print(
                f"Saved updated history ({len(updated_history)} messages) to session {session_id[-6:]}"
            )
        except Exception as e:
            print(f"Redis SET error: {e}. History not saved.")

    # --- Set Cookie If New Session ---
    if (
        is_new_session and session_id and redis_conn
    ):  # Only set cookie if session was successfully created with Redis
        response.set_cookie(
            key="chatbotSessionId",
            value=session_id,
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            samesite="Lax",
            path="/",
            # secure=True, # UNCOMMENT FOR PRODUCTION HTTPS
        )
        print(f"Set session cookie for new session {session_id[-6:]}")

    return {"response": final_response_content}


# --- Health Check Endpoint ---
@app.get("/api/health")
async def health_check():
    global redis_conn  # Access global redis_conn
    redis_status = "not_initialized"
    if redis_conn:
        try:
            await redis_conn.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "error_connecting"
    return {"status": "OK_V2", "redis_status": redis_status}


# --- Optional: To Run Directly ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
