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
import time
from langfuse import Langfuse
from langfuse.decorators import observe

# --- Configuration & Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Langfuse Configuration (Read from .env) ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

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
    app.state.langfuse = None
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

    # Initialize Langfuse client
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            app.state.langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            print("Lifespan: Langfuse client created.")
        except Exception as e:
            print(f"Lifespan Startup Error: Could not initialize Langfuse client: {e}")
            app.state.langfuse = None
    else:
        print("Lifespan Warning: Langfuse keys not found.")

    yield

    print("Lifespan: Shutting down resources...")
    if hasattr(app.state, "redis_conn") and app.state.redis_conn:
        await app.state.redis_conn.close()
        print("Lifespan: Redis connection closed.")

    # Flush all Langfuse events before shutdown
    if hasattr(app.state, "langfuse") and app.state.langfuse:
        app.state.langfuse.flush()
        print("Lifespan: Langfuse events flushed.")

    print("Lifespan: Shutdown complete.")


app = FastAPI(lifespan=lifespan)

# --- CORS configuration (Specific origins for credentialed requests) ---
origins = [
    "http://localhost:5173",  # Local dev frontend
    "https://nextaisolutions.cloud",  # My tetsing FRONTEND
    "https://trade.dev-worldcapital1.com",  # Naor allowed domain
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
async def is_related_to_stocks_crypto(
    query: str, client: OpenAI | None, langfuse: Langfuse | None = None
) -> bool:
    """
    Determines if the query is related to stocks, cryptocurrency, or trading using OpenAI.
    Returns True if related, False otherwise.
    """
    if not client:
        print("OpenAI client not available for stock/crypto classification.")
        return False

    print(f"Classifying if query is related to stocks/crypto: '{query}'")

    # Start Langfuse trace if available
    trace = None
    span = None
    if langfuse:
        trace_id = str(uuid.uuid4())
        trace = langfuse.trace(
            id=trace_id, name="stock_crypto_classification", metadata={"query": query}
        )
        span = trace.span(
            name="classify_stocks_crypto",
            metadata={
                "status": f"Classifying if query is related to stocks/crypto: '{query}'"
            },
        )

    start_time = time.time()
    try:
        classification_messages = [
            {
                "role": "system",
                "content": """Analyze the user query. Is it related to stocks, cryptocurrency, trading, investing, or financial markets?
                
                Consider semantic similarity and not just exact matches. For example:
                - Questions about technical indicators (RSI, MACD, etc.) are related
                - Questions about market analysis, charts, or trading strategies are related
                - Questions about financial instruments, brokers, or trading platforms are related
                - Questions about economic indicators that affect markets are related
                
                If yes, respond with 'True'.
                If no, respond with 'False'.
                
                Answer only with 'True' or 'False'.""",
            },
            {
                "role": "user",
                "content": f'User Query: "{query}"\n\nIs this query related to stocks, cryptocurrency, or trading (True/False):',
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=classification_messages,
            max_tokens=5,
            temperature=0.0,
        )

        # Calculate latency
        latency = time.time() - start_time

        # Parse the response content
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip().lower()
            result = "true" in result_text
            print(f"Stock/crypto classification result: '{result_text}'")

            # Log the result in Langfuse
            if span:
                span.end(
                    output=result_text,
                    metadata={
                        "model": "gpt-4o-mini",
                        "latency": latency,
                        "result": result,
                        "status": f"Stock/crypto classification result: '{result_text}'",
                    },
                )

            return result
        else:
            print(
                "Warning: Could not parse classification response. Defaulting to False."
            )
            if span:
                span.end(
                    output=None,
                    metadata={
                        "model": "gpt-4o-mini",
                        "latency": latency,
                        "error": "Could not parse classification response",
                        "status": "Warning: Could not parse classification response. Defaulting to False.",
                    },
                )
            return False

    except Exception as e:
        error_msg = f"Error during stock/crypto classification: {e}"
        print(error_msg)
        if span:
            span.end(
                output=None,
                metadata={
                    "model": "gpt-4o-mini",
                    "error": str(e),
                    "latency": time.time() - start_time,
                    "status": error_msg,
                },
            )
        return False


def get_stock_price(
    ticker: str, include_history: bool = False
) -> tuple[float | str, dict | None]:
    """
    Get the current price and optionally historical data for a ticker.
    Returns a tuple of (current_price, historical_data).
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))

        historical_data = None
        if include_history:
            # Get 5-day history for weekly context
            hist = stock.history(period="5d")
            if not hist.empty:
                historical_data = {
                    "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                    "open": hist["Open"].tolist(),
                    "high": hist["High"].tolist(),
                    "low": hist["Low"].tolist(),
                    "close": hist["Close"].tolist(),
                    "volume": (
                        hist["Volume"].tolist() if "Volume" in hist.columns else None
                    ),
                }

        if price is not None:
            return round(float(price), 2), historical_data

        # Fallback to historical data if current price not available
        if historical_data:
            return round(float(historical_data["close"][-1]), 2), historical_data

        hist = stock.history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2), None

        return f"Could not retrieve price for {ticker}.", None
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return f"Error retrieving price for {ticker}.", None


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    print(f"DDG Search: '{query}'")
    time_period_used = "unknown"
    try:
        with DDGS() as ddgs:
            # First try to get results from the last week
            results = [
                r
                for r in ddgs.text(
                    query, max_results=max_results, timelimit="w"  # 'w' for week
                )
            ]
            time_period_used = "last week"

            # If not enough results, try the last month
            if len(results) < 2:
                print("Not enough results from the last week, trying the last month...")
                results = [
                    r
                    for r in ddgs.text(
                        query, max_results=max_results, timelimit="m"  # 'm' for month
                    )
                ]
                time_period_used = "last month"

                # If still not enough results, try the last year
                if len(results) < 2:
                    print(
                        "Not enough results from the last month, trying the last year..."
                    )
                    results = [
                        r
                        for r in ddgs.text(
                            query,
                            max_results=max_results,
                            timelimit="y",  # 'y' for year
                        )
                    ]
                    time_period_used = "last year"

            if not results:
                return "No relevant information found."

            # Format results with dates if available
            formatted_results = []
            for res in results:
                title = res.get("title", "No Title")
                body = res.get("body", "No snippet.")
                date = res.get("date", "")

                # Add date to the result if available
                if date:
                    formatted_results.append(f"- {title} ({date}): {body}")
                else:
                    formatted_results.append(f"- {title}: {body}")

            print(
                f"DEBUG: Using search results from the {time_period_used} ({len(results)} results found)"
            )
            return "\n".join(formatted_results)
    except Exception as e:
        print(f"DDG Error: {e}")
        return "Error performing web search."


def validate_usd_result(text: str) -> bool:
    return "$" in text or "USD" in text.upper()


def process_text(text: str) -> str:
    text = re.sub(r"```(json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    text_stripped = text.strip()
    if text_stripped and text_stripped[-1] not in ".!?":
        text_stripped += "."
    return text_stripped


# --- Q&A File Functions ---
def load_qa_file() -> dict:
    """Load the Q&A file and return a dictionary of questions and answers."""
    qa_dict = {}
    try:
        with open("qna_output.txt", "r", encoding="utf-8") as file:
            content = file.read()
            # Split by Q: to get individual Q&A pairs
            qa_pairs = content.split("Q: ")
            for pair in qa_pairs[1:]:  # Skip the first empty element
                if "A: " in pair:
                    question, answer = pair.split("A: ", 1)
                    # Clean up the question and answer
                    question = question.strip()
                    answer = answer.strip()
                    qa_dict[question] = answer
        print(f"Loaded {len(qa_dict)} Q&A pairs from file.")
        return qa_dict
    except Exception as e:
        print(f"Error loading Q&A file: {e}")
        return {}


async def find_qa_match(
    user_query: str, client: OpenAI | None, langfuse: Langfuse | None = None
) -> tuple[bool, str]:
    """
    Check if the user query matches a question in the Q&A file.
    Returns a tuple of (is_match, answer_or_empty_string)
    """
    if not client:
        print("OpenAI client not available for Q&A matching.")
        return (False, "")

    # Load the Q&A file
    qa_dict = load_qa_file()
    if not qa_dict:
        print("Q&A dictionary is empty, cannot find match.")
        return (False, "")

    print(f"Checking if query matches Q&A file: '{user_query}'")

    # Start Langfuse trace if available
    trace = None
    span = None
    if langfuse:
        trace_id = str(uuid.uuid4())
        trace = langfuse.trace(
            id=trace_id, name="qa_match", metadata={"query": user_query}
        )
        span = trace.span(
            name="find_qa_match",
            metadata={"status": f"Checking if query matches Q&A file: '{user_query}'"},
        )

    start_time = time.time()
    try:
        # Create a prompt for the LLM to find the best match
        qa_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_dict.items()])

        classification_messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that matches user questions to predefined Q&A pairs.
                Analyze the user query and determine if it matches any of the questions in the provided Q&A pairs.
                If there's a match, respond with the exact answer from the Q&A pair.
                If there's no match, respond with 'NO_MATCH'.
                
                Consider semantic similarity, not just exact matches. The user might phrase the question differently
                but be asking about the same topic.""",
            },
            {
                "role": "user",
                "content": f"""User Query: "{user_query}"

Q&A Pairs:
{qa_text}

If there's a match, provide ONLY the exact answer from the matching Q&A pair.
If there's no match, respond with ONLY 'NO_MATCH'.""",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=classification_messages,
            temperature=0.0,
        )

        # Calculate latency
        latency = time.time() - start_time

        # Parse the response content
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip()

            if result_text == "NO_MATCH":
                print("No Q&A match found for the query.")
                if span:
                    span.end(
                        output="NO_MATCH",
                        metadata={
                            "model": "gpt-4o-mini",
                            "latency": latency,
                            "result": False,
                            "status": "No Q&A match found for the query.",
                        },
                    )
                return (False, "")
            else:
                # Remove "A: " prefix if it exists
                if result_text.startswith("A: "):
                    result_text = result_text[3:].strip()

                print("Q&A match found for the query.")
                if span:
                    span.end(
                        output=result_text,
                        metadata={
                            "model": "gpt-4o-mini",
                            "latency": latency,
                            "result": True,
                            "status": "Q&A match found for the query.",
                        },
                    )
                return (True, result_text)
        else:
            print(
                "Warning: Could not parse Q&A matching response. Defaulting to no match."
            )
            if span:
                span.end(
                    output=None,
                    metadata={
                        "model": "gpt-4o-mini",
                        "latency": latency,
                        "error": "Could not parse Q&A matching response",
                        "status": "Warning: Could not parse Q&A matching response. Defaulting to no match.",
                    },
                )
            return (False, "")

    except Exception as e:
        error_msg = f"Error during Q&A matching: {e}"
        print(error_msg)
        if span:
            span.end(
                output=None,
                metadata={
                    "model": "gpt-4o-mini",
                    "error": str(e),
                    "latency": time.time() - start_time,
                    "status": error_msg,
                },
            )
        return (False, "")


# --- New Function: Check if query is about current stock price ---
async def is_stock_price_query(
    user_query: str, client: OpenAI | None, langfuse: Langfuse | None = None
) -> tuple[bool, str, bool]:
    """
    Determines if the query is specifically about current stock price and extracts the ticker.
    Returns a tuple of (is_price_query, ticker_symbol_or_empty_string, needs_market_context)
    """
    if not client:
        print("OpenAI client not available for stock price query classification.")
        return (False, "", False)

    print(f"Classifying if query is about current stock price: '{user_query}'")

    # Start Langfuse trace if available
    trace = None
    span = None
    if langfuse:
        trace_id = str(uuid.uuid4())
        trace = langfuse.trace(
            id=trace_id,
            name="stock_price_query_classification",
            metadata={"query": user_query},
        )
        span = trace.span(
            name="classify_stock_price_query",
            metadata={
                "status": f"Classifying if query is about current stock price: '{user_query}'"
            },
        )

    start_time = time.time()
    try:
        classification_messages = [
            {
                "role": "system",
                "content": """Analyze the user query. Is it specifically asking about the current or latest price of a stock, cryptocurrency, market index, commodity, or forex pair? 
                If yes, respond with 'True' followed by the ticker symbol in JSON format like: {"is_price_query": true, "ticker": "AAPL", "needs_market_context": false}
                If no, respond with 'False' in JSON format like: {"is_price_query": false, "ticker": "", "needs_market_context": false}
                
                Also determine if the query needs additional market context (like recent price movements, trends, or explanations for price changes).
                Set "needs_market_context" to true if the query asks about:
                - Price changes or movements (e.g., "did the S&P 500 decline this week?")
                - Reasons for price changes (e.g., "why did Bitcoin drop?")
                - Trends or patterns (e.g., "is gold trending up?")
                - Comparisons (e.g., "how does AAPL compare to MSFT?")
                
                Common ticker symbols:
                - Market indices: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ), ^RUT (Russell 2000), ^VIX (VIX), ^FTSE (FTSE 100), ^GDAXI (DAX), ^N225 (Nikkei 225)
                - Commodities: GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), NG=F (Natural Gas)
                - Currencies: EURUSD=X (EUR/USD), GBPUSD=X (GBP/USD), USDJPY=X (USD/JPY)
                - Treasury Yields: ^TNX (10-Year), ^FVX (5-Year), ^TYX (30-Year)
                - Cryptocurrencies: BTC-USD, ETH-USD, etc.
                
                If the user mentions a market index, commodity, forex pair, or cryptocurrency by name, extract the appropriate ticker symbol.""",
            },
            {
                "role": "user",
                "content": f'User Query: "{user_query}"',
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=classification_messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        # Calculate latency
        latency = time.time() - start_time

        # Parse the response content
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip()
            try:
                result_json = json.loads(result_text)
                is_price_query = result_json.get("is_price_query", False)
                ticker = result_json.get("ticker", "").strip().upper()
                needs_market_context = result_json.get("needs_market_context", False)

                print(
                    f"Stock price query classification: is_price_query={is_price_query}, ticker={ticker}, needs_market_context={needs_market_context}"
                )

                if span:
                    span.end(
                        output=result_text,
                        metadata={
                            "model": "gpt-4o-mini",
                            "latency": latency,
                            "result": {
                                "is_price_query": is_price_query,
                                "ticker": ticker,
                                "needs_market_context": needs_market_context,
                            },
                            "status": f"Stock price query classification: is_price_query={is_price_query}, ticker={ticker}, needs_market_context={needs_market_context}",
                        },
                    )

                return (is_price_query, ticker, needs_market_context)
            except json.JSONDecodeError:
                error_msg = f"Warning: Could not parse JSON response: {result_text}"
                print(error_msg)
                if span:
                    span.end(
                        output=result_text,
                        metadata={
                            "model": "gpt-4o-mini",
                            "latency": latency,
                            "error": "JSON decode error",
                            "status": error_msg,
                        },
                    )
                return (False, "", False)
        else:
            print(
                "Warning: Could not parse classification response. Defaulting to False."
            )
            if span:
                span.end(
                    output=None,
                    metadata={
                        "model": "gpt-4o-mini",
                        "latency": latency,
                        "error": "Could not parse classification response",
                        "status": "Warning: Could not parse classification response. Defaulting to False.",
                    },
                )
            return (False, "", False)

    except Exception as e:
        error_msg = f"Error during stock price query classification: {e}"
        print(error_msg)
        if span:
            span.end(
                output=None,
                metadata={
                    "model": "gpt-4o-mini",
                    "error": str(e),
                    "latency": time.time() - start_time,
                    "status": error_msg,
                },
            )
        return (False, "", False)


# --- Handle direct stock price queries ---
async def handle_stock_price_query(
    ticker: str,
    user_query: str,
    client: OpenAI | None,
    needs_market_context: bool = False,
    langfuse: Langfuse | None = None,
) -> str:
    """Handles direct stock price queries using Yahoo Finance."""
    print(
        f"Handling direct stock price query for ticker: {ticker}, needs_market_context: {needs_market_context}"
    )

    # Start Langfuse trace if available
    trace = None
    span = None
    if langfuse:
        trace_id = str(uuid.uuid4())
        trace = langfuse.trace(
            id=trace_id,
            name="stock_price_query_handler",
            metadata={
                "query": user_query,
                "ticker": ticker,
                "needs_market_context": needs_market_context,
            },
        )
        span = trace.span(
            name="handle_stock_price",
            metadata={
                "status": f"Handling direct stock price query for ticker: {ticker}, needs_market_context: {needs_market_context}"
            },
        )

    # Get price and historical data if market context is needed
    price, historical_data = get_stock_price(
        ticker, include_history=needs_market_context
    )

    # Determine the type of financial instrument
    instrument_type = "stock"
    if ticker.startswith("^"):
        instrument_type = "market index"
    elif ticker.endswith("=F"):
        instrument_type = "commodity"
    elif ticker.endswith("=X"):
        instrument_type = "forex pair"
    elif ticker.endswith("-USD"):
        instrument_type = "cryptocurrency"

    # Create system prompt for formatting the response
    system_prompt = f"""You are a financial assistant specializing in stocks, cryptocurrency, and trading. 
    Format the provided {instrument_type} price information into a clear, helpful response. 
    Ensure prices are presented in USD.
    
    If historical data is provided, analyze the price movements and trends to answer the user's question about price changes or reasons for those changes.
    If the user is asking about recent declines or increases, compare the current price to previous prices in the historical data.
    If the user is asking why prices changed, provide possible explanations based on the historical data and your knowledge of market factors."""

    # Prepare the user message with price and historical data
    user_message_content = f"Original query: {user_query}\n\nFinancial data: The latest price for {ticker} is: {price}"

    if historical_data:
        user_message_content += f"\n\nHistorical data (last 5 days):\n"
        for i in range(len(historical_data["dates"])):
            user_message_content += f"- {historical_data['dates'][i]}: Open: {historical_data['open'][i]}, High: {historical_data['high'][i]}, Low: {historical_data['low'][i]}, Close: {historical_data['close'][i]}\n"

    # If market context is needed but we don't have historical data, suggest web search
    if needs_market_context and not historical_data:
        user_message_content += "\n\nNote: To provide a complete answer about price movements or reasons for changes, additional market context from web search would be helpful."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_content},
    ]

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        # Calculate latency
        latency = time.time() - start_time

        final_content = response.choices[0].message.content

        if span:
            span.end(
                output=final_content,
                metadata={
                    "model": "gpt-4o-mini",
                    "latency": latency,
                    "price_data": {
                        "ticker": ticker,
                        "price": price,
                        "has_historical_data": historical_data is not None,
                    },
                },
            )

        return final_content
    except Exception as e:
        error_msg = f"Error formatting stock price response: {e}"
        print(error_msg)

        if span:
            span.end(
                output=None,
                metadata={
                    "model": "gpt-4o-mini",
                    "error": str(e),
                    "latency": time.time() - start_time,
                    "status": error_msg,
                },
            )

        return f"The latest price for {ticker} is: {price}"


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


async def needs_web_search(
    user_query: str, client: OpenAI | None, langfuse: Langfuse | None = None
) -> bool:
    print(f"Classifying web search need for: '{user_query}'")
    query_lower = user_query.lower()

    # Start Langfuse trace if available
    trace = None
    span = None
    if langfuse:
        trace_id = str(uuid.uuid4())
        trace = langfuse.trace(
            id=trace_id,
            name="web_search_need_classification",
            metadata={"query": user_query},
        )
        span = trace.span(
            name="classify_web_search_need",
            metadata={"status": f"Classifying web search need for: '{user_query}'"},
        )

    # Check for specific keywords that should always trigger web search
    web_search_keywords = [
        "today",
        "today's",
        "today is",
        "today was",
        "today will be",
        "good day",
        "bad day",
        "market sentiment",
        "market mood",
        "trading conditions",
        "market outlook",
        "market analysis",
        "should i trade",
        "should i buy",
        "should i sell",
        "is it a good time",
        "is this a good time",
        "is now a good time",
        "what's happening",
        "what is happening",
        "what happened",
        "latest news",
        "recent news",
        "breaking news",
        "market update",
        "market report",
        "market summary",
        "trading advice",
        "investment advice",
        "trading recommendation",
        "market forecast",
        "market prediction",
        "market trend",
        "why is",
        "why are",
        "why did",
        "why has",
        "why have",
        "how is",
        "how are",
        "how did",
        "how has",
        "how have",
        "what's going on",
        "what is going on",
        "what's the deal",
        "what's the situation",
        "what's the outlook",
        "what's the forecast",
        "what's the prediction",
        "what's the trend",
        "what's the sentiment",
        "what's the mood",
        "what's the condition",
        "what's the state",
        "what's the status",
    ]

    # Check if any of the web search keywords are in the query
    if any(keyword in query_lower for keyword in web_search_keywords):
        print(f"DEBUG: Query contains web search keyword, triggering web search.")
        if span:
            span.end(
                output=True,
                metadata={
                    "result": True,
                    "reason": "keyword_match",
                    "status": "DEBUG: Query contains web search keyword, triggering web search.",
                },
            )
        return True

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
        if span:
            span.end(
                output=False,
                metadata={
                    "result": False,
                    "reason": "recall_query",
                    "status": "DEBUG: Query classified as recall, skipping web search.",
                },
            )
        return False

    if not client:
        print("DEBUG: needs_web_search - OpenAI client is None, cannot classify.")
        if span:
            span.end(
                output=False,
                metadata={
                    "result": False,
                    "reason": "openai_client_missing",
                    "status": "DEBUG: needs_web_search - OpenAI client is None, cannot classify.",
                },
            )
        return False

    start_time = time.time()
    try:
        classification_messages = [
            {
                "role": "system",
                "content": """Analyze the user query. Does it require searching the web for current events (e.g., today's news), real-time data (like specific current stock prices not covered by tools), or very recent information published today or within the last few days? 

IMPORTANT: If the user is asking about:
- Whether today is a good day for trading
- Current market conditions or sentiment
- Trading advice or recommendations
- Why something happened in the market
- What's happening in the market today
- Market outlook or forecasts
- Recent market events or news

Then you MUST respond with 'True' as these questions require current information.

Do NOT say True if the user is asking about the conversation history or what was said before. Answer only with 'True' or 'False'.""",
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

        # Calculate latency
        latency = time.time() - start_time

        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip().lower()
            result = "true" in result_text
            print(f"DEBUG: Web search classification result from LLM: '{result_text}'")

            if span:
                span.end(
                    output=result_text,
                    metadata={
                        "model": "gpt-4o-mini",
                        "latency": latency,
                        "result": result,
                        "status": f"DEBUG: Web search classification result from LLM: '{result_text}'",
                    },
                )

            return result
        else:
            print(
                "DEBUG: Could not parse classification response. Defaulting to False."
            )
            if span:
                span.end(
                    output=None,
                    metadata={
                        "model": "gpt-4o-mini",
                        "latency": latency,
                        "error": "Could not parse classification response",
                        "status": "DEBUG: Could not parse classification response. Defaulting to False.",
                    },
                )
            return False
    except Exception as e:
        error_msg = f"DEBUG: Error during classification LLM call: {e}"
        print(error_msg)
        if span:
            span.end(
                output=None,
                metadata={
                    "model": "gpt-4o-mini",
                    "error": str(e),
                    "latency": time.time() - start_time,
                    "status": error_msg,
                },
            )
        return False


async def handle_tool_calls(
    response_message,
    user_query: str,
    messages_history: list,
    client: OpenAI | None,
    langfuse: Langfuse | None = None,
) -> str:
    if not client:
        return "Error: OpenAI client not available."
    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content or "Error: No tool calls or content."

    print(f"DEBUG: Handling {len(tool_calls)} tool call(s)...")

    # Start Langfuse trace if available
    trace = None
    span = None
    if langfuse:
        trace_id = str(uuid.uuid4())
        trace = langfuse.trace(
            id=trace_id,
            name="handle_tool_calls",
            metadata={
                "query": user_query,
                "tool_calls_count": len(tool_calls),
                "tool_call_ids": [tc.id for tc in tool_calls],
            },
        )
        span = trace.span(
            name="process_tool_calls",
            metadata={"status": f"DEBUG: Handling {len(tool_calls)} tool call(s)..."},
        )

    messages_for_follow_up = messages_history + [response_message]

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        result_content = ""

        # Create a span for each tool call
        tool_span = None
        if span:
            tool_span = span.span(
                name=f"tool_call_{function_name}",
                metadata={
                    "tool_call_id": tool_call_id,
                    "function_name": function_name,
                    "function_args": function_args_str,
                },
            )

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

            if tool_span:
                tool_span.end(
                    output=result_content[:200]
                    + ("..." if len(result_content) > 200 else ""),
                    metadata={
                        "status": f"DEBUG: Tool '{function_name}' executed successfully."
                    },
                )

        except Exception as e:
            error_msg = f"DEBUG: Error executing tool {function_name}: {e}"
            print(error_msg)
            result_content = f"Error executing tool: {e}"

            if tool_span:
                tool_span.end(
                    output=None, metadata={"error": str(e), "status": error_msg}
                )

        messages_for_follow_up.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": result_content,
            }
        )

    print("DEBUG: Making follow-up LLM call with tool results...")
    follow_up_span = None
    if span:
        follow_up_span = span.span(
            name="follow_up_llm_call",
            metadata={
                "status": "DEBUG: Making follow-up LLM call with tool results..."
            },
        )

    start_time = time.time()
    try:
        # Add a system message to emphasize the importance of current information
        follow_up_messages = [
            {
                "role": "system",
                "content": "You are a financial assistant. When providing information from web search results, always emphasize how current the information is. If the information is outdated (more than a few months old), clearly state that it may not reflect current market conditions. Always prioritize the most recent information available.",
            }
        ] + messages_for_follow_up

        follow_up_response = client.chat.completions.create(
            model="gpt-4o-mini", messages=follow_up_messages
        )

        # Calculate latency
        latency = time.time() - start_time

        final_content = follow_up_response.choices[0].message.content
        print(
            f"DEBUG: Follow-up Response snippet: {final_content[:50] if final_content else 'None'}..."
        )

        if follow_up_span:
            follow_up_span.end(
                output=final_content[:200]
                + ("..." if final_content and len(final_content) > 200 else ""),
                metadata={
                    "model": "gpt-4o-mini",
                    "latency": latency,
                    "status": f"DEBUG: Follow-up Response received successfully.",
                },
            )

        if span:
            span.end(
                output=final_content[:200]
                + ("..." if final_content and len(final_content) > 200 else ""),
                metadata={
                    "status": "Tool calls processed and follow-up response received successfully."
                },
            )

        return final_content or "Error: No content in follow-up."
    except Exception as e:
        error_msg = f"DEBUG: Error during follow-up LLM call: {e}"
        print(error_msg)

        if follow_up_span:
            follow_up_span.end(
                output=None,
                metadata={
                    "model": "gpt-4o-mini",
                    "error": str(e),
                    "latency": time.time() - start_time,
                    "status": error_msg,
                },
            )

        if span:
            span.end(output=None, metadata={"error": str(e), "status": error_msg})

        return f"Error summarizing tool results."


# --- /api/chat Endpoint (Uses app.state) ---
@app.post("/api/chat")
async def chat(query: QueryRequest, request: Request, response: Response):
    redis_conn = request.app.state.redis_conn
    client = request.app.state.openai_client
    langfuse = request.app.state.langfuse

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

    # Create main trace for the entire request
    main_trace = None
    chat_span = None
    if langfuse:
        trace_id = str(uuid.uuid4())
        main_trace = langfuse.trace(
            id=trace_id,
            name="chat_request",
            metadata={
                "query": user_query,
                "session_id": session_id,
                "is_new_session": session_id is None,
            },
            user_id=session_id,
        )
        chat_span = main_trace.span(
            name="process_chat_request",
            metadata={
                "status": f"Processing chat request for session {session_id[-6:] if session_id else 'NEW'}"
            },
        )

    # Load History
    if redis_conn and session_id:
        history_span = None
        if chat_span:
            history_span = chat_span.span(
                name="load_history",
                metadata={
                    "status": f"DEBUG: Attempting to load history for session {session_id[-6:]}..."
                },
            )

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

                    if history_span:
                        history_span.end(
                            output=None,
                            metadata={
                                "error": f"Corrupt history type: {type(loaded_history)}",
                                "status": "DEBUG: ERROR - Corrupt history type. Resetting.",
                            },
                        )
                else:
                    print(f"DEBUG: Successfully loaded {len(loaded_history)} messages.")
                    if loaded_history:
                        print(f"DEBUG: Last loaded msg: {loaded_history[-1]}")
                    await redis_conn.expire(session_id, SESSION_TTL_SECONDS)

                    if history_span:
                        history_span.end(
                            output=len(loaded_history),
                            metadata={
                                "history_size": len(loaded_history),
                                "status": f"DEBUG: Successfully loaded {len(loaded_history)} messages.",
                            },
                        )
            else:
                print(f"DEBUG: Session ID {session_id[-6:]} not found in Redis.")
                session_id = None

                if history_span:
                    history_span.end(
                        output=None,
                        metadata={
                            "error": "Session not found",
                            "status": f"DEBUG: Session ID {session_id[-6:]} not found in Redis.",
                        },
                    )
        except json.JSONDecodeError as json_err:
            print(
                f"DEBUG: ERROR - JSON Decode failed for session {session_id}: {json_err}. Resetting."
            )
            session_id = None

            if history_span:
                history_span.end(
                    output=None,
                    metadata={
                        "error": f"JSON Decode error: {str(json_err)}",
                        "status": f"DEBUG: ERROR - JSON Decode failed for session. Resetting.",
                    },
                )
        except Exception as e:
            print(
                f"DEBUG: ERROR - Redis GET/EXPIRE failed: {e}. Proceeding without history."
            )

            if history_span:
                history_span.end(
                    output=None,
                    metadata={
                        "error": f"Redis error: {str(e)}",
                        "status": f"DEBUG: ERROR - Redis GET/EXPIRE failed. Proceeding without history.",
                    },
                )

    # Create New Session ID
    if redis_conn and not session_id:
        is_new_session = True
        session_id = str(uuid.uuid4())
        print(f"DEBUG: Generated NEW session ID: {session_id[-6:]}")
        loaded_history = []

        if main_trace and session_id:
            # Update trace with new session ID
            main_trace.update(user_id=session_id)

    # Core Logic
    final_response_content = ""
    raw_ai_response = None
    messages_sent_to_openai = []

    # Define current_user_message_dict at the beginning to ensure it's always available
    current_user_message_dict = {"role": "user", "content": user_query}

    processing_span = None
    if chat_span:
        processing_span = chat_span.span(
            name="process_query", metadata={"status": "Starting query processing"}
        )

    try:
        # First check if the query matches a question in the Q&A file
        qa_match, qa_answer = await find_qa_match(user_query, client, langfuse)

        if qa_match:
            print(f"DEBUG: Q&A match found, using predefined answer.")
            raw_ai_response = qa_answer
            final_response_content = process_text(raw_ai_response)
            print(f"DEBUG: Returning Q&A answer: {final_response_content}")

            if processing_span:
                processing_span.end(
                    output=final_response_content,
                    metadata={
                        "response_type": "qa_match",
                        "status": "DEBUG: Q&A match found, using predefined answer.",
                    },
                )
        else:
            # If no Q&A match, check if the query is related to stocks/crypto
            stocks_related_span = None
            if processing_span:
                stocks_related_span = processing_span.span(
                    name="check_stocks_related",
                    metadata={
                        "status": "Checking if query is related to stocks/crypto"
                    },
                )

            if not await is_related_to_stocks_crypto(user_query, client, langfuse):
                print("Query not related. Returning restricted response.")

                if stocks_related_span:
                    stocks_related_span.end(
                        output=False,
                        metadata={
                            "result": False,
                            "status": "Query not related. Returning restricted response.",
                        },
                    )

                if processing_span:
                    processing_span.end(
                        output="I can only answer questions about stocks, cryptocurrency, or trading.",
                        metadata={
                            "response_type": "restricted",
                            "status": "Query not related to stocks/crypto/trading.",
                        },
                    )

                if chat_span:
                    chat_span.end(
                        output="I can only answer questions about stocks, cryptocurrency, or trading.",
                        metadata={
                            "session_id": session_id,
                            "response_type": "restricted",
                        },
                    )

                return {
                    "response": "I can only answer questions about stocks, cryptocurrency, or trading."
                }

            if stocks_related_span:
                stocks_related_span.end(
                    output=True,
                    metadata={
                        "result": True,
                        "status": "Query is related to stocks/crypto/trading.",
                    },
                )

            # First check if this is a direct stock price query
            stock_price_span = None
            if processing_span:
                stock_price_span = processing_span.span(
                    name="check_stock_price_query",
                    metadata={"status": "Checking if query is about stock price"},
                )

            is_price_query, ticker, needs_market_context = await is_stock_price_query(
                user_query, client, langfuse
            )

            if stock_price_span:
                stock_price_span.end(
                    output={
                        "is_price_query": is_price_query,
                        "ticker": ticker,
                        "needs_market_context": needs_market_context,
                    },
                    metadata={
                        "status": f"Stock price query check complete: is_price_query={is_price_query}, ticker={ticker}"
                    },
                )

            if is_price_query and ticker:
                print(f"DEBUG: Direct stock price query detected for ticker: {ticker}")
                raw_ai_response = await handle_stock_price_query(
                    ticker, user_query, client, needs_market_context, langfuse
                )
                final_response_content = process_text(raw_ai_response)
                print(
                    f"DEBUG: Returning formatted stock price response: {final_response_content}"
                )

                if processing_span:
                    processing_span.end(
                        output=final_response_content,
                        metadata={
                            "response_type": "stock_price",
                            "ticker": ticker,
                            "needs_market_context": needs_market_context,
                            "status": "DEBUG: Direct stock price query processed successfully.",
                        },
                    )
            else:
                # Continue with the original flow - check if web search is needed
                web_search_span = None
                if processing_span:
                    web_search_span = processing_span.span(
                        name="check_web_search_need",
                        metadata={"status": "Checking if web search is needed"},
                    )

                search_needed = await needs_web_search(user_query, client, langfuse)

                if web_search_span:
                    web_search_span.end(
                        output=search_needed,
                        metadata={
                            "result": search_needed,
                            "status": f"Web search need check complete: {search_needed}",
                        },
                    )

                system_prompt = "You are a financial assistant specializing in stocks, cryptocurrency, and trading. Use the conversation history provided. You must provide very clear and explicit answers in USD. If the user asks for a recommendation, give a direct 'You should...' statement. Use provided tools when necessary. Ensure all prices are presented in USD. Refer back to previous turns in the conversation if the user asks."

                base_messages = [
                    {"role": "system", "content": system_prompt}
                ] + loaded_history

                llm_call_span = None
                if processing_span:
                    llm_call_span = processing_span.span(
                        name="prepare_llm_call",
                        metadata={"status": "Preparing LLM call"},
                    )

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
                    messages_sent_to_openai = base_messages + [
                        contextual_user_message_dict
                    ]
                    print("DEBUG: Making LLM call with History + Search Results...")

                    if llm_call_span:
                        llm_call_span.end(
                            output=len(messages_sent_to_openai),
                            metadata={
                                "messages_count": len(messages_sent_to_openai),
                                "web_search_used": True,
                                "search_results_length": len(search_result_text),
                                "status": "DEBUG: Making LLM call with History + Search Results...",
                            },
                        )
                else:
                    print("DEBUG: Web search determined NOT needed.")
                    messages_sent_to_openai = base_messages + [
                        current_user_message_dict
                    ]
                    print("DEBUG: Making LLM call with History + Current Query...")

                    if llm_call_span:
                        llm_call_span.end(
                            output=len(messages_sent_to_openai),
                            metadata={
                                "messages_count": len(messages_sent_to_openai),
                                "web_search_used": False,
                                "status": "DEBUG: Making LLM call with History + Current Query...",
                            },
                        )

                print(
                    f"DEBUG: TOTAL messages being sent to OpenAI: {len(messages_sent_to_openai)}"
                )
                if messages_sent_to_openai:
                    print(f"DEBUG: First message sent: {messages_sent_to_openai[0]}")
                    if len(messages_sent_to_openai) > 1:
                        print(
                            f"DEBUG: Last message sent: {messages_sent_to_openai[-1]}"
                        )

                main_llm_span = None
                if processing_span:
                    main_llm_span = processing_span.span(
                        name="main_llm_call",
                        metadata={"status": "Making main LLM call to OpenAI"},
                    )

                start_time = time.time()
                openai_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_sent_to_openai,
                    tools=available_tools,
                    tool_choice="auto",
                )

                # Calculate latency
                latency = time.time() - start_time

                response_message = openai_response.choices[0].message

                if main_llm_span:
                    main_llm_span.end(
                        output=response_message.content[:200]
                        + (
                            "..."
                            if response_message.content
                            and len(response_message.content) > 200
                            else ""
                        ),
                        metadata={
                            "model": "gpt-4o-mini",
                            "latency": latency,
                            "has_tool_calls": response_message.tool_calls is not None,
                            "tool_calls_count": (
                                len(response_message.tool_calls)
                                if response_message.tool_calls
                                else 0
                            ),
                            "status": "Main LLM call completed successfully.",
                        },
                    )

                tool_processing_span = None
                if processing_span:
                    if response_message.tool_calls:
                        tool_processing_span = processing_span.span(
                            name="process_tool_calls",
                            metadata={"status": "Processing tool calls"},
                        )
                    else:
                        tool_processing_span = processing_span.span(
                            name="process_direct_response",
                            metadata={"status": "Processing direct text response"},
                        )

                if response_message.tool_calls:
                    print(f"DEBUG: Tool call(s) requested...")
                    raw_ai_response = await handle_tool_calls(
                        response_message,
                        user_query,
                        messages_sent_to_openai,
                        client,
                        langfuse,
                    )

                    if tool_processing_span:
                        tool_processing_span.end(
                            output=raw_ai_response[:200]
                            + (
                                "..."
                                if raw_ai_response and len(raw_ai_response) > 200
                                else ""
                            ),
                            metadata={"status": "Tool calls processed successfully."},
                        )
                else:
                    raw_ai_response = response_message.content
                    print(
                        f"DEBUG: Direct text response received snippet: {raw_ai_response[:50] if raw_ai_response else 'None'}..."
                    )

                    if tool_processing_span:
                        tool_processing_span.end(
                            output=raw_ai_response[:200]
                            + (
                                "..."
                                if raw_ai_response and len(raw_ai_response) > 200
                                else ""
                            ),
                            metadata={"status": "Direct text response processed."},
                        )

                final_processing_span = None
                if processing_span:
                    final_processing_span = processing_span.span(
                        name="final_processing",
                        metadata={"status": "Final response processing"},
                    )

                if not raw_ai_response:
                    print("DEBUG: ERROR - No final content generated.")
                    final_response_content = "I encountered an issue."
                    raw_ai_response = None

                    if final_processing_span:
                        final_processing_span.end(
                            output="I encountered an issue.",
                            metadata={
                                "error": "No content generated",
                                "status": "DEBUG: ERROR - No final content generated.",
                            },
                        )
                else:
                    final_response_content = process_text(raw_ai_response)
                    print(
                        f"DEBUG: Returning formatted response: {final_response_content}"
                    )

                    if final_processing_span:
                        final_processing_span.end(
                            output=final_response_content,
                            metadata={
                                "status": "DEBUG: Final response formatted successfully."
                            },
                        )

                if processing_span:
                    processing_span.end(
                        output=final_response_content,
                        metadata={
                            "response_type": "llm_generated",
                            "web_search_used": search_needed,
                            "tool_calls_used": response_message.tool_calls is not None,
                            "status": "Query processing completed successfully.",
                        },
                    )

    # Error Handling
    except BadRequestError as bre:
        print(f"DEBUG: ERROR - OpenAI Bad Request: {bre}")
        raw_ai_response = None

        if processing_span:
            processing_span.end(
                output=None,
                metadata={
                    "error_type": "bad_request",
                    "error": str(bre),
                    "status": f"DEBUG: ERROR - OpenAI Bad Request: {bre}",
                },
            )

        if chat_span:
            chat_span.end(
                output=None,
                metadata={
                    "error_type": "bad_request",
                    "error": str(bre),
                    "status": f"DEBUG: ERROR - OpenAI Bad Request: {bre}",
                },
            )

        raise HTTPException(
            status_code=400,
            detail=f"API Error: {bre.body.get('message', 'Bad Request')}",
        )
    except HTTPException as http_exc:
        raw_ai_response = None

        if processing_span:
            processing_span.end(
                output=None,
                metadata={
                    "error_type": "http_exception",
                    "error": str(http_exc),
                    "status": f"DEBUG: ERROR - HTTP Exception: {http_exc}",
                },
            )

        if chat_span:
            chat_span.end(
                output=None,
                metadata={
                    "error_type": "http_exception",
                    "error": str(http_exc),
                    "status": f"DEBUG: ERROR - HTTP Exception: {http_exc}",
                },
            )

        raise http_exc
    except Exception as e:
        print(f"DEBUG: ERROR - Critical error in chat endpoint: {e}")
        traceback.print_exc()
        raw_ai_response = None

        if processing_span:
            processing_span.end(
                output=None,
                metadata={
                    "error_type": "critical",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "status": f"DEBUG: ERROR - Critical error in chat endpoint: {e}",
                },
            )

        if chat_span:
            chat_span.end(
                output=None,
                metadata={
                    "error_type": "critical",
                    "error": str(e),
                    "status": f"DEBUG: ERROR - Critical error in chat endpoint: {e}",
                },
            )

        raise HTTPException(status_code=500, detail="Internal server error.")

    # --- Save History ---
    history_save_span = None
    if chat_span:
        history_save_span = chat_span.span(
            name="save_history", metadata={"status": "Saving conversation history"}
        )

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

            if history_save_span:
                history_save_span.end(
                    output=len(updated_history),
                    metadata={
                        "history_size": len(updated_history),
                        "status": f"DEBUG: History save successful for session {session_id[-6:]}.",
                    },
                )
        except Exception as e:
            print(f"DEBUG: ERROR - Redis SET failed: {e}. History not saved.")

            if history_save_span:
                history_save_span.end(
                    output=None,
                    metadata={
                        "error": str(e),
                        "status": f"DEBUG: ERROR - Redis SET failed: {e}. History not saved.",
                    },
                )
    elif history_save_span:
        history_save_span.end(
            output=None,
            metadata={
                "status": "History not saved - missing redis, session_id, or response."
            },
        )

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

    if chat_span:
        chat_span.end(
            output=final_response_content,
            metadata={
                "session_id": session_id,
                "is_new_session": is_new_session,
                "response_length": (
                    len(final_response_content) if final_response_content else 0
                ),
                "status": "Chat request processed successfully.",
            },
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
