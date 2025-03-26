# Refactor with classify call
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from openai import (
    OpenAI,
    BadRequestError,
)  # Import BadRequestError for specific handling
from dotenv import load_dotenv
import os
import json
from duckduckgo_search import DDGS
from datetime import datetime  # Added for dynamic date
import re


# --- Configuration & Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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


# Initialize FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:5173",  # Local dev
    "https://localhost",  # Replace with your WordPress domain
    "https://nextdawnai.cloud",
    "https://nextaisolutions.cloud",
    "*",  # Temporary wildcard; refine for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.nextaisolutions\.cloud$",  # Regex for subdomains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "X-Request-ID"],
    max_age=600,  # Cache preflight requests for 10 minutes
)


# --- Helper Functions --- (Keep as they are)
def is_related_to_stocks_crypto(query: str) -> bool:
    keywords = [
        "stock",
        "crypto",
        "trading",
        "shares",
        "bitcoin",
        "ethereum",
        "market",
        "price",
        "investment",
        "buy",
        "sell",
        "ticker",
        "portfolio",
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)


def get_stock_price(ticker: str) -> float | str:
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
        if price:
            print(f"Fetched price for {ticker}: ${price}")
            return round(float(price), 2)
        else:
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
                print(f"Fetched closing price for {ticker}: ${price}")
                return round(float(price), 2)
            else:
                print(f"Could not find price for {ticker}.")
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
                [f"{res.get('title', '')}: {res.get('body', '')}" for res in results]
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


# --- OpenAI Tool Definitions --- (Keep as they are - correct format for chat.completions)
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


async def needs_web_search(user_query: str) -> bool:
    """
    Uses `client.chat.completions.create` to classify if the user query requires a web search.
    """
    if not client:
        print("OpenAI client not available for web search classification.")
        return False

    print(f"Classifying query for web search need: '{user_query}'")
    try:
        # Use the messages format for chat.completions
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
            model="gpt-4o-mini",
            messages=classification_messages,
            max_tokens=5,  # max_tokens is valid here
            temperature=0.0,
            # No tools needed for this simple classification
        )

        # Parse the response content
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip().lower()
            print(f"Web search classification result: '{result_text}'")
            if "true" in result_text:
                return True
            elif "false" in result_text:
                return False
            else:
                print("Warning: Classification unclear. Defaulting to False.")
                return False
        else:
            print(
                "Warning: Could not parse classification response. Defaulting to False."
            )
            print(f"Raw classification response: {response}")
            return False

    except Exception as e:
        # Catch specific errors if needed, e.g., BadRequestError
        print(f"Error during web search classification LLM call: {e}")
        return False


# Use the handle_tool_calls version compatible with chat.completions
async def handle_tool_calls(
    response_message,  # Pass the message object containing tool_calls
    user_query: str,
    initial_messages: list,  # Pass the message history that led to the tool call
) -> str:
    """Handles tool calls made by the LLM via chat.completions and gets a final response."""
    if not client:
        return "Error: OpenAI client not available to handle tool calls."

    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content or "Error: No tool calls found and no content."

    print(f"Handling {len(tool_calls)} tool call(s)...")
    messages_history = initial_messages + [
        response_message
    ]  # Add assistant's turn with tool call requests

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

        # Append the tool result message to the history
        messages_history.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": result_content,
            }
        )

    # --- Make the follow-up call with tool results ---
    print("Making follow-up LLM call with tool results...")
    try:
        follow_up_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_history,  # Pass the full history including tool results
            # No tools needed here unless you expect chained tool calls
        )

        final_content = follow_up_response.choices[0].message.content
        print(f"Follow-up Response Content: {final_content}")
        return final_content or "Error: No content in follow-up response."

    except Exception as e:
        print(f"Error during follow-up LLM call: {e}")
        # Fallback: Try to return combined raw tool results if API fails
        raw_results_text = "\n".join(
            [msg["content"] for msg in messages_history if msg["role"] == "tool"]
        )
        return f"Could not get final summary due to API error. Tool results:\n{raw_results_text}"


@app.post("/api/chat")
async def chat(query: QueryRequest):
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")

    user_query = query.message
    print(f"\n--- New Request ---")
    print(f"Received query: {user_query}")

    if not is_related_to_stocks_crypto(user_query):
        print("Query not related to stocks/crypto, returning restricted response")
        return {
            "response": "I can only answer questions about stocks, cryptocurrency, or trading."
        }

    final_response_content = ""
    try:
        search_needed = await needs_web_search(user_query)

        system_prompt = "You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers in USD. If the user asks for a recommendation, give a direct 'You should...' statement. Use provided tools when necessary. Ensure all prices are presented in USD."

        response = None
        messages_for_api = []  # Keep track of messages sent to API

        if search_needed:
            print("Web search determined to be necessary.")
            current_date = datetime.now().strftime("%B %d, %Y")
            search_query = f"{user_query} price in USD on {current_date}"
            search_result_text = duckduckgo_search(search_query)
            print(f"Web search result snippet: {search_result_text[:200]}...")

            if (
                not validate_usd_result(search_result_text)
                and "No relevant information found" not in search_result_text
            ):
                print("Initial search might not be USD, refining...")
                refined_search_query = (
                    f"{user_query} price in US dollars on {current_date}"
                )
                search_result_text = duckduckgo_search(refined_search_query)
                print(f"Refined search snippet: {search_result_text[:200]}...")

            # Prepare messages for summarizing search results
            summarization_prompt = f"""Use the following web search results to answer the user's query: "{user_query}". Summarize the relevant information concisely, ensure the final answer is in USD, and respond directly to the user's question.

Web Search Results:
---
{search_result_text}
---

Your concise answer based *only* on the provided search results and user query:"""

            messages_for_api = [
                {"role": "system", "content": system_prompt},
                # Provide the user query and search results together in the user message for context
                {"role": "user", "content": summarization_prompt},
            ]

            print("Making LLM call to summarize web search results...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_api,
                tools=available_tools,  # Allow tools even during summarization
                tool_choice="auto",
            )

        else:
            print("Web search not needed. Proceeding with direct LLM call.")
            messages_for_api = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]

            print("Making direct LLM call...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_api,
                tools=available_tools,
                tool_choice="auto",
            )

        # --- Process the response ---
        response_message = response.choices[0].message

        if response_message.tool_calls:
            print(f"Tool call(s) requested. Handling...")
            final_response_content = await handle_tool_calls(
                response_message, user_query, messages_for_api  # Pass initial messages
            )
        else:
            # No tool calls, use the response content directly
            final_response_content = response_message.content
            print(f"Direct text response extracted: {final_response_content}")

        # 3. Process and Format Final Response
        if not final_response_content:
            print("Error: No final content generated.")
            final_response_content = (
                "I encountered an issue processing your request. Please try again."
            )

        formatted_text = process_text(final_response_content)
        print(f"Returning formatted response: {formatted_text}")
        final_response_content = formatted_text

    except BadRequestError as bre:  # Catch specific OpenAI errors
        print(f"OpenAI API Bad Request Error: {bre}")
        raise HTTPException(
            status_code=400,
            detail=f"API Error: {bre.body.get('message', 'Bad Request')}",
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An critical error occurred in the chat endpoint: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )

    return {"response": final_response_content}


@app.get("/api/health")
async def health_check():
    return {"status": "OK_V2"}

    # Working version before refactor
    # from fastapi import FastAPI
    # from fastapi.middleware.cors import CORSMiddleware
    # from pydantic import BaseModel
    # import yfinance as yf
    # from openai import OpenAI
    # from dotenv import load_dotenv
    # import os
    # import json
    # from langchain_community.tools import DuckDuckGoSearchRun
    # from datetime import datetime  # Added for dynamic date

    # # llm tracing with phoenix
    # # # import phoenix as px
    # # from phoenix.otel import register
    # # from openinference.instrumentation.openai import OpenAIInstrumentor

    # # PHOENIX_COLLECTOR_ENDPOINT = "http://localhost:6006"

    # # tracer_provider = register(
    # #     project_name="naor_test", endpoint="http://localhost:6006/v1/traces"
    # # )
    # # OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    # # Load environment variables
    # load_dotenv()
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # if not OPENAI_API_KEY:
    #     raise ValueError("OPENAI_API_KEY not found in environment variables!")

    # # Initialize OpenAI client
    # client = OpenAI(api_key=OPENAI_API_KEY)

    # # Initialize DuckDuckGo search
    # search = DuckDuckGoSearchRun()

    # # Initialize FastAPI app
    # app = FastAPI()

    # # CORS configuration
    # origins = [
    #     "http://localhost:5173",  # Local dev
    #     "https://localhost",  # Replace with your WordPress domain
    #     "https://nextdawnai.cloud",
    #     "https://nextaisolutions.cloud",
    #     "*",  # Temporary wildcard; refine for production
    # ]

    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=origins,
    #     allow_origin_regex=r"https://.*\.nextaisolutions\.cloud$",  # Regex for subdomains
    #     allow_credentials=True,
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    #     expose_headers=["Content-Length", "X-Request-ID"],
    #     max_age=600,  # Cache preflight requests for 10 minutes
    # )

    # class QueryRequest(BaseModel):
    #     message: str

    # def get_stock_price(ticker: str) -> str:
    #     print(f"Calling get_stock_price tool for ticker: {ticker}")
    #     try:
    #         stock = yf.Ticker(ticker.upper())
    #         price = stock.history(period="1d")["Close"].iloc[-1]
    #         return str(round(price, 2))
    #     except Exception as e:
    #         return f"Error fetching price for {ticker}: {str(e)}"

    # stock_price_function = {
    #     "type": "function",
    #     "name": "get_stock_price",
    #     "description": "Get the most recent closing price of a stock by its ticker symbol using Yahoo Finance data",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "ticker": {
    #                 "type": "string",
    #                 "description": "The stock ticker symbol (e.g., AAPL for Apple)",
    #             }
    #         },
    #         "required": ["ticker"],
    #         "additionalProperties": False,
    #     },
    # }

    # web_search_function = {
    #     "type": "function",
    #     "name": "web_search",
    #     "description": "Perform a web search using DuckDuckGo to find current information or prices in USD relevant to the user's query",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "string",
    #                 "description": "The search query to look up on the web",
    #             }
    #         },
    #         "required": ["query"],
    #         "additionalProperties": False,
    #     },
    # }

    # def is_related_to_stocks_crypto(query: str) -> bool:
    #     keywords = [
    #         "stock",
    #         "stocks",
    #         "crypto",
    #         "cryptocurrency",
    #         "trade",
    #         "trading",
    #         "market",
    #         "price",
    #         "invest",
    #         "investment",
    #         "bitcoin",
    #         "ethereum",
    #         "portfolio",
    #         "bull",
    #         "bear",
    #         "exchange",
    #         "gold",
    #         "XAUUSD",
    #     ]
    #     company_keywords = [
    #         "company",
    #         "business",
    #         "corporation",
    #         "inc",
    #         "ltd",
    #         "information",
    #         "operations",
    #         "industry",
    #         "revenue",
    #         "products",
    #         "services",
    #     ]
    #     known_companies = [
    #         "tesla",
    #         "apple",
    #         "microsoft",
    #         "google",
    #         "amazon",
    #         "facebook",
    #         "nvidia",
    #         "coinbase",
    #         "binance",
    #         "netflix",
    #         "ford",
    #         "gm",
    #         "boeing",
    #         "hp",
    #     ]

    #     query_lower = query.lower()
    #     if any(keyword in query_lower for keyword in keywords):
    #         return True
    #     if any(keyword in query_lower for keyword in company_keywords):
    #         words = query.split()
    #         potential_companies = [
    #             word for word in words if word[0].isupper() and len(word) > 2
    #         ]
    #         for company in potential_companies:
    #             try:
    #                 ticker = yf.Ticker(company.upper())
    #                 if ticker.info and "symbol" in ticker.info:
    #                     return True
    #             except Exception:
    #                 pass
    #     if any(company in query_lower for company in known_companies):
    #         return True
    #     return False

    # def requires_current_data(query: str) -> bool:
    #     current_keywords = ["latest", "current", "today", "now", "real-time", "price"]
    #     query_lower = query.lower()
    #     return any(keyword in query_lower for keyword in current_keywords)

    # def validate_usd_result(result: str) -> bool:
    #     return "$" in result or "USD" in result.lower()

    # def process_text(text):
    #     lines = text.split("\n")
    #     processed_lines = []
    #     for line in lines:
    #         if not line.strip():
    #             processed_lines.append("<br>")
    #             continue
    #         if line.startswith("### "):
    #             line = f"<h3>{line[4:].strip()}</h3>"
    #         elif line.startswith("## "):
    #             line = f"<h2>{line[3:].strip()}</h2>"
    #         elif line.startswith("# "):
    #             line = f"<h1>{line[2:].strip()}</h1>"
    #         else:
    #             line = f"<p>{line.strip()}</p>"
    #         while "**" in line:
    #             line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
    #         while "__" in line:
    #             line = line.replace("__", "<b>", 1).replace("__", "</b>", 1)
    #         while "*" in line and line.count("*") >= 2:
    #             line = line.replace("*", "<i>", 1).replace("*", "</i>", 1)
    #         while "_" in line and line.count("_") >= 2 and "__" not in line:
    #             line = line.replace("_", "<i>", 1).replace("_", "</i>", 1)
    #         processed_lines.append(line)
    #     return "".join(processed_lines)

    # @app.post("/api/chat")
    # async def chat(query: QueryRequest):
    #     user_query = query.message
    #     print(f"Received query: {user_query}")

    #     if not is_related_to_stocks_crypto(user_query):
    #         print("Query not related to stocks/crypto, returning restricted response")
    #         return {
    #             "response": "I can only answer questions about stocks, cryptocurrency, or trading. Please ask about one of those topics!"
    #         }

    #     if requires_current_data(user_query):
    #         print("Query requires current data, triggering web search")
    #         current_date = datetime.now().strftime("%B %d, %Y")
    #         search_query = f"{user_query} in USD on {current_date}"
    #         search_result = search.invoke(search_query)
    #         print("current date is: " + str(current_date))
    #         print(f"Web search result: {search_result}")

    #         if not validate_usd_result(search_result):
    #             print("Search result not in USD, refining search")
    #             search_result = search.invoke(
    #                 f"{user_query} price in US dollars on {current_date}"
    #             )
    #             print(f"Refined web search result: {search_result}")

    #         # Convert INR to USD if necessary
    #         if "Rs." in search_result or "₹" in search_result:
    #             try:
    #                 inr_price = float(
    #                     search_result.split("Rs.")[1].split()[0].replace(",", "")
    #                 )
    #                 grams = 10  # Assuming 10 grams from context
    #                 usd_price = inr_price / 83  # Approx exchange rate
    #                 usd_per_ounce = usd_price / 0.3215  # Convert to per ounce
    #                 search_result += (
    #                     f"\nConverted to USD: approximately ${usd_per_ounce:.2f} per ounce"
    #                 )
    #                 print(f"Converted INR to USD: ${usd_per_ounce:.2f} per ounce")
    #             except Exception as e:
    #                 print(f"Conversion error: {e}")
    #                 search_result += "\nNote: INR price detected, but conversion failed. Using USD data where available."

    #         initial_input = f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers. If the user asks for a recommendation, give a direct 'You should...' statement. The user asked: '{user_query}'. Use the following web search result (in USD) to inform your response:\n\n{search_result}\n\nNow, respond to the user with a concise, explicit answer in USD only, ending with: 'It’s always good to get advice from our professionals here at NRDX.com.'"
    #     else:
    #         print("Query does not require current data, proceeding with standard logic")
    #         initial_input = f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers. If the user asks for a recommendation, give a direct 'You should...' statement. Use the get_stock_price function when asked for a stock price. Use the web_search function when the query requires external information, ensuring all prices are in USD. Provide a clear comparison when asked about multiple stocks.\n\nUser: {user_query}"

    #     response = client.responses.create(
    #         model="gpt-4o-mini",
    #         input=initial_input,
    #         tools=[stock_price_function, web_search_function],
    #         stream=False,
    #     )
    #     print(f"First Response: {str(response.output)}")
    #     print("-" * 60)

    #     if not response.output or len(response.output) == 0:
    #         print("No response received from API")
    #         return {"response": "No response received from the API"}

    #     tool_calls = [
    #         output
    #         for output in response.output
    #         if hasattr(output, "type") and output.type == "function_call"
    #     ]

    #     if tool_calls:
    #         tool_results = []
    #         for tool_call in tool_calls:
    #             if tool_call.name == "get_stock_price":
    #                 args = json.loads(tool_call.arguments)
    #                 ticker = args["ticker"]
    #                 price = get_stock_price(ticker)
    #                 tool_results.append(f"The latest price for {ticker} is ${price}")
    #             elif tool_call.name == "web_search":
    #                 args = json.loads(tool_call.arguments)
    #                 search_query = f"{args['query']} in USD"
    #                 search_result = search.invoke(search_query)
    #                 print(
    #                     f"Web search tool called with query: {search_query}, result: {search_result}"
    #                 )
    #                 if not validate_usd_result(search_result):
    #                     search_result = search.invoke(
    #                         f"{args['query']} price in US dollars"
    #                     )
    #                     print(f"Refined web search result: {search_result}")
    #                 tool_results.append(
    #                     f"Web search result for '{args['query']}': {search_result}"
    #                 )

    #         tool_results_text = "\n".join(tool_results)
    #         follow_up_response = client.responses.create(
    #             model="gpt-4o-mini",
    #             input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers. If the user asks for a recommendation, give a direct 'You should...' statement. The user asked: '{user_query}'. Using the tool results below, provide a concise and explicit text response in USD only, ending with: 'It’s always good to get advice from our professionals here at NRDX.com.'\n\nTool results:\n{tool_results_text}\n\nNow, respond to the user.",
    #             tools=[stock_price_function, web_search_function],
    #             stream=False,
    #         )
    #         print(f"Follow-up Response: {str(follow_up_response)}")
    #         print("-" * 60)

    #         if (
    #             follow_up_response.output
    #             and len(follow_up_response.output) > 0
    #             and hasattr(follow_up_response.output[0], "content")
    #             and len(follow_up_response.output[0].content) > 0
    #         ):
    #             raw_text = follow_up_response.output[0].content[0].text
    #             formatted_text = process_text(raw_text)
    #             print(f"Returning formatted response: {formatted_text}")
    #             return {"response": formatted_text}
    #         else:
    #             formatted_text = process_text(tool_results_text)
    #             print(f"Fallback: Returning raw tool results: {formatted_text}")
    #             return {"response": formatted_text}

    #     elif (
    #         hasattr(response.output[0], "content")
    #         and len(response.output[0].content) > 0
    #         and hasattr(response.output[0].content[0], "text")
    #     ):
    #         raw_text = response.output[0].content[0].text
    #         formatted_text = process_text(raw_text)
    #         print(f"Returning direct response: {formatted_text}")
    #         return {"response": formatted_text}
    #     else:
    #         print("Error processing response")
    #         return {"response": "Error: Unable to process the response"}

    # @app.get("/api/health")
    # async def health_check():
    # return {"status": "OK"}
