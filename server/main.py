# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import yfinance as yf
# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# import json

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found in environment variables!")

# # Initialize OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Initialize FastAPI app
# app = FastAPI()

# # CORS configuration
# origins = [
#     "http://localhost:5173",  # Local dev
#     "https://localhost",  # Replace with your WordPress domain
#     "*",  # Temporary wildcard; refine for production
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class QueryRequest(BaseModel):
#     message: str


# def get_stock_price(ticker: str) -> str:
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


# def is_related_to_stocks_crypto(query: str) -> bool:
#     # Keywords related to finance
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
#     # Company/business-related keywords
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
#     # Well-known companies with stocks
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

#     # Check direct financial keywords
#     if any(keyword in query_lower for keyword in keywords):
#         return True

#     # Check company-related keywords
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

#     # Check well-known companies
#     if any(company in query_lower for company in known_companies):
#         return True

#     return False


# # Function to process text with Markdown-like formatting
# def process_text(text):
#     # Split text into lines for processing
#     lines = text.split("\n")
#     processed_lines = []

#     for line in lines:
#         if not line.strip():
#             # Preserve empty lines as breaks
#             processed_lines.append("<br>")
#             continue

#             # Handle headings
#         if line.startswith("### "):
#             line = f"<h3>{line[4:].strip()}</h3>"
#         elif line.startswith("## "):
#             line = f"<h2>{line[3:].strip()}</h2>"
#         elif line.startswith("# "):
#             line = f"<h1>{line[2:].strip()}</h1>"
#         else:
#             # Wrap non-heading lines in a paragraph tag for consistency
#             line = f"<p>{line.strip()}</p>"

#             # Process inline Markdown within the line
#             # Bold (**text** or __text__)
#         while "**" in line:
#             line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
#         while "__" in line:
#             line = line.replace("__", "<b>", 1).replace("__", "</b>", 1)

#             # Italics (*text* or _text_)
#         while "*" in line and line.count("*") >= 2:
#             line = line.replace("*", "<i>", 1).replace("*", "</i>", 1)
#         while (
#             "_" in line and line.count("_") >= 2 and "__" not in line
#         ):  # Avoid conflict with bold
#             line = line.replace("_", "<i>", 1).replace("_", "</i>", 1)

#         processed_lines.append(line)

#         # Join lines with breaks where needed
#     return "".join(processed_lines)


# @app.post("/chat")
# async def chat(query: QueryRequest):
#     user_query = query.message
#     print(f"Received query: {user_query}")

#     # Check if query is related to finance
#     if not is_related_to_stocks_crypto(user_query):
#         return {
#             "response": "I can only answer questions about stocks, cryptocurrency, or trading. Please ask about one of those topics!"
#         }

#     # Initial call to OpenAI
#     response = client.responses.create(
#         model="gpt-4o-mini",
#         input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. Use the get_stock_price function when asked for a stock price. Provide a clear comparison when asked about multiple stocks.\n\nUser: {user_query}",
#         tools=[stock_price_function],
#         stream=False,
#     )

#     print("First Response: " + str(response.output))
#     print("-" * 60)

#     # Check if response contains output
#     if not response.output or len(response.output) == 0:
#         return {"response": "No response received from the API"}

#     tool_calls = [
#         output
#         for output in response.output
#         if hasattr(output, "type") and output.type == "function_call"
#     ]

#     if tool_calls:
#         # Process all stock prices
#         stock_prices = {}
#         for tool_call in tool_calls:
#             if tool_call.name == "get_stock_price":
#                 args = json.loads(tool_call.arguments)
#                 ticker = args["ticker"]
#                 price = get_stock_price(ticker)
#                 stock_prices[ticker] = price

#         # Prepare tool results for follow-up
#         tool_results = "\n".join(
#             [
#                 f"The latest price for {ticker} is ${price}"
#                 for ticker, price in stock_prices.items()
#             ]
#         )

#         # Follow-up call with explicit instruction to summarize results
#         follow_up_response = client.responses.create(
#             model="gpt-4o-mini",
#             input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. The user asked: '{user_query}'. Using the tool results below, provide a concise text response summarizing the information. Do not invoke additional tool calls unless explicitly requested.\n\nTool results:\n{tool_results}\n\nNow, respond to the user with the stock prices in a clear, natural language format.",
#             tools=[
#                 stock_price_function
#             ],  # Still provide tools in case they're needed later
#             stream=False,
#         )
#         print("Follow-up Response: " + str(follow_up_response))
#         print("-" * 60)
#         print(
#             f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading.\n\nUser: {user_query}\n\nTool results:\n{tool_results}"
#         )
#         print("-" * 60)

#         # Handle follow-up response
#         if (
#             follow_up_response.output
#             and len(follow_up_response.output) > 0
#             and hasattr(follow_up_response.output[0], "content")
#             and len(follow_up_response.output[0].content) > 0
#         ):
#             raw_text = follow_up_response.output[0].content[0].text
#             formatted_text = process_text(raw_text)
#             return {"response": formatted_text}
#         else:
#             # Fallback: If no text response, return the tool results directly
#             formatted_text = process_text(tool_results)
#             return {"response": formatted_text}

#     # Check if it's a direct response (ResponseOutputMessage)
#     elif (
#         hasattr(response.output[0], "content")
#         and len(response.output[0].content) > 0
#         and hasattr(response.output[0].content[0], "text")
#     ):
#         raw_text = response.output[0].content[0].text
#         formatted_text = process_text(raw_text)
#         return {"response": formatted_text}
#     else:
#         return {"response": "Error: Unable to process the response"}


# @app.get("/health")
# async def health_check():
#     return {"status": "OK"}


# # ver 2, with duckduck go search tool
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime  # Added for dynamic date


# llm tracing with phoenix
# # import phoenix as px
# from phoenix.otel import register
# from openinference.instrumentation.openai import OpenAIInstrumentor


# PHOENIX_COLLECTOR_ENDPOINT = "http://localhost:6006"

# tracer_provider = register(
#     project_name="naor_test", endpoint="http://localhost:6006/v1/traces"
# )
# OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize DuckDuckGo search
search = DuckDuckGoSearchRun()

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:5173",  # Local dev
    "https://localhost",  # Replace with your WordPress domain
    "https://nextdawnai.cloud",
    "*",  # Temporary wildcard; refine for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    message: str


def get_stock_price(ticker: str) -> str:
    print(f"Calling get_stock_price tool for ticker: {ticker}")
    try:
        stock = yf.Ticker(ticker.upper())
        price = stock.history(period="1d")["Close"].iloc[-1]
        return str(round(price, 2))
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"


stock_price_function = {
    "type": "function",
    "name": "get_stock_price",
    "description": "Get the most recent closing price of a stock by its ticker symbol using Yahoo Finance data",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., AAPL for Apple)",
            }
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
}

web_search_function = {
    "type": "function",
    "name": "web_search",
    "description": "Perform a web search using DuckDuckGo to find current information or prices in USD relevant to the user's query",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def is_related_to_stocks_crypto(query: str) -> bool:
    keywords = [
        "stock",
        "stocks",
        "crypto",
        "cryptocurrency",
        "trade",
        "trading",
        "market",
        "price",
        "invest",
        "investment",
        "bitcoin",
        "ethereum",
        "portfolio",
        "bull",
        "bear",
        "exchange",
        "gold",
        "XAUUSD",
    ]
    company_keywords = [
        "company",
        "business",
        "corporation",
        "inc",
        "ltd",
        "information",
        "operations",
        "industry",
        "revenue",
        "products",
        "services",
    ]
    known_companies = [
        "tesla",
        "apple",
        "microsoft",
        "google",
        "amazon",
        "facebook",
        "nvidia",
        "coinbase",
        "binance",
        "netflix",
        "ford",
        "gm",
        "boeing",
        "hp",
    ]

    query_lower = query.lower()
    if any(keyword in query_lower for keyword in keywords):
        return True
    if any(keyword in query_lower for keyword in company_keywords):
        words = query.split()
        potential_companies = [
            word for word in words if word[0].isupper() and len(word) > 2
        ]
        for company in potential_companies:
            try:
                ticker = yf.Ticker(company.upper())
                if ticker.info and "symbol" in ticker.info:
                    return True
            except Exception:
                pass
    if any(company in query_lower for company in known_companies):
        return True
    return False


def requires_current_data(query: str) -> bool:
    current_keywords = ["latest", "current", "today", "now", "real-time", "price"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in current_keywords)


def validate_usd_result(result: str) -> bool:
    return "$" in result or "USD" in result.lower()


def process_text(text):
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        if not line.strip():
            processed_lines.append("<br>")
            continue
        if line.startswith("### "):
            line = f"<h3>{line[4:].strip()}</h3>"
        elif line.startswith("## "):
            line = f"<h2>{line[3:].strip()}</h2>"
        elif line.startswith("# "):
            line = f"<h1>{line[2:].strip()}</h1>"
        else:
            line = f"<p>{line.strip()}</p>"
        while "**" in line:
            line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
        while "__" in line:
            line = line.replace("__", "<b>", 1).replace("__", "</b>", 1)
        while "*" in line and line.count("*") >= 2:
            line = line.replace("*", "<i>", 1).replace("*", "</i>", 1)
        while "_" in line and line.count("_") >= 2 and "__" not in line:
            line = line.replace("_", "<i>", 1).replace("_", "</i>", 1)
        processed_lines.append(line)
    return "".join(processed_lines)


@app.post("/api/chat")
async def chat(query: QueryRequest):
    user_query = query.message
    print(f"Received query: {user_query}")

    if not is_related_to_stocks_crypto(user_query):
        print("Query not related to stocks/crypto, returning restricted response")
        return {
            "response": "I can only answer questions about stocks, cryptocurrency, or trading. Please ask about one of those topics!"
        }

    if requires_current_data(user_query):
        print("Query requires current data, triggering web search")
        current_date = datetime.now().strftime("%B %d, %Y")
        search_query = f"{user_query} in USD on {current_date}"
        search_result = search.invoke(search_query)
        print("current date is: " + str(current_date))
        print(f"Web search result: {search_result}")

        if not validate_usd_result(search_result):
            print("Search result not in USD, refining search")
            search_result = search.invoke(
                f"{user_query} price in US dollars on {current_date}"
            )
            print(f"Refined web search result: {search_result}")

        # Convert INR to USD if necessary
        if "Rs." in search_result or "₹" in search_result:
            try:
                inr_price = float(
                    search_result.split("Rs.")[1].split()[0].replace(",", "")
                )
                grams = 10  # Assuming 10 grams from context
                usd_price = inr_price / 83  # Approx exchange rate
                usd_per_ounce = usd_price / 0.3215  # Convert to per ounce
                search_result += (
                    f"\nConverted to USD: approximately ${usd_per_ounce:.2f} per ounce"
                )
                print(f"Converted INR to USD: ${usd_per_ounce:.2f} per ounce")
            except Exception as e:
                print(f"Conversion error: {e}")
                search_result += "\nNote: INR price detected, but conversion failed. Using USD data where available."

        initial_input = f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers. If the user asks for a recommendation, give a direct 'You should...' statement. The user asked: '{user_query}'. Use the following web search result (in USD) to inform your response:\n\n{search_result}\n\nNow, respond to the user with a concise, explicit answer in USD only, ending with: 'It’s always good to get advice from our professionals here at NRDX.com.'"
    else:
        print("Query does not require current data, proceeding with standard logic")
        initial_input = f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers. If the user asks for a recommendation, give a direct 'You should...' statement. Use the get_stock_price function when asked for a stock price. Use the web_search function when the query requires external information, ensuring all prices are in USD. Provide a clear comparison when asked about multiple stocks.\n\nUser: {user_query}"

    response = client.responses.create(
        model="gpt-4o-mini",
        input=initial_input,
        tools=[stock_price_function, web_search_function],
        stream=False,
    )
    print(f"First Response: {str(response.output)}")
    print("-" * 60)

    if not response.output or len(response.output) == 0:
        print("No response received from API")
        return {"response": "No response received from the API"}

    tool_calls = [
        output
        for output in response.output
        if hasattr(output, "type") and output.type == "function_call"
    ]

    if tool_calls:
        tool_results = []
        for tool_call in tool_calls:
            if tool_call.name == "get_stock_price":
                args = json.loads(tool_call.arguments)
                ticker = args["ticker"]
                price = get_stock_price(ticker)
                tool_results.append(f"The latest price for {ticker} is ${price}")
            elif tool_call.name == "web_search":
                args = json.loads(tool_call.arguments)
                search_query = f"{args['query']} in USD"
                search_result = search.invoke(search_query)
                print(
                    f"Web search tool called with query: {search_query}, result: {search_result}"
                )
                if not validate_usd_result(search_result):
                    search_result = search.invoke(
                        f"{args['query']} price in US dollars"
                    )
                    print(f"Refined web search result: {search_result}")
                tool_results.append(
                    f"Web search result for '{args['query']}': {search_result}"
                )

        tool_results_text = "\n".join(tool_results)
        follow_up_response = client.responses.create(
            model="gpt-4o-mini",
            input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. You must provide very clear and explicit answers. If the user asks for a recommendation, give a direct 'You should...' statement. The user asked: '{user_query}'. Using the tool results below, provide a concise and explicit text response in USD only, ending with: 'It’s always good to get advice from our professionals here at NRDX.com.'\n\nTool results:\n{tool_results_text}\n\nNow, respond to the user.",
            tools=[stock_price_function, web_search_function],
            stream=False,
        )
        print(f"Follow-up Response: {str(follow_up_response)}")
        print("-" * 60)

        if (
            follow_up_response.output
            and len(follow_up_response.output) > 0
            and hasattr(follow_up_response.output[0], "content")
            and len(follow_up_response.output[0].content) > 0
        ):
            raw_text = follow_up_response.output[0].content[0].text
            formatted_text = process_text(raw_text)
            print(f"Returning formatted response: {formatted_text}")
            return {"response": formatted_text}
        else:
            formatted_text = process_text(tool_results_text)
            print(f"Fallback: Returning raw tool results: {formatted_text}")
            return {"response": formatted_text}

    elif (
        hasattr(response.output[0], "content")
        and len(response.output[0].content) > 0
        and hasattr(response.output[0].content[0], "text")
    ):
        raw_text = response.output[0].content[0].text
        formatted_text = process_text(raw_text)
        print(f"Returning direct response: {formatted_text}")
        return {"response": formatted_text}
    else:
        print("Error processing response")
        return {"response": "Error: Unable to process the response"}


@app.get("/api/health")
async def health_check():
    return {"status": "OK"}


# version 2.1 refactor of web search
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import yfinance as yf
# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# import json
# from langchain_community.tools import DuckDuckGoSearchRun
# from datetime import datetime

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found in environment variables!")

# # Initialize OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Initialize DuckDuckGo search tool
# search = DuckDuckGoSearchRun()

# # Initialize FastAPI app and configure CORS
# app = FastAPI()
# origins = [
#     "http://localhost:5173",  # Local dev
#     "https://localhost",  # Replace with your WordPress domain
#     "*",  # Temporary wildcard; refine for production
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Request model
# class QueryRequest(BaseModel):
#     message: str


# # Function definitions for tools
# def get_stock_price(ticker: str) -> str:
#     """Fetch the latest closing price for a given ticker symbol."""
#     print(f"Calling get_stock_price for ticker: {ticker}")
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


# # Query analysis functions
# def is_related_to_stocks_crypto(query: str) -> bool:
#     """Check if the query is related to stocks or cryptocurrency."""
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
#     """Determine if the query requires current or real-time data."""
#     current_keywords = ["latest", "current", "today", "now", "real-time", "price"]
#     return any(keyword in query.lower() for keyword in current_keywords)


# def validate_usd_result(result: str) -> bool:
#     """Check if the result contains USD information."""
#     return "$" in result or "USD" in result.lower()


# def convert_inr_to_usd(search_result: str) -> str:
#     """
#     If the search result contains INR pricing (e.g., 'Rs.' or '₹'),
#     attempt to convert it to USD per ounce.
#     """
#     if "Rs." in search_result or "₹" in search_result:
#         try:
#             inr_str = search_result.split("Rs.")[1].split()[0].replace(",", "")
#             inr_price = float(inr_str)
#             grams = 10  # Assumed quantity from context
#             usd_price = inr_price / 83  # Approximate conversion rate
#             usd_per_ounce = usd_price / 0.3215  # Conversion to per ounce
#             conversion_info = (
#                 f"\nConverted to USD: approximately ${usd_per_ounce:.2f} per ounce"
#             )
#             print(f"Converted INR to USD: {conversion_info}")
#             return search_result + conversion_info
#         except Exception as e:
#             print(f"Conversion error: {e}")
#             return (
#                 search_result
#                 + "\nNote: INR price detected, but conversion failed. Using USD data where available."
#             )
#     return search_result


# def get_current_date() -> str:
#     """Return the current date formatted as 'Month day, Year'."""
#     return datetime.now().strftime("%B %d, %Y")


# def perform_web_search(query: str, current_date: str) -> str:
#     """
#     Invoke the DuckDuckGo search tool with a query appended with date information.
#     Refines the query if the result does not contain USD information.
#     """
#     search_query = f"{query} in USD on {current_date}"
#     result = search.invoke(search_query)
#     print(f"Initial web search result: {result}")
#     if not validate_usd_result(result):
#         refined_query = f"{query} price in US dollars on {current_date}"
#         result = search.invoke(refined_query)
#         print(f"Refined web search result: {result}")
#     return convert_inr_to_usd(result)


# def process_tool_calls(tool_calls: list, user_query: str) -> str:
#     """Process each tool call and return a combined string of tool results."""
#     tool_results = []
#     for tool_call in tool_calls:
#         if tool_call.name == "get_stock_price":
#             args = json.loads(tool_call.arguments)
#             ticker = args["ticker"]
#             price = get_stock_price(ticker)
#             tool_results.append(f"The latest price for {ticker} is ${price}")
#         elif tool_call.name == "web_search":
#             args = json.loads(tool_call.arguments)
#             query = args["query"]
#             refined_query = f"{query} in USD"
#             result = search.invoke(refined_query)
#             print(
#                 f"Web search tool called with query: {refined_query}, result: {result}"
#             )
#             if not validate_usd_result(result):
#                 result = search.invoke(f"{query} price in US dollars")
#                 print(f"Refined web search result: {result}")
#             tool_results.append(f"Web search result for '{query}': {result}")
#     return "\n".join(tool_results)


# def format_response(raw_text: str) -> str:
#     """
#     Process text by converting markdown-like syntax into HTML.
#     Handles headers and simple bold/italic markers.
#     """
#     lines = raw_text.split("\n")
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


# @app.post("/chat")
# async def chat(query: QueryRequest):
#     """Endpoint for processing chat queries."""
#     user_query = query.message
#     print(f"Received query: {user_query}")

#     # Ensure query is related to stocks/crypto
#     if not is_related_to_stocks_crypto(user_query):
#         print("Query not related to stocks/crypto. Returning restricted response.")
#         return {
#             "response": "I can only answer questions about stocks, cryptocurrency, or trading. Please ask about one of those topics!"
#         }

#     current_date = get_current_date()

#     # Build initial input based on whether current data is required
#     if requires_current_data(user_query):
#         print("Query requires current data. Using web search.")
#         search_result = perform_web_search(user_query, current_date)
#         initial_input = (
#             f"System: You are a financial assistant specializing in stocks, cryptocurrency, "
#             f"and trading. Provide clear, explicit answers and use the following web search result "
#             f"(in USD) to inform your response:\n\n{search_result}\n\n"
#             f"Now, respond to the user with a concise answer in USD only, ending with: "
#             f"'It’s always good to get advice from our professionals here at NRDX.com.'"
#         )
#     else:
#         print("Query does not require current data. Proceeding with standard logic.")
#         initial_input = (
#             f"System: You are a financial assistant specializing in stocks, cryptocurrency, "
#             f"and trading. Provide clear and explicit answers. Use the get_stock_price function when "
#             f"asked for a stock price and the web_search function for external data, ensuring all prices "
#             f"are in USD. Provide clear comparisons when discussing multiple stocks.\n\nUser: {user_query}"
#         )

#     # Initial API response call using responses.api
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

#     # Extract and process tool calls
#     tool_calls = [
#         output
#         for output in response.output
#         if hasattr(output, "type") and output.type == "function_call"
#     ]

#     if tool_calls:
#         tool_results_text = process_tool_calls(tool_calls, user_query)
#         follow_up_input = (
#             f"System: You are a financial assistant specializing in stocks, cryptocurrency, "
#             f"and trading. Provide clear, explicit answers in USD only. Use the tool results below "
#             f"to inform your response, and end with: 'It’s always good to get advice from our professionals here at NRDX.com.'\n\n"
#             f"Tool results:\n{tool_results_text}\n\nNow, respond to the user."
#         )
#         follow_up_response = client.responses.create(
#             model="gpt-4o-mini",
#             input=follow_up_input,
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
#             formatted_text = format_response(raw_text)
#             print(f"Returning formatted response: {formatted_text}")
#             return {"response": formatted_text}
#         else:
#             fallback_text = format_response(tool_results_text)
#             print(f"Fallback: Returning raw tool results: {fallback_text}")
#             return {"response": fallback_text}

#     elif (
#         hasattr(response.output[0], "content")
#         and len(response.output[0].content) > 0
#         and hasattr(response.output[0].content[0], "text")
#     ):
#         raw_text = response.output[0].content[0].text
#         formatted_text = format_response(raw_text)
#         print(f"Returning direct response: {formatted_text}")
#         return {"response": formatted_text}
#     else:
#         print("Error processing response")
#         return {"response": "Error: Unable to process the response"}


# @app.get("/health")
# async def health_check():
#     """Simple health check endpoint."""
#     return {"status": "OK_2.1"}
