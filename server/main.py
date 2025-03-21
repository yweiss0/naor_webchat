from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:5173",  # Local dev
    "https://localhost",  # Replace with your WordPress domain
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


def is_related_to_stocks_crypto(query: str) -> bool:
    # Keywords related to finance
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
    # Company/business-related keywords
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
    # Well-known companies with stocks
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

    # Check direct financial keywords
    if any(keyword in query_lower for keyword in keywords):
        return True

    # Check company-related keywords
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

    # Check well-known companies
    if any(company in query_lower for company in known_companies):
        return True

    return False


# Function to process text with Markdown-like formatting
def process_text(text):
    # Split text into lines for processing
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        if not line.strip():
            # Preserve empty lines as breaks
            processed_lines.append("<br>")
            continue

            # Handle headings
        if line.startswith("### "):
            line = f"<h3>{line[4:].strip()}</h3>"
        elif line.startswith("## "):
            line = f"<h2>{line[3:].strip()}</h2>"
        elif line.startswith("# "):
            line = f"<h1>{line[2:].strip()}</h1>"
        else:
            # Wrap non-heading lines in a paragraph tag for consistency
            line = f"<p>{line.strip()}</p>"

            # Process inline Markdown within the line
            # Bold (**text** or __text__)
        while "**" in line:
            line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
        while "__" in line:
            line = line.replace("__", "<b>", 1).replace("__", "</b>", 1)

            # Italics (*text* or _text_)
        while "*" in line and line.count("*") >= 2:
            line = line.replace("*", "<i>", 1).replace("*", "</i>", 1)
        while (
            "_" in line and line.count("_") >= 2 and "__" not in line
        ):  # Avoid conflict with bold
            line = line.replace("_", "<i>", 1).replace("_", "</i>", 1)

        processed_lines.append(line)

        # Join lines with breaks where needed
    return "".join(processed_lines)


@app.post("/chat")
async def chat(query: QueryRequest):
    user_query = query.message
    print(f"Received query: {user_query}")

    # Check if query is related to finance
    if not is_related_to_stocks_crypto(user_query):
        return {
            "response": "I can only answer questions about stocks, cryptocurrency, or trading. Please ask about one of those topics!"
        }

    # Initial call to OpenAI
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. Use the get_stock_price function when asked for a stock price. Provide a clear comparison when asked about multiple stocks.\n\nUser: {user_query}",
        tools=[stock_price_function],
        stream=False,
    )

    print("First Response: " + str(response.output))
    print("-" * 60)

    # Check if response contains output
    if not response.output or len(response.output) == 0:
        return {"response": "No response received from the API"}

    tool_calls = [
        output
        for output in response.output
        if hasattr(output, "type") and output.type == "function_call"
    ]

    if tool_calls:
        # Process all stock prices
        stock_prices = {}
        for tool_call in tool_calls:
            if tool_call.name == "get_stock_price":
                args = json.loads(tool_call.arguments)
                ticker = args["ticker"]
                price = get_stock_price(ticker)
                stock_prices[ticker] = price

        # Prepare tool results for follow-up
        tool_results = "\n".join(
            [
                f"The latest price for {ticker} is ${price}"
                for ticker, price in stock_prices.items()
            ]
        )

        # Follow-up call with explicit instruction to summarize results
        follow_up_response = client.responses.create(
            model="gpt-4o-mini",
            input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. The user asked: '{user_query}'. Using the tool results below, provide a concise text response summarizing the information. Do not invoke additional tool calls unless explicitly requested.\n\nTool results:\n{tool_results}\n\nNow, respond to the user with the stock prices in a clear, natural language format.",
            tools=[
                stock_price_function
            ],  # Still provide tools in case they're needed later
            stream=False,
        )
        print("Follow-up Response: " + str(follow_up_response))
        print("-" * 60)
        print(
            f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading.\n\nUser: {user_query}\n\nTool results:\n{tool_results}"
        )
        print("-" * 60)

        # Handle follow-up response
        if (
            follow_up_response.output
            and len(follow_up_response.output) > 0
            and hasattr(follow_up_response.output[0], "content")
            and len(follow_up_response.output[0].content) > 0
        ):
            raw_text = follow_up_response.output[0].content[0].text
            formatted_text = process_text(raw_text)
            return {"response": formatted_text}
        else:
            # Fallback: If no text response, return the tool results directly
            formatted_text = process_text(tool_results)
            return {"response": formatted_text}

    # Check if it's a direct response (ResponseOutputMessage)
    elif (
        hasattr(response.output[0], "content")
        and len(response.output[0].content) > 0
        and hasattr(response.output[0].content[0], "text")
    ):
        raw_text = response.output[0].content[0].text
        formatted_text = process_text(raw_text)
        return {"response": formatted_text}
    else:
        return {"response": "Error: Unable to process the response"}


@app.get("/health")
async def health_check():
    return {"status": "OK"}
