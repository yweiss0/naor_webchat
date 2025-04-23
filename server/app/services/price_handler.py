from openai import OpenAI
from langfuse import Langfuse
from datetime import datetime
from typing import Optional, Any, Dict
from app.utils.text_processing import extract_date_from_query, normalize_ticker
from app.services.market_data import get_stock_price


async def handle_stock_price_query(
    ticker: str,
    user_query: str,
    client: Optional[OpenAI],
    needs_market_context: bool = False,
    langfuse_client: Optional[Langfuse] = None,
    parent_span: Any = None,
) -> str:
    """Handles direct stock price queries using Yahoo Finance."""
    # Normalize the ticker symbol first (especially for gold)
    normalized_ticker = normalize_ticker(ticker, user_query)
    if normalized_ticker != ticker:
        print(f"Normalized ticker from {ticker} to {normalized_ticker}")
        ticker = normalized_ticker

    print(
        f"Handling direct stock price query for ticker: {ticker}, needs_market_context: {needs_market_context}"
    )

    # Create a span for this operation if Langfuse is available
    span = None
    if langfuse_client and parent_span:
        span = parent_span.span(
            name="handle_stock_price_query",
            input={
                "ticker": ticker,
                "original_ticker": (
                    ticker if normalized_ticker == ticker else normalized_ticker
                ),
                "query": user_query,
                "needs_market_context": needs_market_context,
            },
        )

    # Extract date if present in the query
    query_date = extract_date_from_query(user_query)
    if query_date:
        print(f"Extracted date from query: {query_date}")
        if span:
            span.event(name="date_extraction", output={"extracted_date": query_date})

    # Get price and historical data
    price, historical_data = get_stock_price(
        ticker, date=query_date, include_history=needs_market_context
    )

    if span:
        span.event(
            name="stock_price_data",
            output={
                "price": price,
                "has_historical_data": historical_data is not None,
                "date_queried": query_date if query_date else "current",
            },
        )

    # Determine the type of financial instrument
    instrument_type = "stock"
    if ticker.startswith("^"):
        instrument_type = "market index"
    elif ticker.endswith("=F"):
        if ticker == "GC=F":
            instrument_type = "gold"
        elif ticker == "SI=F":
            instrument_type = "silver"
        else:
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
    date_context = f" on {query_date}" if query_date else ""

    # For gold specifically, make sure to clarify this is gold price
    display_name = (
        "Gold" if ticker == "GC=F" and "gold" in user_query.lower() else ticker
    )

    user_message_content = f"Original query: {user_query}\n\nFinancial data: The price for {display_name}{date_context} is: {price}"

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

    try:
        # Use regular OpenAI client without tracing the generation
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        final_content = response.choices[0].message.content

        if langfuse_client and span:
            span.end(output={"response": final_content})

        return final_content
    except Exception as e:
        print(f"Error formatting stock price response: {e}")

        if langfuse_client and span:
            span.end(
                output={
                    "error": str(e),
                    "fallback_response": f"The price for {display_name}{date_context} is: {price}",
                },
                status="error",
            )

        return f"The price for {display_name}{date_context} is: {price}"
