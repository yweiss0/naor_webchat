from openai import OpenAI
from langfuse import Langfuse
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
import json


async def is_stock_price_query(
    user_query: str,
    client: Optional[OpenAI],
    langfuse_client: Optional[Langfuse] = None,
    parent_span: Any = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str, bool]:
    """
    Determines if the query is specifically about current stock price and extracts the ticker.
    Returns a tuple of (is_price_query, ticker_symbol_or_empty_string, needs_market_context)

    Now handles follow-up questions by analyzing conversation history if provided.
    """
    # Quick check for gold-related terms
    query_lower = user_query.lower()
    if any(
        gold_term in query_lower
        for gold_term in [
            "gold price",
            "price of gold",
            "gold cost",
            "xauusd",
            "xau/usd",
        ]
    ):
        print("Direct gold price query detected, automatically classifying")
        return (True, "GOLD", False)

    # Check if this is a follow-up question about a previously mentioned stock
    if conversation_history:
        try:
            # Analyze if this is a follow-up and extract the ticker from previous messages
            follow_up_messages = [
                {
                    "role": "system",
                    "content": """Analyze the conversation history and the user's latest query. 
                    Determine if the latest query is a follow-up question about a stock/asset price mentioned earlier in the conversation.
                    
                    If it is a follow-up about price (e.g., "what was its price on [date]?", "how much was it yesterday?"), 
                    extract the ticker symbol from the conversation history.
                    
                    Respond in JSON format:
                    {"is_followup_price_query": true/false, "ticker": "SYMBOL", "needs_market_context": true/false}
                    
                    Set "needs_market_context" to true if the query asks about:
                    - Price changes or movements
                    - Reasons for price changes
                    - Trends or patterns
                    - Comparisons
                    """,
                },
            ]

            # Add conversation history
            for msg in conversation_history:
                follow_up_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            # Add current query
            follow_up_messages.append({"role": "user", "content": user_query})

            # Create a span for this follow-up analysis if Langfuse is available
            followup_span = None
            if langfuse_client and parent_span:
                followup_span = parent_span.span(
                    name="analyze_price_followup",
                    input={
                        "query": user_query,
                        "history_length": len(conversation_history),
                    },
                )

            # Use regular OpenAI client without tracing the generation
            followup_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=follow_up_messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # Parse the response
            if (
                followup_response.choices
                and followup_response.choices[0].message
                and followup_response.choices[0].message.content
            ):
                result_text = followup_response.choices[0].message.content.strip()
                try:
                    result_json = json.loads(result_text)
                    is_followup = result_json.get("is_followup_price_query", False)
                    ticker = result_json.get("ticker", "").strip().upper()
                    needs_market_context = result_json.get(
                        "needs_market_context", False
                    )

                    if is_followup and ticker:
                        print(f"Detected follow-up price query for ticker: {ticker}")

                        if followup_span:
                            followup_span.end(
                                output={
                                    "is_followup_price_query": is_followup,
                                    "ticker": ticker,
                                    "needs_market_context": needs_market_context,
                                }
                            )

                        return (True, ticker, needs_market_context)

                    if followup_span:
                        followup_span.end(
                            output={
                                "is_followup_price_query": is_followup,
                                "ticker": ticker,
                                "needs_market_context": needs_market_context,
                            }
                        )

                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse follow-up JSON response: {e}")
                    if followup_span:
                        followup_span.end(
                            output={"error": str(e)},
                            status="error",
                        )
        except Exception as e:
            print(f"Error during follow-up analysis: {e}")
            if "followup_span" in locals() and followup_span:
                followup_span.end(
                    output={"error": str(e)},
                    status="error",
                )

    if not client:
        print("OpenAI client not available for stock price query classification.")
        return (False, "", False)

    print(f"Classifying if query is about current stock price: '{user_query}'")

    # Create a span for this operation if Langfuse is available
    span = None
    if langfuse_client and parent_span:
        span = parent_span.span(
            name="is_stock_price_query",
            input={"query": user_query},
        )

    try:
        classification_messages = [
            {
                "role": "system",
                "content": """Analyze the user query. Is it specifically asking about the current or historical price of a stock, cryptocurrency, market index, commodity, or forex pair? 
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
                  - For gold specifically, use "GOLD" if user mentions gold, XAU/USD, or XAUUSD
                  - For silver specifically, use "SILVER" if user mentions silver, XAG/USD, or XAGUSD
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

        # Use regular OpenAI client without tracing the generation
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=classification_messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

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

                if langfuse_client and span:
                    span.end(
                        output={
                            "is_price_query": is_price_query,
                            "ticker": ticker,
                            "needs_market_context": needs_market_context,
                        }
                    )

                return (is_price_query, ticker, needs_market_context)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON response: {result_text}")

                if langfuse_client and span:
                    span.end(
                        output={
                            "error": f"Could not parse JSON response: {result_text}"
                        },
                        status="error",
                    )

                return (False, "", False)
        else:
            print(
                "Warning: Could not parse classification response. Defaulting to False."
            )

            if langfuse_client and span:
                span.end(
                    output={"error": "Could not parse classification response"},
                    status="error",
                )

            return (False, "", False)

    except Exception as e:
        print(f"Error during stock price query classification: {e}")

        if langfuse_client and span:
            span.end(output={"error": str(e)}, status="error")

        return (False, "", False)
