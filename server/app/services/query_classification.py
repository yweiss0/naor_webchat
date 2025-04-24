from openai import OpenAI
from langfuse import Langfuse
from datetime import datetime
from typing import Optional, Any
import json


async def is_related_to_stocks_crypto(
    query: str,
    client: Optional[OpenAI],
    langfuse_client: Optional[Langfuse] = None,
    parent_span: Any = None,
) -> bool:
    """
    Determines if the query is related to stocks, cryptocurrency, or trading using OpenAI.
    Returns True if related, False otherwise.
    """
    if not client:
        print("OpenAI client not available for stock/crypto classification.")
        return False

    # Quick check for common commodities (especially gold)
    query_lower = query.lower()
    if any(
        gold_term in query_lower
        for gold_term in ["gold", "xau", "xauusd", "xau/usd", "gold price"]
    ):
        print("Query contains gold-related terms, automatically classifying as related")
        if langfuse_client and parent_span:
            span = parent_span.span(
                name="classify_stocks_crypto",
                input={"query": query},
            )
            span.end(output={"is_related": True, "reason": "gold_keyword"})
        return True

    print(f"Classifying if query is related to stocks/crypto: '{query}'")

    # Create a span for this operation if Langfuse is available
    span = None
    if langfuse_client and parent_span:
        span = parent_span.span(
            name="classify_stocks_crypto",
            input={"query": query},
        )

    try:
        classification_messages = [
            {
                "role": "system",
                "content": """Analyze the user query. Is it related to stocks, cryptocurrency, trading, investing, financial markets, or commodities (like gold and silver)?
                
                Consider semantic similarity and not just exact matches. For example:
                - Questions about technical indicators (RSI, MACD, etc.) are related
                - Questions about market analysis, charts, or trading strategies are related
                - Questions about financial instruments, brokers, or trading platforms are related
                - Questions about economic indicators that affect markets are related
                - Questions about commodity prices, especially gold and silver, are related
                
                If yes, respond with 'True'.
                If no, respond with 'False'.
                
                Answer only with 'True' or 'False'.""",
            },
            {
                "role": "user",
                "content": f'User Query: "{query}"\n\nIs this query related to stocks, cryptocurrency, commodities, or trading (True/False):',
            },
        ]

        start_time = datetime.now()
        if langfuse_client and span:
            # Use Langfuse-instrumented OpenAI client if available
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=classification_messages,
                max_tokens=5,
                temperature=0.0,
            )

            # Capture the generation with Langfuse
            span.generation(
                name="stock_crypto_classification",
                model="gpt-4o-mini",
                input=classification_messages,
                output=response.choices[0].message if response.choices else None,
                start_time=start_time,
                end_time=datetime.now(),
            )
        else:
            # Use regular OpenAI client
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=classification_messages,
                max_tokens=5,
                temperature=0.0,
            )

        # Parse the response content
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip().lower()
            print(f"Stock/crypto classification result: '{result_text}'")
            is_related = "true" in result_text

            # Update the span with the result if available
            if langfuse_client and span:
                span.end(output={"is_related": is_related, "result_text": result_text})

            return is_related
        else:
            print(
                "Warning: Could not parse classification response. Defaulting to False."
            )

            # Update the span with the failure if available
            if langfuse_client and span:
                span.end(
                    output={
                        "error": "Could not parse classification response",
                        "is_related": False,
                    }
                )

            return False

    except Exception as e:
        print(f"Error during stock/crypto classification: {e}")

        # Update the span with the error if available
        if langfuse_client and span:
            span.end(output={"error": str(e), "is_related": False}, status="error")

        return False
