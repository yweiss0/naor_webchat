from openai import OpenAI
from langfuse import Langfuse
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


async def needs_web_search(
    user_query: str,
    client: Optional[OpenAI],
    langfuse_client: Optional[Langfuse] = None,
    parent_span: Any = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Determines if the query requires web search for up-to-date information.
    Also handles follow-up questions by checking conversation history.
    """
    print(f"Classifying web search need for: '{user_query}'")
    query_lower = user_query.lower()

    # Create a span for this operation if Langfuse is available
    span = None
    if langfuse_client and parent_span:
        span = parent_span.span(
            name="needs_web_search",
            input={"query": user_query},
        )

    # Check if this is a follow-up about stock price first
    if conversation_history and len(conversation_history) > 0:
        try:
            # Check for common follow-up patterns about prices
            price_followup_indicators = [
                "price",
                "cost",
                "worth",
                "value",
                "was it",
                "was the",
                "what was",
                "how much",
                "how was",
            ]

            date_indicators = [
                "yesterday",
                "last week",
                "last month",
                "at that time",
                "that day",
                "that time",
                "on that date",
                "back then",
                "previous",
                "earlier",
                "before",
                "prior",
            ]

            # If it looks like a price follow-up with a date, likely doesn't need web search
            if any(pi in query_lower for pi in price_followup_indicators) and any(
                di in query_lower for di in date_indicators
            ):
                print(
                    "DEBUG: Query appears to be a follow-up about historical price, skipping web search."
                )
                if langfuse_client and span:
                    span.end(
                        output={
                            "needs_web_search": False,
                            "reason": "historical_price_followup",
                        }
                    )
                return False

            # For short queries that are likely follow-ups about prices
            if len(query_lower.split()) <= 10 and any(
                pi in query_lower for pi in price_followup_indicators
            ):
                # Inspect the last few messages to see if they were about a specific stock
                for i in range(
                    len(conversation_history) - 1,
                    max(0, len(conversation_history) - 6),
                    -1,
                ):
                    last_msg = conversation_history[i]
                    if last_msg.get("role") == "assistant" and any(
                        term in last_msg.get("content", "").lower()
                        for term in ["price", "stock", "ticker", "$", "usd"]
                    ):
                        print(
                            "DEBUG: Recent messages were about stock prices, this is likely a follow-up."
                        )
                        if langfuse_client and span:
                            span.end(
                                output={
                                    "needs_web_search": False,
                                    "reason": "price_followup_context",
                                }
                            )
                        return False
        except Exception as e:
            print(f"DEBUG: Error checking for follow-up about price: {e}")
            # Continue with normal classification if error

    # Check for specific keywords that should always trigger web search
    web_search_keywords = [
        "today",
        "today's",
        "today is",
        "today was",
        "today will be",
        "at the moment",
        "at the moment",
        "current status" "good day",
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
        if langfuse_client and span:
            span.end(output={"needs_web_search": True, "reason": "keyword_match"})
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
        if langfuse_client and span:
            span.end(output={"needs_web_search": False, "reason": "recall_query"})
        return False

    if not client:
        print("DEBUG: needs_web_search - OpenAI client is None, cannot classify.")
        if langfuse_client and span:
            span.end(
                output={"needs_web_search": False, "reason": "no_openai_client"},
                status="error",
            )
        return False
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
- which stocks are the best at the moment to acquire?

Then you MUST respond with 'True' as these questions require current information.

Do NOT say True if the user is asking about the conversation history or what was said before. Answer only with 'True' or 'False'.""",
            },
            {
                "role": "user",
                "content": f'User Query: "{user_query}"\n\nRequires Web Search (True/False):',
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
                name="web_search_classification",
                model="gpt-4o-mini",
                input=classification_messages,
                output=response.choices[0].message if response.choices else None,
                start_time=start_time,
                end_time=datetime.now(),
            )
        else:
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
            needs_search = "true" in result_text

            if langfuse_client and span:
                span.end(
                    output={
                        "needs_web_search": needs_search,
                        "reason": "llm_classification",
                        "result_text": result_text,
                    }
                )

            return needs_search
        else:
            print(
                "DEBUG: Could not parse classification response. Defaulting to False."
            )

            if langfuse_client and span:
                span.end(
                    output={"needs_web_search": False, "reason": "parse_error"},
                    status="error",
                )

            return False
    except Exception as e:
        print(f"DEBUG: Error during classification LLM call: {e}")

        if langfuse_client and span:
            span.end(
                output={"needs_web_search": False, "reason": "error", "error": str(e)},
                status="error",
            )

        return False
