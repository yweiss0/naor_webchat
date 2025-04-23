from openai import OpenAI
from langfuse import Langfuse
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
from app.services.market_data import get_stock_price
from app.services.web_search import duckduckgo_search

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


async def handle_tool_calls(
    response_message: Any,
    user_query: str,
    messages_history: List[Dict[str, Any]],
    client: Optional[OpenAI],
    langfuse_client: Optional[Langfuse] = None,
    parent_span: Any = None,
) -> str:
    """
    Handle OpenAI tool calls and generate a follow-up response.
    """
    if not client:
        return "Error: OpenAI client not available."
    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content or "Error: No tool calls or content."

    print(f"DEBUG: Handling {len(tool_calls)} tool call(s)...")
    messages_for_follow_up = messages_history + [response_message]

    # Create a span for this operation if Langfuse is available
    span = None
    if langfuse_client and parent_span:
        span = parent_span.span(
            name="handle_tool_calls",
            input={
                "query": user_query,
                "tool_calls": [
                    {
                        "function": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                    for tool_call in tool_calls
                ],
            },
        )

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        result_content = ""

        # Create a span for this specific tool call if Langfuse is available
        tool_span = None
        if langfuse_client and span:
            tool_span = span.span(
                name=f"tool_call_{function_name}",
                input={"function": function_name, "arguments": function_args_str},
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

                if langfuse_client and tool_span:
                    tool_span.end(
                        output={
                            "ticker": ticker,
                            "price": price,
                            "result": result_content,
                        }
                    )

            elif function_name == "web_search":
                search_query_arg = args_dict.get("query")
                search_start_time = datetime.now()
                result_content = (
                    duckduckgo_search(
                        f"{search_query_arg} in USD on {datetime.now():%B %d, %Y}"
                    )
                    if search_query_arg
                    else "Error: Search query missing."
                )

                if langfuse_client and tool_span:
                    tool_span.end(
                        output={
                            "query": search_query_arg,
                            "result_length": len(result_content),
                            "search_time_ms": (
                                datetime.now() - search_start_time
                            ).total_seconds()
                            * 1000,
                        }
                    )

            else:
                result_content = f"Error: Unknown function '{function_name}'."

                if langfuse_client and tool_span:
                    tool_span.end(
                        output={"error": f"Unknown function '{function_name}'"},
                        status="error",
                    )

            print(
                f"DEBUG: Tool '{function_name}' executed. Result snippet: {result_content[:50]}..."
            )
        except Exception as e:
            print(f"DEBUG: Error executing tool {function_name}: {e}")
            result_content = f"Error executing tool: {e}"

            if langfuse_client and tool_span:
                tool_span.end(output={"error": str(e)}, status="error")

        messages_for_follow_up.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": result_content,
            }
        )

    print("DEBUG: Making follow-up LLM call with tool results...")
    try:
        # Add a system message to emphasize the importance of current information
        follow_up_messages = [
            {
                "role": "system",
                "content": "You are a financial assistant. When providing information from web search results, always emphasize how current the information is. If the information is outdated (more than a few months old), clearly state that it may not reflect current market conditions. Always prioritize the most recent information available.",
            }
        ] + messages_for_follow_up

        start_time = datetime.now()
        if langfuse_client and span:
            # Use Langfuse-instrumented OpenAI client if available
            follow_up_response = client.chat.completions.create(
                model="gpt-4o-mini", messages=follow_up_messages
            )

            # Capture the generation with Langfuse
            span.generation(
                name="follow_up_response",
                model="gpt-4o-mini",
                input=follow_up_messages,
                output=(
                    follow_up_response.choices[0].message
                    if follow_up_response.choices
                    else None
                ),
                start_time=start_time,
                end_time=datetime.now(),
            )
        else:
            follow_up_response = client.chat.completions.create(
                model="gpt-4o-mini", messages=follow_up_messages
            )

        final_content = follow_up_response.choices[0].message.content

        if langfuse_client and span:
            span.end(output={"final_content": final_content})

        print(
            f"DEBUG: Follow-up Response snippet: {final_content[:50] if final_content else 'None'}..."
        )
        return final_content or "Error: No content in follow-up."
    except Exception as e:
        print(f"DEBUG: Error during follow-up LLM call: {e}")

        if langfuse_client and span:
            span.end(output={"error": str(e)}, status="error")

        return f"Error summarizing tool results."
