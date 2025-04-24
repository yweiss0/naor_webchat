from openai import OpenAI, BadRequestError
from langfuse import Langfuse
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from app.services.query_classification import is_related_to_stocks_crypto
from app.services.price_query_detector import is_stock_price_query
from app.services.web_search_classifier import needs_web_search
from app.services.qa_service import find_qa_match
from app.services.price_handler import handle_stock_price_query
from app.services.web_search import duckduckgo_search
from app.services.tool_handler import handle_tool_calls, available_tools
from app.utils.text_processing import process_text
from app.utils.response_processor import (
    synthesize_search_results,
    apply_response_guardrails,
    is_reason_query,
)


async def process_chat_request(
    user_query: str,
    client: OpenAI,
    redis_conn: Any,
    session_id: str,
    loaded_history: list,
    trace: Optional[Any] = None,
    langfuse_client: Optional[Langfuse] = None,
) -> Tuple[str, str, bool, bool]:
    """
    Process chat request and generate a response

    Args:
        user_query: The user's query text
        client: OpenAI client
        redis_conn: Redis connection
        session_id: Session ID string
        loaded_history: Loaded conversation history
        trace: Optional Langfuse trace
        langfuse_client: Optional Langfuse client

    Returns:
        Tuple of (final_response_content, raw_ai_response, is_new_session, early_exit)
    """
    web_search_used = False
    web_search_result = ""
    early_exit = False
    raw_ai_response = None
    final_response_content = ""
    messages_sent_to_openai = []

    # Define current_user_message_dict at the beginning to ensure it's always available
    current_user_message_dict = {"role": "user", "content": user_query}

    try:
        # First check if the query matches a question in the Q&A file
        qa_match, qa_answer = await find_qa_match(
            user_query, client, langfuse_client, trace
        )

        restricted_response = None

        if qa_match:
            print(f"DEBUG: Q&A match found, using predefined answer.")
            raw_ai_response = qa_answer
            final_response_content = process_text(raw_ai_response)
            print(f"DEBUG: Returning Q&A answer: {final_response_content}")

            if trace:
                trace.event(
                    name="qa_match_response", output={"answer": final_response_content}
                )
            # No early return; let it flow to guardrails/review
        else:
            # If no Q&A match, check if the query is related to stocks/crypto
            is_related_span = None
            if trace:
                is_related_span = trace.span(name="check_stocks_crypto_relevance")

            if not await is_related_to_stocks_crypto(
                user_query, client, langfuse_client, trace
            ):
                print("Query not related. Returning restricted response.")

                if is_related_span:
                    is_related_span.end(output={"is_related": False})

                if trace:
                    trace.update(
                        output={
                            "response": "I can only answer questions about stocks, cryptocurrency, or trading."
                        },
                        status="restricted",
                    )

                restricted_response = "I can only answer questions about stocks, cryptocurrency, or trading."
                final_response_content = restricted_response
                early_exit = True
            else:
                if is_related_span:
                    is_related_span.end(output={"is_related": True})

                # First check if this is a direct stock price query
                is_price_query, ticker, needs_market_context = (
                    await is_stock_price_query(
                        user_query,
                        client,
                        langfuse_client,
                        trace,
                        conversation_history=loaded_history if loaded_history else None,
                    )
                )

                if is_price_query and ticker:
                    print(
                        f"DEBUG: Direct stock price query detected for ticker: {ticker}"
                    )
                    raw_ai_response = await handle_stock_price_query(
                        ticker,
                        user_query,
                        client,
                        needs_market_context,
                        langfuse_client,
                        trace,
                    )
                    # If the query is asking for reasons, also do a web search and combine
                    if is_reason_query(user_query):
                        print(
                            "DEBUG: Query is a reason/explanation query, adding web search for reasons..."
                        )
                        web_search_used = True
                        if trace:
                            web_search_span = trace.span(name="web_search_for_reason")
                            web_search_result = duckduckgo_search(
                                f"{user_query} market reasons news in USD on {datetime.now():%B %d, %Y}"
                            )
                            web_search_span.end(
                                output={
                                    "query": user_query,
                                    "result_snippet": web_search_result[:200],
                                }
                            )
                        else:
                            web_search_result = duckduckgo_search(
                                f"{user_query} market reasons news in USD on {datetime.now():%B %d, %Y}"
                            )
                        raw_ai_response = f"{raw_ai_response}\n\n---\nAdditional recent reasons from the web:\n{web_search_result}"
                    final_response_content = process_text(raw_ai_response)
                    print(
                        f"DEBUG: Returning formatted stock price response: {final_response_content}"
                    )

                    if trace:
                        trace.event(
                            name="stock_price_response",
                            output={
                                "ticker": ticker,
                                "market_context_needed": needs_market_context,
                            },
                        )
                    early_exit = True
                else:
                    # Continue with the original flow - check if web search is needed
                    search_needed = await needs_web_search(
                        user_query,
                        client,
                        langfuse_client,
                        trace,
                        conversation_history=loaded_history if loaded_history else None,
                    )
                    system_prompt = (
                        "You are a financial assistant specializing in stocks, cryptocurrency, and trading. "
                        "Use the conversation history provided. You must provide very clear and explicit answers in USD. "
                        "If the user asks for a recommendation, give a direct 'You should...' statement. Use provided tools when necessary. "
                        "Ensure all prices are presented in USD. "
                        "Always interpret and present dates in European format (day before month, e.g., DD-MM-YYYY). If the date is ambiguous, ask the user for clarification. "
                        "If the user refers back to previous turns in the conversation (for example, by using pronouns like 'it', 'its', 'her', or 'his'), resolve those pronouns to the most relevant entity from the previous conversation turns (such as a stock, company, or asset). "
                        "If the reference is ambiguous, ask the user for clarification."
                    )

                    base_messages = [
                        {"role": "system", "content": system_prompt}
                    ] + loaded_history

                    llm_call_span = None
                    if trace:
                        llm_call_span = trace.span(
                            name="main_llm_call", input={"search_needed": search_needed}
                        )

                    if search_needed:
                        print("DEBUG: Web search determined NEEDED.")
                        search_span = None
                        if trace:
                            search_span = trace.span(name="web_search")

                        search_result_text = duckduckgo_search(
                            f"{user_query} price in USD on {datetime.now():%B %d, %Y}"
                        )
                        web_search_result = search_result_text
                        web_search_used = True

                        if search_span:
                            search_span.end(
                                output={"results_length": len(search_result_text)}
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

                        if trace:
                            trace.event(
                                name="web_search_used",
                                output={"search_query": user_query},
                            )
                    else:
                        print("DEBUG: Web search determined NOT needed.")
                        messages_sent_to_openai = base_messages + [
                            current_user_message_dict
                        ]
                        print("DEBUG: Making LLM call with History + Current Query...")

                    print(
                        f"DEBUG: TOTAL messages being sent to OpenAI: {len(messages_sent_to_openai)}"
                    )
                    if messages_sent_to_openai:
                        print(
                            f"DEBUG: First message sent: {messages_sent_to_openai[0]}"
                        )
                        if len(messages_sent_to_openai) > 1:
                            print(
                                f"DEBUG: Last message sent: {messages_sent_to_openai[-1]}"
                            )

                    generation_start_time = datetime.now()
                    openai_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages_sent_to_openai,
                        tools=available_tools,
                        tool_choice="auto",
                    )

                    if llm_call_span and trace:
                        llm_call_span.generation(
                            name="openai_completion",
                            model="gpt-4o-mini",
                            input=messages_sent_to_openai,
                            output=(
                                openai_response.choices[0].message
                                if openai_response.choices
                                else None
                            ),
                            start_time=generation_start_time,
                            end_time=datetime.now(),
                        )

                    response_message = openai_response.choices[0].message
                    if response_message.tool_calls:
                        print(f"DEBUG: Tool call(s) requested...")
                        tool_call_span = None
                        if trace:
                            tool_call_span = trace.span(name="tool_calls_handling")

                        raw_ai_response = await handle_tool_calls(
                            response_message,
                            user_query,
                            messages_sent_to_openai,
                            client,
                            langfuse_client,
                            tool_call_span,
                        )

                        if tool_call_span:
                            tool_call_span.end()

                    else:
                        raw_ai_response = response_message.content
                        print(
                            f"DEBUG: Direct text response received snippet: {raw_ai_response[:50] if raw_ai_response else 'None'}..."
                        )

                    if llm_call_span:
                        llm_call_span.end()

                    if not raw_ai_response:
                        print("DEBUG: ERROR - No final content generated.")
                        final_response_content = "I encountered an issue."
                        raw_ai_response = None

                        if trace:
                            trace.update(
                                status="error",
                                error={"message": "No final content generated"},
                            )
                    else:
                        final_response_content = process_text(raw_ai_response)
                        print(
                            f"DEBUG: Returning formatted response: {final_response_content}"
                        )

    # Error Handling
    except BadRequestError as bre:
        print(f"DEBUG: ERROR - OpenAI Bad Request: {bre}")
        raw_ai_response = None
        final_response_content = "API Error: Bad Request."

        if trace:
            trace.update(
                status="error",
                error={
                    "message": f"API Error: {bre.body.get('message', 'Bad Request')}"
                },
            )

        raise HTTPException(
            status_code=400,
            detail=f"API Error: {bre.body.get('message', 'Bad Request')}",
        )
    except HTTPException as http_exc:
        raw_ai_response = None
        final_response_content = (
            http_exc.detail if hasattr(http_exc, "detail") else "HTTP error."
        )

        if trace:
            trace.update(
                status="error",
                error={"message": http_exc.detail, "status_code": http_exc.status_code},
            )

        raise http_exc
    except Exception as e:
        import traceback

        print(f"DEBUG: ERROR - Critical error in chat endpoint: {e}")
        traceback.print_exc()
        raw_ai_response = None
        final_response_content = "Internal server error."

        if trace:
            trace.update(
                status="error",
                error={"message": "Internal server error", "details": str(e)},
            )

        raise HTTPException(status_code=500, detail="Internal server error.")

    # If web search was used, synthesize tool and web search results with LLM before guardrails
    if web_search_used and web_search_result and not early_exit:
        final_response_content = await synthesize_search_results(
            user_query, final_response_content, web_search_result, client, trace
        )

    # Apply guardrails and LLM review
    if not early_exit:
        final_response_content = await apply_response_guardrails(
            final_response_content, user_query, client, trace
        )

    # Always update trace with final response
    if trace:
        trace.update(output={"response": final_response_content}, status="success")

    return (
        final_response_content,
        current_user_message_dict,
        raw_ai_response,
        early_exit,
    )
