from fastapi import APIRouter, Request, Response, HTTPException
from openai import BadRequestError
from openai import OpenAI
from langfuse import Langfuse
from app.models.data_models import QueryRequest
from app.services.classification import (
    is_related_to_stocks_crypto,
    is_stock_price_query,
    needs_web_search,
)
from app.services.qa_service import find_qa_match
from app.services.price_handler import handle_stock_price_query
from app.services.web_search import duckduckgo_search
from app.services.tool_handler import handle_tool_calls, available_tools
from app.utils.text_processing import process_text, apply_guardrails
from app.config import SESSION_TTL_SECONDS, MAX_HISTORY_MESSAGES
import json
import uuid
import traceback
from datetime import datetime

router = APIRouter()


@router.post("/chat")
async def chat(query: QueryRequest, request: Request, response: Response):
    redis_conn = request.app.state.redis_conn
    client = request.app.state.openai_client
    langfuse_client = request.app.state.langfuse

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

    # Start a Langfuse trace for the entire request
    trace = None
    if langfuse_client:
        client_ip = request.client.host if request.client else "unknown"
        trace = langfuse_client.trace(
            name="chat_request",
            user_id=session_id if session_id else "new_user",
            session_id=session_id if session_id else "new_session",
            metadata={
                "ip": client_ip,
                "query": user_query,
                "new_session": session_id is None,
            },
            input={"message": user_query},
        )
        print(f"DEBUG: Langfuse trace created with ID: {trace.id}")

    print(
        f"\n--- Request Start (Session: {session_id[-6:] if session_id else 'NEW'}) ---"
    )
    print(f"DEBUG: User Query: {user_query}")
    print(f"DEBUG: Cookie 'chatbotSessionId' value: {session_id}")

    # Load History
    if redis_conn and session_id:
        history_span = None
        if trace:
            history_span = trace.span(name="load_history")

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
                            output={"error": "Corrupt history type", "status": "reset"},
                            status="error",
                        )
                else:
                    print(f"DEBUG: Successfully loaded {len(loaded_history)} messages.")
                    if loaded_history:
                        print(f"DEBUG: Last loaded msg: {loaded_history[-1]}")
                    await redis_conn.expire(session_id, SESSION_TTL_SECONDS)

                    if history_span:
                        history_span.end(output={"messages_count": len(loaded_history)})
            else:
                print(f"DEBUG: Session ID {session_id[-6:]} not found in Redis.")
                session_id = None

                if history_span:
                    history_span.end(
                        output={"error": "Session not found", "status": "reset"},
                        status="error",
                    )
        except json.JSONDecodeError as json_err:
            print(
                f"DEBUG: ERROR - JSON Decode failed for session {session_id}: {json_err}. Resetting."
            )
            session_id = None

            if history_span:
                history_span.end(
                    output={
                        "error": f"JSON Decode error: {str(json_err)}",
                        "status": "reset",
                    },
                    status="error",
                )
        except Exception as e:
            print(
                f"DEBUG: ERROR - Redis GET/EXPIRE failed: {e}. Proceeding without history."
            )

            if history_span:
                history_span.end(
                    output={"error": f"Redis error: {str(e)}", "status": "continue"},
                    status="error",
                )

    # Create New Session ID
    if redis_conn and not session_id:
        is_new_session = True
        session_id = str(uuid.uuid4())
        print(f"DEBUG: Generated NEW session ID: {session_id[-6:]}")
        loaded_history = []

        # Update trace with new session ID if we created one
        if trace:
            trace.update(
                user_id=session_id,
                session_id=session_id,
                metadata={"new_session_created": True},
            )

    # Core Logic
    final_response_content = ""
    raw_ai_response = None
    messages_sent_to_openai = []

    # Define current_user_message_dict at the beginning to ensure it's always available
    current_user_message_dict = {"role": "user", "content": user_query}

    try:
        # First check if the query matches a question in the Q&A file
        qa_match, qa_answer = await find_qa_match(
            user_query, client, langfuse_client, trace
        )

        early_exit = False
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

    # Apply guardrails before returning the response
    guarded_response = apply_guardrails(final_response_content, user_query)
    guardrails_applied = guarded_response != final_response_content

    # LLM check for politeness, clarity, and competitor filtering
    llm_reviewed_response = guarded_response
    try:
        llm_review_span = None
        if trace:
            llm_review_span = trace.span(
                name="llm_guardrails_review", input={"review_prompt": review_prompt}
            )
        review_prompt = (
            "You are a helpful, polite, and professional assistant for NRDX. "
            "Review the following response to ensure it is polite, clear, and ONLY mentions NRDX as the platform or brand. "
            "The answer must not imply, suggest, or hint that there are other brands, platforms, or options available, nor that any names were omitted or replaced. "
            "Rewrite the answer if needed so that it presents NRDX as the sole recommended and discussed platform, with no reference to other options, even implicitly. "
            "If the response is already good and only mentions NRDX, return it unchanged. "
            "Do NOT remove relevant information or make the answer less helpful. "
            "Here is the response to review: '" + guarded_response + "'"
        )
        review_messages = [
            {"role": "system", "content": review_prompt},
            {
                "role": "user",
                "content": f"User Query: {user_query}\nResponse: {guarded_response}",
            },
        ]
        review_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=review_messages,
            max_tokens=512,
            temperature=0.0,
        )
        if (
            review_response.choices
            and review_response.choices[0].message
            and review_response.choices[0].message.content
        ):
            llm_reviewed_response = review_response.choices[0].message.content.strip()
        if llm_review_span:
            llm_review_span.end(
                output={
                    "guardrails_applied": guardrails_applied,
                    "original_message": final_response_content,
                    "guardrails_response": guarded_response,
                    "llm_reviewed_response": llm_reviewed_response,
                }
            )
    except Exception as e:
        print(f"DEBUG: LLM review step failed: {e}")
        # Fallback to guarded_response if LLM review fails
        llm_reviewed_response = guarded_response

    # Update the main trace output with the final response
    if trace:
        trace.update(output={"response": llm_reviewed_response}, status="success")

    # --- Save History ---
    if redis_conn and session_id and llm_reviewed_response:
        try:
            history_save_span = None
            if trace:
                history_save_span = trace.span(name="save_history")

            new_history_entry = [
                current_user_message_dict,
                {"role": "assistant", "content": llm_reviewed_response},
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
                history_save_span.end(output={"messages_saved": len(updated_history)})
        except Exception as e:
            print(f"DEBUG: ERROR - Redis SET failed: {e}. History not saved.")

            if trace:
                error_event = trace.event(
                    name="history_save_error", output={"error": str(e)}
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

        if trace:
            cookie_event = trace.event(
                name="session_cookie_set", output={"session_id": session_id}
            )

    print(f"--- Request End (Session: {session_id[-6:] if session_id else 'NEW'}) ---")
    return {"response": llm_reviewed_response}
