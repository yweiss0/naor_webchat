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
from app.endpoints.chat_handler import process_chat_request
from app.utils.session_manager import (
    load_chat_history,
    save_chat_history,
    set_session_cookie,
)
import json
import uuid
import traceback
from datetime import datetime
from typing import Any

router = APIRouter()


@router.post("/chat")
async def chat(query: QueryRequest, request: Request, response: Response):
    """
    Chat endpoint that handles user queries about stocks, crypto, and financial data

    This endpoint:
    1. Loads or creates a user session
    2. Processes the user query
    3. Manages conversation history
    4. Returns a response with appropriate cookie headers
    """
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

    # Load or create chat history
    loaded_history, session_id, is_new_session = await load_chat_history(
        redis_conn, session_id, trace
    )

    # Update trace with session ID if this is a new session
    if is_new_session and trace:
        trace.update(
            user_id=session_id,
            session_id=session_id,
            metadata={"new_session_created": True},
        )

    try:
        # Process the user's query
        assistant_response, user_message_dict, raw_response, early_exit = (
            await process_chat_request(
                user_query,
                client,
                redis_conn,
                session_id,
                loaded_history,
                trace,
                langfuse_client,
            )
        )

        # Save history for future reference
        if redis_conn and session_id:
            await save_chat_history(
                redis_conn,
                session_id,
                loaded_history,
                user_message_dict,
                assistant_response,
                trace,
            )

        # Set session cookie if this is a new session
        if is_new_session and session_id:
            set_session_cookie(response, session_id, trace)

        print(
            f"--- Request End (Session: {session_id[-6:] if session_id else 'NEW'}) ---"
        )
        return {"response": assistant_response}

    except BadRequestError as bre:
        print(f"DEBUG: ERROR - OpenAI Bad Request: {bre}")
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
        if trace:
            trace.update(
                status="error",
                error={"message": http_exc.detail, "status_code": http_exc.status_code},
            )
        raise http_exc
    except Exception as e:
        print(f"DEBUG: ERROR - Critical error in chat endpoint: {e}")
        traceback.print_exc()
        if trace:
            trace.update(
                status="error",
                error={"message": "Internal server error", "details": str(e)},
            )
        raise HTTPException(status_code=500, detail="Internal server error.")


def is_reason_query(query: str) -> bool:
    """
    Returns True if the query is asking for reasons (contains 'why' or 'reason').
    """
    q = query.lower()
    return ("why" in q) or ("reason" in q)
