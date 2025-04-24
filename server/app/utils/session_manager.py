import json
import uuid
from typing import List, Dict, Any, Tuple, Optional
from app.config import SESSION_TTL_SECONDS, MAX_HISTORY_MESSAGES


async def load_chat_history(
    redis_conn, session_id: str, trace=None
) -> Tuple[List[Dict[str, Any]], str, bool]:
    """
    Load chat history from Redis for a given session ID.

    Args:
        redis_conn: Redis connection
        session_id: Session ID to load history for
        trace: Optional Langfuse trace for instrumentation

    Returns:
        Tuple of (loaded_history, session_id, is_new_session)
        If session_id is None or history is corrupted, a new session will be created
    """
    loaded_history = []
    is_new_session = False

    if not redis_conn or not session_id:
        return create_new_session(redis_conn)

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

    # If session_id was reset to None, create a new session
    if not session_id:
        return create_new_session(redis_conn)

    return loaded_history, session_id, is_new_session


def create_new_session(redis_conn) -> Tuple[List[Dict[str, Any]], str, bool]:
    """Create a new chat session with a unique ID"""
    session_id = str(uuid.uuid4())
    print(f"DEBUG: Generated NEW session ID: {session_id[-6:]}")
    return [], session_id, True


async def save_chat_history(
    redis_conn,
    session_id: str,
    loaded_history: List[Dict[str, Any]],
    user_message: Dict[str, Any],
    assistant_response: str,
    trace=None,
) -> bool:
    """
    Save updated chat history to Redis

    Args:
        redis_conn: Redis connection
        session_id: Session ID
        loaded_history: Existing history
        user_message: Current user message
        assistant_response: Assistant's response to the user
        trace: Optional Langfuse trace

    Returns:
        Success status
    """
    if not redis_conn or not session_id or not assistant_response:
        return False

    try:
        history_save_span = None
        if trace:
            history_save_span = trace.span(name="save_history")

        new_history_entry = [
            user_message,
            {"role": "assistant", "content": assistant_response},
        ]
        updated_history = loaded_history + new_history_entry
        if len(updated_history) > MAX_HISTORY_MESSAGES:
            updated_history = updated_history[-MAX_HISTORY_MESSAGES:]
            print(f"DEBUG: History truncated to {len(updated_history)} messages.")

        history_to_save_json = json.dumps(updated_history)
        print(
            f"DEBUG: Attempting to save history for session {session_id[-6:]}. Size: {len(updated_history)} messages."
        )
        await redis_conn.set(session_id, history_to_save_json, ex=SESSION_TTL_SECONDS)
        print(f"DEBUG: History save successful for session {session_id[-6:]}.")

        if history_save_span:
            history_save_span.end(output={"messages_saved": len(updated_history)})
        return True
    except Exception as e:
        print(f"DEBUG: ERROR - Redis SET failed: {e}. History not saved.")

        if trace:
            trace.event(name="history_save_error", output={"error": str(e)})
        return False


def set_session_cookie(response, session_id: str, trace=None) -> None:
    """
    Set session cookie for cross-origin requests

    Args:
        response: FastAPI Response object
        session_id: Session ID to set in cookie
        trace: Optional Langfuse trace
    """
    print(f"DEBUG: Setting CROSS-ORIGIN cookie for new session {session_id[-6:]}...")
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
        trace.event(name="session_cookie_set", output={"session_id": session_id})
