from typing import Optional, Dict, Any
from datetime import datetime
from app.utils.text_processing import process_text, apply_guardrails


async def synthesize_search_results(
    user_query: str,
    original_response: str,
    web_search_result: str,
    client: Any,
    trace: Optional[Any] = None,
) -> str:
    """
    Synthesizes LLM response with web search results to provide a comprehensive answer

    Args:
        user_query: Original query from user
        original_response: LLM response before synthesis
        web_search_result: Web search results text
        client: OpenAI client
        trace: Optional Langfuse trace

    Returns:
        Synthesized response
    """
    if not web_search_result:
        return original_response

    print(
        "DEBUG: Synthesizing tool and web search results with LLM before guardrails..."
    )
    synthesis_prompt = (
        "You are a helpful, polite, and professional assistant. "
        "Combine the following financial data and recent web search results into a single, clear, and helpful answer for the user. "
        "Base your answer specifically on the user's question. Do not copy-paste, but synthesize in your own words.\n\n"
        f"User question: {user_query}\n\nFinancial data:\n{original_response}\n\nRecent web search results:\n{web_search_result}\n"
    )
    synthesis_messages = [
        {"role": "system", "content": synthesis_prompt},
    ]

    # Create trace span if available
    synthesis_span = None
    if trace:
        synthesis_span = trace.span(
            name="llm_synthesis_tool_and_websearch",
            input={
                "user_query": user_query,
                "tool_data": original_response,
                "web_search": web_search_result[:200],
            },
        )

    try:
        synthesis_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=synthesis_messages,
            max_tokens=512,
            temperature=0.2,
        )

        if (
            synthesis_response.choices
            and synthesis_response.choices[0].message
            and synthesis_response.choices[0].message.content
        ):
            result = synthesis_response.choices[0].message.content.strip()

            if trace and synthesis_span:
                synthesis_span.end(output={"synthesized_answer": result})

            return result
        else:
            # Return original if synthesis failed
            if trace and synthesis_span:
                synthesis_span.end(
                    output={"error": "No content in synthesis response"}, status="error"
                )
            return original_response

    except Exception as e:
        print(f"DEBUG: LLM synthesis step failed: {e}")
        if trace and synthesis_span:
            synthesis_span.end(output={"error": str(e)}, status="error")
        # fallback: keep original response
        return original_response


async def apply_response_guardrails(
    content: str, user_query: str, client: Any, trace: Optional[Any] = None
) -> str:
    """
    Apply guardrails and LLM review to ensure the response meets quality standards

    Args:
        content: Original response content
        user_query: The user's query
        client: OpenAI client
        trace: Optional Langfuse trace

    Returns:
        Processed and guarded response
    """
    # First apply basic guardrails
    guarded_response = apply_guardrails(content, user_query)
    guardrails_applied = guarded_response != content

    # Then do LLM review for politeness, clarity and competitor filtering
    llm_reviewed_response = guarded_response
    try:
        llm_review_span = None
        review_prompt = (
            "You are a helpful, polite, and professional assistant for NRDX. "
            "Review the following response to ensure it is polite, clear, and ONLY mentions NRDX as the platform or brand. "
            "The answer must not imply, suggest, or hint that there are other brands, platforms, or options available, nor that any names were omitted or replaced. "
            "Rewrite the answer if needed so that it presents NRDX as the sole recommended and discussed platform, with no reference to other options, even implicitly. "
            "If the response is already good and only mentions NRDX, return it unchanged. "
            "Do NOT remove relevant information or make the answer less helpful. "
            "Here is the response to review: '" + guarded_response + "'"
        )

        if trace:
            llm_review_span = trace.span(
                name="llm_guardrails_review", input={"review_prompt": review_prompt}
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
                    "original_message": content,
                    "guardrails_response": guarded_response,
                    "llm_reviewed_response": llm_reviewed_response,
                }
            )
    except Exception as e:
        print(f"DEBUG: LLM review step failed: {e}")
        # Fallback to guarded_response if LLM review fails
        llm_reviewed_response = guarded_response

    return llm_reviewed_response


def is_reason_query(query: str) -> bool:
    """
    Returns True if the query is asking for reasons (contains 'why' or 'reason').
    """
    q = query.lower()
    return ("why" in q) or ("reason" in q)
