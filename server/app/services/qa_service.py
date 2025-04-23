from openai import OpenAI
from langfuse import Langfuse
from datetime import datetime
from typing import Tuple, Optional, Dict, Any


def load_qa_file() -> dict:
    """Load the Q&A file and return a dictionary of questions and answers."""
    qa_dict = {}
    try:
        with open("qna_output.txt", "r", encoding="utf-8") as file:
            content = file.read()
            # Split by Q: to get individual Q&A pairs
            qa_pairs = content.split("Q: ")
            for pair in qa_pairs[1:]:  # Skip the first empty element
                if "A: " in pair:
                    question, answer = pair.split("A: ", 1)
                    # Clean up the question and answer
                    question = question.strip()
                    answer = answer.strip()
                    qa_dict[question] = answer
        print(f"Loaded {len(qa_dict)} Q&A pairs from file.")
        return qa_dict
    except Exception as e:
        print(f"Error loading Q&A file: {e}")
        return {}


async def find_qa_match(
    user_query: str,
    client: Optional[OpenAI],
    langfuse_client: Optional[Langfuse] = None,
    parent_span: Any = None,
) -> Tuple[bool, str]:
    """
    Check if the user query matches a question in the Q&A file.
    Returns a tuple of (is_match, answer_or_empty_string)
    """
    if not client:
        print("OpenAI client not available for Q&A matching.")
        return (False, "")

    # Create a span for this operation if Langfuse is available
    span = None
    if langfuse_client and parent_span:
        span = parent_span.span(
            name="find_qa_match",
            input={"query": user_query},
        )

    # Load the Q&A file
    qa_dict = load_qa_file()
    if not qa_dict:
        print("Q&A dictionary is empty, cannot find match.")
        if langfuse_client and span:
            span.end(
                output={"error": "Q&A dictionary is empty", "is_match": False},
                status="error",
            )
        return (False, "")

    print(f"Checking if query matches Q&A file: '{user_query}'")
    try:
        # Create a prompt for the LLM to find the best match
        qa_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_dict.items()])

        classification_messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that matches user questions to predefined Q&A pairs.
                Analyze the user query and determine if it matches any of the questions in the provided Q&A pairs.
                If there's a match, respond with the exact answer from the Q&A pair.
                If there's no match, respond with 'NO_MATCH'.
                
                Consider semantic similarity, not just exact matches. The user might phrase the question differently
                but be asking about the same topic.""",
            },
            {
                "role": "user",
                "content": f"""User Query: "{user_query}"

Q&A Pairs:
{qa_text}

If there's a match, provide ONLY the exact answer from the matching Q&A pair.
If there's no match, respond with ONLY 'NO_MATCH'.""",
            },
        ]

        start_time = datetime.now()
        if langfuse_client and span:
            # Use Langfuse-instrumented OpenAI client if available
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=classification_messages,
                temperature=0.0,
            )

            # Capture the generation with Langfuse
            span.generation(
                name="qa_match_classification",
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
                temperature=0.0,
            )

        # Parse the response content
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result_text = response.choices[0].message.content.strip()

            if result_text == "NO_MATCH":
                print("No Q&A match found for the query.")
                if langfuse_client and span:
                    span.end(output={"is_match": False})
                return (False, "")
            else:
                # Remove "A: " prefix if it exists
                if result_text.startswith("A: "):
                    result_text = result_text[3:].strip()

                print("Q&A match found for the query.")
                if langfuse_client and span:
                    span.end(output={"is_match": True, "answer": result_text})
                return (True, result_text)
        else:
            print(
                "Warning: Could not parse Q&A matching response. Defaulting to no match."
            )
            if langfuse_client and span:
                span.end(
                    output={
                        "error": "Could not parse Q&A matching response",
                        "is_match": False,
                    }
                )
            return (False, "")

    except Exception as e:
        print(f"Error during Q&A matching: {e}")
        if langfuse_client and span:
            span.end(output={"error": str(e), "is_match": False}, status="error")
        return (False, "")
