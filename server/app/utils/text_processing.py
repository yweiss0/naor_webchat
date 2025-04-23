import re


def process_text(text: str) -> str:
    """
    Clean up text by removing code blocks markers and ensuring it ends with punctuation.
    """
    text = re.sub(r"```(json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    text_stripped = text.strip()
    if text_stripped and text_stripped[-1] not in ".!?":
        text_stripped += "."
    return text_stripped


def validate_usd_result(text: str) -> bool:
    """
    Check if the text contains USD currency indicators.
    """
    return "$" in text or "USD" in text.upper()


def extract_date_from_query(query: str) -> str:
    """
    Extracts a date from the query if present.
    Handles various formats including:
    - Natural language: "April 22 2024", "April 22nd, 2024"
    - Month abbreviations: "Apr 16, 2025", "apr 16 25"
    - Day first: "16 April 2025", "16 april 25"
    - Relative dates: "yesterday", "last week", etc.
    - DD.MM.YYYY: "01.05.2024", "1.5.2024"
    - DD/MM/YYYY: "01/05/2024", "1/5/2024"
    - DD.MM.YY: "01.05.24", "1.5.24"
    - DD/MM/YY: "01/05/24", "1/5/24"

    Returns an empty string if no date is found.
    """
    query_lower = query.lower()

    # Check for relative date terms
    if "yesterday" in query_lower:
        return "yesterday"
    if "last week" in query_lower:
        return "last week"
    if "last month" in query_lower:
        return "last month"
    if "last year" in query_lower:
        return "last year"

    # Check for common date patterns

    # DD.MM.YYYY format (e.g., 01.05.2024 or 1.5.2024)
    dot_full_year = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", query)
    if dot_full_year:
        day, month, year = dot_full_year.groups()
        return f"{int(day):02d}.{int(month):02d}.{year}"

    # DD/MM/YYYY format (e.g., 01/05/2024 or 1/5/2024)
    slash_full_year = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", query)
    if slash_full_year:
        day, month, year = slash_full_year.groups()
        return f"{int(day):02d}.{int(month):02d}.{year}"

    # DD.MM.YY format (e.g., 01.05.24 or 1.5.24)
    dot_short_year = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2})\b", query)
    if dot_short_year:
        day, month, year = dot_short_year.groups()
        full_year = (
            f"20{year}" if int(year) < 50 else f"19{year}"
        )  # Assume 20xx for years < 50
        return f"{int(day):02d}.{int(month):02d}.{full_year}"

    # DD/MM/YY format (e.g., 01/05/24 or 1/5/24)
    slash_short_year = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b", query)
    if slash_short_year:
        day, month, year = slash_short_year.groups()
        full_year = (
            f"20{year}" if int(year) < 50 else f"19{year}"
        )  # Assume 20xx for years < 50
        return f"{int(day):02d}.{int(month):02d}.{full_year}"

    # Full month names and month abbreviations
    months = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }

    # Format: "Month DD, YYYY" or "Month DD YYYY" or "Month DD YY" (e.g., "April 16, 2025" or "apr 16 25")
    for month_name, month_num in months.items():
        # Check for "Month DD, YYYY" or "Month DD YYYY" or "Month DD YY"
        pattern = (
            rf"\b{month_name}\s+(\d{{1,2}})(st|nd|rd|th)?\s*,?\s*(\d{{4}}|\d{{2}})\b"
        )
        match = re.search(pattern, query_lower)
        if match:
            day = match.group(1)
            year = match.group(3)

            # Handle 2-digit years
            if len(year) == 2:
                year = f"20{year}" if int(year) < 50 else f"19{year}"

            return f"{int(day):02d}.{month_num:02d}.{year}"

    # Format: "DD Month YYYY" or "DD Month YY" (e.g., "16 April 2025" or "16 april 25")
    for month_name, month_num in months.items():
        pattern = rf"\b(\d{{1,2}})(st|nd|rd|th)?\s+{month_name}\s+(\d{{4}}|\d{{2}})\b"
        match = re.search(pattern, query_lower)
        if match:
            day = match.group(1)
            year = match.group(3)

            # Handle 2-digit years
            if len(year) == 2:
                year = f"20{year}" if int(year) < 50 else f"19{year}"

            return f"{int(day):02d}.{month_num:02d}.{year}"

    return ""


def normalize_ticker(ticker: str, query: str) -> str:
    """
    Normalizes ticker symbols, especially for ambiguous cases like gold.
    """
    query_lower = query.lower()
    ticker_upper = ticker.upper()

    # Handle gold-related queries
    if ticker_upper in ["GOLD", "XAU", "XAUUSD", "XAU/USD", "GLD"] or any(
        gold_term in query_lower for gold_term in ["gold", "xau", "gold price"]
    ):
        print("Normalizing gold-related ticker to GC=F")
        return "GC=F"  # Yahoo Finance ticker for gold futures

    # Return the ticker as is in other cases
    return ticker_upper
