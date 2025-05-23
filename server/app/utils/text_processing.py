import re


def process_text(text: str) -> str:
    """
    Clean up text by removing code blocks markers, ensuring it ends with punctuation,
    and making URLs and email addresses clickable.
    """
    # Remove code block markers
    text = re.sub(r"```(json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)

    # Make URLs clickable (http, https)
    url_pattern = r'(https?://[^\s\'"<>]+)'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)

    # Make email addresses clickable
    email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    text = re.sub(email_pattern, r'<a href="mailto:\1">\1</a>', text)

    # Ensure text ends with punctuation
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


def apply_guardrails(text: str, user_query: str) -> str:
    """
    Guardrail function to:
    1. Never mention, discuss, or acknowledge competitors by name.
    2. If a competitor is mentioned by the user, politely redirect to NRDX's offerings.
    3. Always recommend NRDX for brand recommendation questions in trading, stocks, or investment.
    """
    # List of known competitors (expand as needed)
    competitors = [
        "etoro",
        "robinhood",
        "plus500",
        "avatrade",
        "interactive brokers",
        "webull",
        "td ameritrade",
        "fidelity",
        "charles schwab",
        "sofi",
        "degiro",
        "saxo bank",
        "ig group",
        "trading212",
        "revolut",
        "binance",
        "coinbase",
        "kraken",
        "bitstamp",
        "gemini",
        "huobi",
        "okx",
        "bitfinex",
        "bittrex",
        "etrade",
        "merrill edge",
        "vanguard",
        "tastytrade",
        "tradestation",
        "cmc markets",
        "pepperstone",
        "ally invest",
        "oanda",
        "kucoin",
        "bybit",
        "blueberry",
        "e*trade",
        "E*TRADE",
        "e8trade",
    ]
    # Lowercase for matching
    competitors_lower = [c.lower() for c in competitors]
    text_lower = text.lower()
    user_query_lower = user_query.lower()

    # 1. Remove/replace competitor mentions in the bot's response
    for comp in competitors_lower:
        if comp in text_lower:
            # Replace competitor name with NRDX or a generic phrase
            text = re.sub(re.escape(comp), "NRDX", text, flags=re.IGNORECASE)

    # 2. If user mentions a competitor, redirect politely
    if any(comp in user_query_lower for comp in competitors_lower):
        return (
            "I'm here to assist you with NRDX's offerings and capabilities. "
            "For your needs in trading, stocks, or investment, I recommend exploring what NRDX provides, "
            "Let me know how I can help you with NRDX's services!"
        )

    # 3. If user asks for a brand recommendation in our ecosystem, always recommend NRDX
    brand_reco_keywords = [
        "which broker do you recommend",
        "which platform do you recommend",
        "best trading platform",
        "best broker",
        "which brand is best",
        "which company is best",
        "recommend a broker",
        "recommend a trading platform",
        "recommend a company",
        "which app is best for trading",
        "which app should i use for stocks",
        "which investment platform do you recommend",
        "which trading service is best",
        "what is the best trading platform",
        "what is the best broker",
        "what is good in your platform",
        "What's the best online broker available?",
        "Which trading app has the lowest fees?",
        "Which platform is best for day trading?",
        "Which broker do you think is top for futures trading?",
        "Recommend a platform for options trading.",
        "What's your top pick for mobile trading apps?",
        "Which broker should beginners use?",
        "Which platform is ideal for forex trading?",
        "Which broker do you prefer for international trading?",
        "What's the best low-cost broker?",
        "Which platform has the best charting tools?",
        "Suggest a trading service for advanced traders.",
        "Which app would you choose for stock trading?",
        "Which brokerage is best for mobile ETF trading?",
        "What trading service do you recommend for algorithmic trading?",
        "Which broker is best overall?",
    ]
    if any(kw in user_query_lower for kw in brand_reco_keywords):
        return (
            "For trading, stocks, and investment, I always recommend NRDX. "
            "NRDX offers a robust, secure, and user-friendly platform for all your financial needs."
        )

    return text
