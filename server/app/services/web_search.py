from duckduckgo_search import DDGS


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using DuckDuckGo and return formatted results.
    Tries progressively longer time periods if not enough results are found.
    """
    print(f"DDG Search: '{query}'")
    time_period_used = "unknown"
    try:
        with DDGS() as ddgs:
            # First try to get results from the last week
            results = [
                r
                for r in ddgs.text(
                    query, max_results=max_results, timelimit="w"  # 'w' for week
                )
            ]
            time_period_used = "last week"

            # If not enough results, try the last month
            if len(results) < 2:
                print("Not enough results from the last week, trying the last month...")
                results = [
                    r
                    for r in ddgs.text(
                        query, max_results=max_results, timelimit="m"  # 'm' for month
                    )
                ]
                time_period_used = "last month"

                # If still not enough results, try the last year
                if len(results) < 2:
                    print(
                        "Not enough results from the last month, trying the last year..."
                    )
                    results = [
                        r
                        for r in ddgs.text(
                            query,
                            max_results=max_results,
                            timelimit="y",  # 'y' for year
                        )
                    ]
                    time_period_used = "last year"

            if not results:
                return "No relevant information found."

            # Format results with dates if available
            formatted_results = []
            for res in results:
                title = res.get("title", "No Title")
                body = res.get("body", "No snippet.")
                date = res.get("date", "")

                # Add date to the result if available
                if date:
                    formatted_results.append(f"- {title} ({date}): {body}")
                else:
                    formatted_results.append(f"- {title}: {body}")

            print(
                f"DEBUG: Using search results from the {time_period_used} ({len(results)} results found)"
            )
            return "\n".join(formatted_results)
    except Exception as e:
        print(f"DDG Error: {e}")
        return "Error performing web search."
