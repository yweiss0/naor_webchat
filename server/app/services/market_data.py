import yfinance as yf
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union


def get_stock_price(
    ticker: str, date: str = None, include_history: bool = False
) -> tuple[Union[float, str], Optional[Dict[str, Any]]]:
    """
    Get the current price or historical price for a specific date and optionally historical data for a ticker.
    Returns a tuple of (price, historical_data).
    """
    try:
        stock = yf.Ticker(ticker)
        if date:
            # Parse the date string to a datetime object
            try:
                target_date = datetime.strptime(date, "%d.%m.%Y")
            except ValueError:
                # Handle relative date terms
                today = datetime.now()
                if date.lower() == "yesterday":
                    target_date = today - timedelta(days=1)
                elif date.lower() == "last week":
                    target_date = today - timedelta(weeks=1)
                elif date.lower() == "last month":
                    target_date = today - timedelta(weeks=4)
                elif date.lower() == "last year":
                    target_date = today - timedelta(weeks=52)
                else:
                    return (
                        f"Invalid date format or unrecognized relative date term: {date}",
                        None,
                    )

            # Fetch historical data for the specific date
            hist = stock.history(start=target_date, end=target_date + timedelta(days=1))
            if not hist.empty:
                price = hist["Close"].iloc[0]
                return round(float(price), 2), None
            else:
                return (
                    f"No data available for {ticker} on {target_date.strftime('%d.%m.%Y')}",
                    None,
                )
        else:
            # Get current price
            price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))

            historical_data = None
            if include_history:
                # Get 5-day history for weekly context
                hist = stock.history(period="5d")
                if not hist.empty:
                    historical_data = {
                        "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                        "open": hist["Open"].tolist(),
                        "high": hist["High"].tolist(),
                        "low": hist["Low"].tolist(),
                        "close": hist["Close"].tolist(),
                        "volume": (
                            hist["Volume"].tolist()
                            if "Volume" in hist.columns
                            else None
                        ),
                    }

            if price is not None:
                return round(float(price), 2), historical_data

            # Fallback to historical data if current price not available
            if historical_data:
                return round(float(historical_data["close"][-1]), 2), historical_data

            hist = stock.history(period="1d")
            if not hist.empty:
                return round(float(hist["Close"].iloc[-1]), 2), None

            return f"Could not retrieve price for {ticker}.", None
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return f"Error retrieving price for {ticker}.", None
