# This file is deprecated and will be removed in a future version.
# Use the specialized classification modules instead.

from app.services.query_classification import is_related_to_stocks_crypto
from app.services.price_query_detector import is_stock_price_query
from app.services.web_search_classifier import needs_web_search

# Re-export the functions for backwards compatibility
__all__ = ["is_related_to_stocks_crypto", "is_stock_price_query", "needs_web_search"]
