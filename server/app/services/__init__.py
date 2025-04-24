# Services module initialization

# Import classification-related services
from app.services.query_classification import is_related_to_stocks_crypto
from app.services.price_query_detector import is_stock_price_query
from app.services.web_search_classifier import needs_web_search

# Import other services
from app.services.tool_handler import handle_tool_calls, available_tools
from app.services.qa_service import find_qa_match
from app.services.web_search import duckduckgo_search
from app.services.price_handler import handle_stock_price_query
from app.services.market_data import get_stock_price
