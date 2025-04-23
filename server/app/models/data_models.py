from pydantic import BaseModel


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    message: str
