from contextlib import asynccontextmanager
from fastapi import FastAPI
from openai import OpenAI
from langfuse import Langfuse
import redis.asyncio as redis
from app.config import (
    REDIS_URL,
    OPENAI_API_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_HOST,
)


# --- Lifespan Context Manager (Handles ALL Client Initialization) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_conn = None
    app.state.openai_client = None
    app.state.langfuse = None
    print("Lifespan: Initializing resources...")
    try:
        redis_connection = redis.Redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True
        )
        await redis_connection.ping()
        app.state.redis_conn = redis_connection
        print("Lifespan: Redis client created and connected.")
    except Exception as e:
        print(f"Lifespan Startup Error: Could not connect to Redis: {e}")
        app.state.redis_conn = None
    if OPENAI_API_KEY:
        try:
            app.state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("Lifespan: OpenAI client created.")
        except Exception as e:
            print(f"Lifespan Startup Error: Could not initialize OpenAI client: {e}")
            app.state.openai_client = None
    else:
        print("Lifespan Warning: OPENAI_API_KEY not found.")

    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        try:
            app.state.langfuse = Langfuse(
                secret_key=LANGFUSE_SECRET_KEY,
                public_key=LANGFUSE_PUBLIC_KEY,
                host=LANGFUSE_HOST,
            )
            print("Lifespan: Langfuse client created.")
        except Exception as e:
            print(f"Lifespan Startup Error: Could not initialize Langfuse client: {e}")
            app.state.langfuse = None
    else:
        print("Lifespan Warning: LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY not found.")

    yield
    print("Lifespan: Shutting down resources...")
    if hasattr(app.state, "redis_conn") and app.state.redis_conn:
        await app.state.redis_conn.close()
        print("Lifespan: Redis connection closed.")
    if hasattr(app.state, "langfuse") and app.state.langfuse:
        app.state.langfuse.flush()
        print("Lifespan: Langfuse resources flushed.")
    print("Lifespan: Shutdown complete.")
