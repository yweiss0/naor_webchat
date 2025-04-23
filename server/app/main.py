from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.lifecycle import lifespan
from app.endpoints import chat, health
from app.config import origins

# Create FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "X-Request-ID"],
    max_age=600,
)

# Include routers
app.include_router(chat.router, prefix="/api")
app.include_router(health.router, prefix="/api")

# Optional: To Run Directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=4, lifespan="on")
