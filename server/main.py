from app.main import app

# This file serves as an entry point for the application
# All implementation details have been moved to the app package

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4, lifespan="on")
