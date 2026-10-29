from fastapi import FastAPI

app = FastAPI(
    title="PlacementDesk AI",
    description=(
        "An evidence=grounded AI assistant "
        "for placement-related information."
    ),
    version="0.1.0", 
)

@app.get("/")
def root():
    return {
        "project": "PlacementDesk AI",
        "status": "running",
        "version": "0.1.0",
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "The application is running smoothly."
    }