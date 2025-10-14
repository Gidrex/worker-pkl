"""FastAPI backend for vehicle tracking system."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import detection, training

app = FastAPI(
    title="Vehicle Tracking API",
    description="API for training and detecting parked/moving vehicles",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(detection.router, prefix="/api/detection", tags=["detection"])


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Vehicle Tracking API",
        "version": "1.0.0",
        "endpoints": {
            "training": "/api/training",
            "detection": "/api/detection",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
