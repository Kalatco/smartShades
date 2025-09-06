"""
Root API endpoints for health checks and documentation
"""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint

    Returns a basic health status to verify the API is running.
    """
    return {"status": "healthy", "service": "Smart Shades Agent"}
