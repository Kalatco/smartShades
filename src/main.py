"""
Smart Shades Agent - Main Application Entry Point
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.smart_shades_agent import SmartShadesAgent
from models.api import (
    ShadeStatusResponse,
    RoomsResponse,
    ShadeControlCommand,
)


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent

    # Startup
    logger.info("Starting Smart Shades Agent...")
    agent = SmartShadesAgent()
    await agent.initialize()
    logger.info("Smart Shades Agent initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Smart Shades Agent...")
    if agent:
        await agent.shutdown()
    logger.info("Smart Shades Agent shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Smart Shades Agent API",
    description="""
    ## Smart Shades Control System
    
    LangGraph-based intelligent agent for smart shades control with natural language processing.
    
    ### Features:
    * **Natural Language Control**: Control shades using voice commands or text
    * **Multi-Room Support**: Manage shades across different rooms
    * **Solar Intelligence**: Automatic sun exposure detection and glare management
    * **Multi-Blind Operations**: Control multiple blinds with different positions
    * **Real-time Status**: Get current blind positions and room information
    
    ### Usage Examples:
    * "Open the blinds halfway"
    * "Close all blinds in the living room"
    * "Block the sun in the bedroom"
    * "Open the side window halfway, and front window fully"
    """,
    version="1.0.0",
    contact={
        "name": "SmartShadesAgent",
        "url": "https://github.com/kalatco/SmartShadesAgent",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc documentation
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint

    Returns the current status of the Smart Shades Agent system.
    """
    return {"status": "healthy", "agent": "running"}


@app.get("/rooms", response_model=RoomsResponse, tags=["Room Management"])
async def get_available_rooms():
    """
    Get list of available rooms and their blind configurations

    Returns all configured rooms with their associated blinds and metadata.
    """
    try:
        if not agent or not agent.config:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        rooms = {}
        for room_name, room_config in agent.config.rooms.items():
            rooms[room_name] = {
                "blind_count": len(room_config.blinds),
                "blinds": [
                    {"id": blind.id, "name": blind.name} for blind in room_config.blinds
                ],
            }

        return {"rooms": rooms}
    except Exception as e:
        logger.error(f"Error getting rooms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/rooms/{room}/control", response_model=ShadeStatusResponse, tags=["Shade Control"]
)
async def control_shades_post(
    room: str,
    request: ShadeControlCommand,
):
    """
    Control shades via POST request using natural language commands

    This endpoint accepts JSON payloads with shade control commands.

    **Request Body:**
    ```json
    {
        "command": "Open the blinds halfway"
    }
    ```

    **Command Examples:**
    * "Open the blinds halfway"
    * "Close all blinds"
    * "Block the sun"
    * "Open the side window halfway, and front window fully"
    * "Set blinds to 75 percent"
    * "Reduce glare"

    **Parameters:**
    * **room**: The room name (e.g., "guest_bedroom", "living_room", "master_bedroom")
    * **request**: JSON body containing the command

    **Response:** Returns a ShadeStatusResponse with operation results and affected blinds.
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        result = await agent.process_request(request.command, room, None)

        # Create a short, voice-friendly message for Apple Shortcuts
        affected_blinds = result.get("affected_blinds", [])
        position = result.get("position", 50)

        if len(affected_blinds) == 1:
            voice_message = f"{affected_blinds[0]} set to {position}%"
        elif len(affected_blinds) > 1:
            voice_message = f"{len(affected_blinds)} blinds adjusted"
        else:
            voice_message = f"Blinds set to {position}%"

        return ShadeStatusResponse(
            success=True,
            position=position,
            message=voice_message,
            room=result.get("room", room),
            affected_blinds=affected_blinds,
            timestamp=result.get("timestamp"),
        )
    except Exception as e:
        logger.error(f"Error processing shade control request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/rooms/{room}/status", response_model=ShadeStatusResponse, tags=["Shade Status"]
)
async def get_shade_status(room: str):
    """
    Get current shade status for a specific room

    Returns the current position and status of all blinds in the specified room.
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        status = await agent.get_current_status(room)
        return ShadeStatusResponse(
            success=True,
            position=status.get("position", 50),
            message=status.get("message", "Status retrieved successfully"),
            room=status.get("room", room),
            affected_blinds=status.get("affected_blinds", []),
            timestamp=status.get("timestamp"),
        )
    except Exception as e:
        logger.error(f"Error getting shade status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rooms/{room}/solar", tags=["Solar Intelligence"])
async def get_solar_info(room: str):
    """
    Get solar information and sun exposure analysis for a specific room

    Returns detailed solar information including:
    * Current sun position (azimuth, elevation)
    * Sunrise and sunset times
    * Per-window sun exposure analysis
    * Recommendations for glare management

    This endpoint is used by the intelligent shade control system to automatically
    manage blinds based on sun position and potential glare.
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        solar_info = agent._get_window_sun_exposure(room)
        return {"room": room, "solar_data": solar_info}
    except Exception as e:
        logger.error(f"Error getting solar info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def main():
    """Main application function"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Starting Smart Shades Agent API on {host}:{port}")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
