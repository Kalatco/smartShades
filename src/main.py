"""
Smart Shades Agent - Main Application Entry Point
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.smart_shades_agent import SmartShadesAgent
from models.requests import ShadeControlRequest, ShadeStatusResponse

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
    title="Smart Shades Agent",
    description="LangGraph-based intelligent agent for smart shades control",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "running"}


@app.post("/control", response_model=ShadeStatusResponse)
async def control_shades(request: ShadeControlRequest):
    """Control shades based on natural language input"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        result = await agent.process_request(
            request.command, request.room, request.context
        )
        return ShadeStatusResponse(
            success=True,
            position=result.get("position", 50),
            message=result.get("message", "Command processed successfully"),
            room=result.get("room", request.room),
            affected_blinds=result.get("affected_blinds", []),
            timestamp=result.get("timestamp"),
        )
    except Exception as e:
        logger.error(f"Error processing shade control request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/control", response_model=ShadeStatusResponse)
async def control_shades_get(command: str, room: str):
    """Control shades via GET request for Apple Shortcuts - URL parameters: command and room"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        result = await agent.process_request(command, room, None)
        return ShadeStatusResponse(
            success=True,
            position=result.get("position", 50),
            message=result.get("message", "Command processed successfully"),
            room=result.get("room", room),
            affected_blinds=result.get("affected_blinds", []),
            timestamp=result.get("timestamp"),
        )
    except Exception as e:
        logger.error(f"Error processing shade control request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{room}", response_model=ShadeStatusResponse)
async def get_shade_status(room: str):
    """Get current shade status for a specific room"""
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


@app.get("/solar/{room}")
async def get_solar_info(room: str):
    """Get solar information and sun exposure for a specific room"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        solar_info = agent._get_window_sun_exposure(room)
        return {"room": room, "solar_data": solar_info}
    except Exception as e:
        logger.error(f"Error getting solar info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rooms")
async def get_available_rooms():
    """Get list of available rooms"""
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
