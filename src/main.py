"""
Smart Shades Agent - Main Application Entry Point
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from utils.solar import SolarUtils

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.smart_shades_agent import SmartShadesAgent
from models.api import (
    ShadeStatusResponse,
    RoomsResponse,
    ShadeControlCommand,
    ScheduleRequest,
    ScheduleResponse,
    ScheduleInfo,
    ScheduleListResponse,
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
        "name": "smartShades",
        "url": "https://github.com/kalatco/smartShades",
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

        solar_info = SolarUtils.get_window_sun_exposure(agent.config, room)
        return {"room": room, "solar_data": solar_info}
    except Exception as e:
        logger.error(f"Error getting solar info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/schedules", response_model=ScheduleListResponse, tags=["Schedule Management"]
)
async def get_all_schedules():
    """
    Get all active schedules

    Returns a list of all currently active schedules including their details,
    next run times, and associated rooms.
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        # Get all schedules from the agent's scheduler
        schedules = agent.scheduler.get_all_schedules()
        schedule_info_list = []

        for schedule_id, schedule_data in schedules.items():
            schedule_info = ScheduleInfo(
                id=schedule_id,
                room=schedule_data.get("room"),
                command=schedule_data.get("command", ""),
                description=schedule_data.get("description", ""),
                trigger_type=schedule_data.get("trigger_type", "unknown"),
                next_run_time=schedule_data.get("next_run_time"),
                created_at=schedule_data.get("created_at", datetime.now()),
                is_active=schedule_data.get("is_active", True),
            )
            schedule_info_list.append(schedule_info)

        return ScheduleListResponse(
            schedules=schedule_info_list, total_count=len(schedule_info_list)
        )

    except Exception as e:
        logger.error(f"Error getting schedules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/rooms/{room}/schedules",
    response_model=ScheduleResponse,
    tags=["Schedule Management"],
)
async def create_schedule(room: str, request: ScheduleRequest):
    """
    Create a new schedule for a specific room

    **Request Body:**
    ```json
    {
        "command": "Close the blinds every weekday at 6 PM"
    }
    ```

    **Schedule Command Examples:**
    * "Close the blinds every weekday at 6 PM"
    * "Open blinds at sunrise"
    * "Block the sun after 2 PM on weekends"
    * "Close blinds for the next 3 days at 8 PM"
    * "Stop all scheduled blind operations"

    **Parameters:**
    * **room**: The room name for the scheduled action
    * **request**: JSON body containing the scheduling command

    **Response:** Returns a ScheduleResponse with the created schedule details.
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        # Process the scheduling request through the agent
        result = await agent.process_request(request.command, room, None)

        # Check if this was a scheduling operation
        if result.get("operation") and result.get("schedule_id"):
            # Schedule was created successfully
            schedule_info = ScheduleInfo(
                id=result["schedule_id"],
                room=room,
                command=request.command,
                description=result.get("message", ""),
                trigger_type="unknown",  # TODO: get from scheduler
                next_run_time=result.get("next_run"),
                created_at=datetime.now(),
                is_active=True,
            )

            return ScheduleResponse(
                success=True,
                message=result.get("message", "Schedule created successfully"),
                schedule=schedule_info,
            )
        elif result.get("operation"):
            # Schedule operation but no schedule created (e.g., delete operation)
            return ScheduleResponse(
                success=True,
                message=result.get(
                    "message", "Schedule operation completed successfully"
                ),
            )
        else:
            # Check if there's an error in the result
            if (
                "error" in result.get("message", "").lower()
                or result.get("position") == 0
            ):
                # This was likely an attempt at scheduling that failed
                return ScheduleResponse(
                    success=False,
                    message=result.get(
                        "message",
                        "Command was not recognized as a scheduling operation. Try commands like 'close blinds every day at 6 PM' or 'open blinds at sunrise'.",
                    ),
                )
            else:
                # This was a regular command, not a scheduling operation
                return ScheduleResponse(
                    success=False,
                    message="Command was not recognized as a scheduling operation. Try commands like 'close blinds every day at 6 PM' or 'open blinds at sunrise'.",
                )

    except Exception as e:
        logger.error(f"Error creating schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/schedules/{schedule_id}",
    response_model=ScheduleResponse,
    tags=["Schedule Management"],
)
async def delete_schedule(schedule_id: str):
    """
    Delete a specific schedule

    Removes the specified schedule from the system. The schedule will no longer
    execute and will be permanently deleted.

    **Parameters:**
    * **schedule_id**: The unique ID of the schedule to delete

    **Response:** Returns a ScheduleResponse confirming the deletion.
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        # Delete the schedule using the agent's scheduler
        success = agent.scheduler.delete_schedule(schedule_id)

        if success:
            return ScheduleResponse(
                success=True, message=f"Schedule {schedule_id} deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=404, detail=f"Schedule {schedule_id} not found"
            )

    except Exception as e:
        logger.error(f"Error deleting schedule: {e}")
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
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
