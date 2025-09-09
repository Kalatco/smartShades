"""
Room-related API endpoints for Smart Shades Agent
"""

import logging
from fastapi import APIRouter, HTTPException
from models.api import (
    ShadeControlCommand,
    ShadeStatusResponse,
    RoomsResponse,
)
from utils.solar import SolarUtils

logger = logging.getLogger(__name__)

router = APIRouter()

# This will be injected by main.py
agent = None


def set_agent(agent_instance):
    """Set the global agent instance"""
    global agent
    agent = agent_instance


@router.get("/rooms", response_model=RoomsResponse, tags=["Room Management"])
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


@router.post(
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

        # Handle error responses from V2 agent
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Extract data from V2 agent response format
        operation = result.get("operation", "unknown")

        if operation == "current_execution":
            # Current execution response format
            successful_blinds = result.get("successful_blinds", {})
            failed_blinds = result.get("failed_blinds", {})
            total_successful = result.get("total_successful", 0)

            # Calculate average position from successful blinds
            if successful_blinds:
                position = sum(successful_blinds.values()) // len(successful_blinds)
            else:
                position = 50  # Default fallback

            # Get blind names for affected_blinds list
            affected_blinds = list(successful_blinds.keys())

            # Create voice-friendly message
            if len(affected_blinds) == 1:
                blind_name = next(
                    (
                        blind.name
                        for room_config in agent.config.rooms.values()
                        for blind in room_config.blinds
                        if blind.id == affected_blinds[0]
                    ),
                    affected_blinds[0],
                )
                voice_message = f"{blind_name} set to {position}%"
            elif len(affected_blinds) > 1:
                voice_message = f"{len(affected_blinds)} blinds adjusted"
            else:
                voice_message = "No blinds were affected"

            # Add failure information if any
            if failed_blinds:
                voice_message += f" ({len(failed_blinds)} failed)"

        elif operation == "scheduled_execution":
            # Scheduled execution response format
            execution_result = result.get("execution_result", {})
            successful_blinds = execution_result.get("successful_blinds", {})
            total_successful = execution_result.get("total_successful", 0)

            # Calculate average position from successful blinds
            if successful_blinds:
                position = sum(successful_blinds.values()) // len(successful_blinds)
            else:
                position = 50  # Default fallback

            affected_blinds = list(successful_blinds.keys())
            voice_message = (
                f"Schedule created: {total_successful} blinds will be controlled"
            )

        else:
            # Fallback for unknown operation types
            position = 50
            affected_blinds = []
            voice_message = result.get("message", "Operation completed")

        return ShadeStatusResponse(
            success=True,
            position=position,
            message=voice_message,
            room=result.get("room", room),
            affected_blinds=affected_blinds,
            timestamp=result.get("timestamp"),
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing shade control request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
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

        # Handle error responses from V2 agent
        if "error" in status:
            raise HTTPException(status_code=400, detail=status["error"])

        # Extract data from V2 agent status response
        current_positions = status.get("current_positions", {})

        # Calculate average position if multiple blinds
        if current_positions:
            position = sum(current_positions.values()) // len(current_positions)
            affected_blinds = list(current_positions.keys())
        else:
            position = 0
            affected_blinds = []

        # Create status message
        if len(affected_blinds) == 1:
            blind_name = next(
                (
                    blind.name
                    for room_config in agent.config.rooms.values()
                    for blind in room_config.blinds
                    if blind.id == affected_blinds[0]
                ),
                affected_blinds[0],
            )
            message = f"{blind_name} at {position}%"
        elif len(affected_blinds) > 1:
            message = f"{len(affected_blinds)} blinds average: {position}%"
        else:
            message = "No blinds found in room"

        return ShadeStatusResponse(
            success=True,
            position=position,
            message=message,
            room=status.get("room", room),
            affected_blinds=affected_blinds,
            timestamp=status.get("timestamp"),
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting shade status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rooms/{room}/solar", tags=["Solar Intelligence"])
async def get_solar_info(room: str):
    """
    Get sunrise and sunset information for scheduling purposes

    Returns basic solar information for scheduling:
    * Sunrise and sunset times
    * Current time and timezone
    * Location coordinates

    Note: Solar position analysis has been removed. This endpoint
    now only provides sunrise/sunset data for automated scheduling.
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        # Only provide sunrise/sunset info for scheduling
        from utils.solar import SolarUtils

        solar_info = SolarUtils.get_solar_info(agent.config)

        return {
            "room": room,
            "solar_data": {
                "sunrise_sunset": solar_info,
                "note": "Only sunrise/sunset data available for scheduling",
            },
        }
    except Exception as e:
        logger.error(f"Error getting solar info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
