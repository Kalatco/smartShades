"""
Schedule-related API endpoints for Smart Shades Agent
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from models.api import (
    ScheduleRequest,
    ScheduleResponse,
    ScheduleInfo,
    ScheduleListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# This will be injected by main.py
agent = None


def set_agent(agent_instance):
    """Set the global agent instance"""
    global agent
    agent = agent_instance


@router.get(
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
                end_date=schedule_data.get("end_date"),
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


@router.post(
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


@router.delete(
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
