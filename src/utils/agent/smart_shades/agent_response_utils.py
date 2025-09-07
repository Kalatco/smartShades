"""
Response handling utilities for Smart Shades Agent
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from models.config import HubitatConfig
from models.agent import ExecutionResult
from utils.hubitat_utils import HubitatUtils

logger = logging.getLogger(__name__)


class AgentResponseUtils:
    """Utility class for handling and formatting agent responses"""

    @staticmethod
    def create_error_response(message: str, room: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "position": 0,
            "message": message,
            "room": room,
            "affected_blinds": [],
            "timestamp": datetime.now(),
        }

    @staticmethod
    def build_response_from_execution(
        execution_result: ExecutionResult, room: str
    ) -> Dict[str, Any]:
        """Build response from execution result"""
        try:
            affected_blinds = execution_result.executed_blinds
            message = f"{execution_result.reasoning}. Affected blinds: {', '.join(affected_blinds)}"

            return {
                "position": execution_result.position,
                "message": message,
                "room": room,
                "affected_blinds": affected_blinds,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.warning(f"Error building response: {e}")
            return AgentResponseUtils.create_error_response(
                "Position updated successfully", room
            )

    @staticmethod
    async def get_current_status(config: HubitatConfig, room: str) -> Dict[str, Any]:
        """Get current shade status for a room"""
        if room not in config.rooms:
            return {
                "position": 0,
                "message": f"Invalid room: {room}",
                "room": room,
                "affected_blinds": [],
                "timestamp": datetime.now(),
            }

        try:
            # Get actual current positions from Hubitat
            current_positions = await HubitatUtils.get_room_current_positions(
                config, room
            )
            blind_names = list(current_positions.keys())

            # Calculate average position for the room
            avg_position = (
                sum(current_positions.values()) // len(current_positions)
                if current_positions
                else 50
            )

            # Create status message with individual blind positions
            status_details = ", ".join(
                [f"{name}: {pos}%" for name, pos in current_positions.items()]
            )

            return {
                "position": avg_position,
                "message": f"Current status for {room}: {status_details}",
                "room": room,
                "affected_blinds": blind_names,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error getting current status for {room}: {e}")
            # Fallback to basic info
            blind_names = [blind.name for blind in config.rooms[room].blinds]
            return {
                "position": 50,
                "message": f"Status for {room} ({len(blind_names)} blinds) - Could not retrieve current positions",
                "room": room,
                "affected_blinds": blind_names,
                "timestamp": datetime.now(),
            }

    @staticmethod
    def create_standard_response(
        position: int,
        message: str,
        room: str,
        affected_blinds: List[str],
    ) -> Dict[str, Any]:
        """Create a standardized response with common fields"""
        return {
            "position": position,
            "message": message,
            "room": room,
            "affected_blinds": affected_blinds,
            "timestamp": datetime.now(),
        }

    @staticmethod
    def create_schedule_response(
        operation_type: str,
        description: str,
        room: str,
        schedule_id: str = None,
    ) -> Dict[str, Any]:
        """Create a response for schedule operations"""
        message = f"Schedule {operation_type}: {description}"
        if schedule_id:
            message += f" (ID: {schedule_id})"

        return {
            "position": 0,  # Schedules don't have immediate position changes
            "message": message,
            "room": room,
            "affected_blinds": [],
            "timestamp": datetime.now(),
            "schedule_id": schedule_id,
            "operation_type": operation_type,
        }
