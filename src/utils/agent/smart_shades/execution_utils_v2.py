"""
V2 Execution management utilities for Smart Shades Agent
Direct blind execution with simplified input structure
"""

import logging
from typing import Dict

from models.config import HubitatConfig
from models.agent import BlindExecutionRequest, BlindExecutionResult
from utils.hubitat_utils import HubitatUtils

logger = logging.getLogger(__name__)


class ExecutionUtilsV2:
    """V2 Utility class for direct blind execution operations"""

    @staticmethod
    async def execute_blinds(
        config: HubitatConfig, execution_request: BlindExecutionRequest
    ) -> BlindExecutionResult:
        """
        Execute blind positions based on the V2 execution request structure

        Args:
            config: HubitatConfig object for API access
            execution_request: BlindExecutionRequest with rooms and blind positions

        Returns:
            BlindExecutionResult with execution summary and results
        """
        successful_blinds = {}
        failed_blinds = {}
        total_attempted = 0

        logger.info(
            f"Starting V2 blind execution for {len(execution_request.rooms)} rooms"
        )

        # Process each room
        for room_name, room_data in execution_request.rooms.items():
            logger.info(
                f"Processing room: {room_name} with {len(room_data.blinds)} blinds"
            )

            # Process each blind in the room
            for blind_id, position in room_data.blinds.items():
                total_attempted += 1

                # Validate position
                if not (0 <= position <= 100):
                    error_msg = f"Invalid position {position} for blind {blind_id}. Must be 0-100."
                    logger.error(error_msg)
                    failed_blinds[blind_id] = error_msg
                    continue

                # Execute the blind control
                try:
                    success = await HubitatUtils.control_blind_v2(
                        config, blind_id, position
                    )

                    if success:
                        successful_blinds[blind_id] = position
                        logger.info(
                            f"Successfully controlled blind {blind_id} to {position}%"
                        )
                    else:
                        failed_blinds[blind_id] = "API call failed"
                        logger.error(f"Failed to control blind {blind_id}")

                except Exception as e:
                    error_msg = f"Exception occurred: {str(e)}"
                    failed_blinds[blind_id] = error_msg
                    logger.error(f"Error controlling blind {blind_id}: {e}")

        # Build execution summary
        total_successful = len(successful_blinds)
        execution_summary = (
            f"Executed {total_successful}/{total_attempted} blinds successfully"
        )

        if failed_blinds:
            execution_summary += f". {len(failed_blinds)} failed."

        result = BlindExecutionResult(
            successful_blinds=successful_blinds,
            failed_blinds=failed_blinds,
            total_attempted=total_attempted,
            total_successful=total_successful,
            execution_summary=execution_summary,
        )

        logger.info(f"V2 Execution completed: {execution_summary}")
        return result

    @staticmethod
    async def get_room_current_positions(
        config: HubitatConfig, room: str
    ) -> Dict[str, int]:
        """
        Get current positions of all blinds in a room

        Args:
            config: HubitatConfig object for API access
            room: Room name to get positions for

        Returns:
            Dictionary mapping blind IDs to their current positions
        """
        logger.info(f"Getting current positions for room: {room}")

        if room not in config.rooms:
            logger.warning(f"Room '{room}' not found in configuration")
            return {}

        positions = {}
        for blind in config.rooms[room].blinds:
            try:
                position = await HubitatUtils.get_blind_current_position(
                    config, blind.id
                )
                positions[blind.id] = position
                logger.info(f"Blind {blind.id} ({blind.name}) is at {position}%")
            except Exception as e:
                logger.error(f"Error getting position for blind {blind.id}: {e}")
                positions[blind.id] = 50  # Default fallback

        return positions
