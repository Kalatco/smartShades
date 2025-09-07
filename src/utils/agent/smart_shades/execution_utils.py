"""
Execution management utilities for Smart Shades Agent
"""

import logging
from typing import Dict, Any, List

from models.config import HubitatConfig
from models.agent import ShadeAnalysis, ExecutionResult, BlindOperation
from utils.solar import SolarUtils
from utils.hubitat_utils import HubitatUtils
from utils.blind_utils import BlindUtils
from chains.shade_analysis import ShadeAnalysisChain

logger = logging.getLogger(__name__)


class ExecutionUtils:
    """Utility class for handling shade execution and analysis operations"""

    @staticmethod
    async def analyze_request(
        shade_analysis_chain: ShadeAnalysisChain,
        config: HubitatConfig,
        command: str,
        room: str,
        is_house_wide: bool = False,
    ) -> ShadeAnalysis:
        """Analyze the incoming request and determine target position and affected blinds"""

        # Get available blinds for the current room
        room_blinds = []
        if room in config.rooms:
            room_blinds = config.rooms[room].blinds

        # Get current positions of blinds in this room
        try:
            current_positions = await HubitatUtils.get_room_current_positions(
                config, room
            )
        except Exception as e:
            logger.warning(f"Could not get current positions: {e}")
            current_positions = {blind.name: 50 for blind in room_blinds}

        # Get solar information and window sun exposure
        try:
            window_sun_info = SolarUtils.get_window_sun_exposure(config, room)
        except Exception as e:
            logger.warning(f"Could not get solar info: {e}")
            window_sun_info = {"error": "Solar info unavailable"}

        # Prepare the command with house-wide hint if needed
        if is_house_wide:
            enhanced_command = f"[HOUSE-WIDE COMMAND] {command}"
        else:
            enhanced_command = command

        # Use the shade analysis chain
        analysis_input = {
            "command": enhanced_command,
            "room": room,
            "room_blinds": room_blinds,
            "current_positions": current_positions,
            "window_sun_info": window_sun_info,
            "house_orientation": getattr(config.houseInformation, "orientation", ""),
            "notes": getattr(config.houseInformation, "notes", ""),
        }

        try:
            analysis = await shade_analysis_chain.ainvoke(analysis_input)
            return analysis
        except Exception as e:
            logger.error(f"Error in shade analysis: {e}")
            # Simple fallback
            return ShadeAnalysis(
                scope="room",
                reasoning=f"Fallback parsing due to error: {e}",
            )

    @staticmethod
    async def execute_action(
        config: HubitatConfig, analysis: ShadeAnalysis, room: str
    ) -> ExecutionResult:
        """Execute the shade position change via Hubitat API"""
        if not room or room not in config.rooms:
            return ExecutionResult(
                executed_blinds=[],
                affected_rooms=[],
                total_blinds=0,
                position=0,
                scope="error",
                reasoning=f"Invalid room: {room}",
            )

        try:
            all_executed_blinds = []
            all_affected_rooms = []
            executed_position = None

            # Process operations (new format only)
            if analysis.operations:
                # Multi-operation format: execute each operation separately
                for operation in analysis.operations:
                    target_blinds, affected_rooms = (
                        BlindUtils.get_target_blinds_for_operation(
                            config, analysis.scope, operation.blind_filter, room
                        )
                    )

                    if target_blinds:
                        # Execute this operation
                        await HubitatUtils.control_blinds(
                            config, target_blinds, operation.position
                        )
                        all_executed_blinds.extend(
                            [blind.name for blind in target_blinds]
                        )
                        all_affected_rooms.extend(affected_rooms)
                        # Use the first operation's position as the main position
                        if executed_position is None:
                            executed_position = operation.position
            else:
                # No operations - this shouldn't happen with new format
                logger.warning("ShadeAnalysis received with no operations")
                return ExecutionResult(
                    executed_blinds=[],
                    affected_rooms=[],
                    total_blinds=0,
                    position=0,
                    scope=analysis.scope,
                    reasoning="No operations to execute",
                )

            if not all_executed_blinds:
                return ExecutionResult(
                    executed_blinds=[],
                    affected_rooms=[],
                    total_blinds=0,
                    position=executed_position or 0,  # Fallback to 0 if still None
                    scope=analysis.scope,
                    reasoning="No matching blinds found",
                )

            # Remove duplicates while preserving order
            unique_blinds = list(dict.fromkeys(all_executed_blinds))
            unique_rooms = list(dict.fromkeys(all_affected_rooms))

            result = ExecutionResult(
                executed_blinds=unique_blinds,
                affected_rooms=unique_rooms,
                total_blinds=len(unique_blinds),
                position=executed_position or 0,  # Fallback to 0 if still None
                scope=analysis.scope,
                reasoning=analysis.reasoning,
            )

            logger.info(
                f"Executed: Set {len(unique_blinds)} blinds to {executed_position}%"
            )
            return result

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return ExecutionResult(
                executed_blinds=[],
                affected_rooms=[],
                total_blinds=0,
                position=0,  # Default to 0 on error
                scope="error",
                reasoning=f"Error controlling blinds: {e}",
            )

    @staticmethod
    async def process_current_execution(
        shade_analysis_chain: ShadeAnalysisChain,
        house_wide_chain,
        config: HubitatConfig,
        command: str,
        room: str,
    ) -> Dict[str, Any]:
        """Process immediate execution commands using extracted utilities"""
        from utils.agent_response_utils import AgentResponseUtils

        try:
            # 1. Detect if the command is house-wide using LLM prompt
            is_house_wide = await house_wide_chain.ainvoke({"command": command})

            # 2. Analyze the request
            analysis = await ExecutionUtils.analyze_request(
                shade_analysis_chain, config, command, room, is_house_wide.is_house_wide
            )

            # 3. Execute the action
            execution_result = await ExecutionUtils.execute_action(
                config, analysis, room
            )

            # 4. Build and return response
            return AgentResponseUtils.build_response_from_execution(
                execution_result, room
            )

        except Exception as e:
            logger.error(f"Error in current execution: {e}")
            return AgentResponseUtils.create_error_response(
                f"Error executing command: {e}", room
            )

    @staticmethod
    def validate_room(config: HubitatConfig, room: str) -> bool:
        """Validate that a room exists in the configuration"""
        return bool(room) and room in config.rooms

    @staticmethod
    def get_room_blinds(config: HubitatConfig, room: str) -> List[Any]:
        """Get blinds for a specific room"""
        if not ExecutionUtils.validate_room(config, room):
            return []
        return config.rooms[room].blinds

    @staticmethod
    async def get_room_current_positions(
        config: HubitatConfig, room: str
    ) -> Dict[str, int]:
        """Get current positions for all blinds in a room with error handling"""
        try:
            return await HubitatUtils.get_room_current_positions(config, room)
        except Exception as e:
            logger.warning(f"Could not get current positions for {room}: {e}")
            # Fallback to default positions
            room_blinds = ExecutionUtils.get_room_blinds(config, room)
            return {blind.name: 50 for blind in room_blinds}
