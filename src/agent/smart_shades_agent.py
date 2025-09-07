"""
Smart Shades Agent implementation using LangChain
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from models.config import HubitatConfig
from models.agent import (
    ShadeAnalysis,
    ExecutionResult,
    BlindOperation,
    ExecutionTiming,
    ScheduleOperation,
)
from chains.house_wide_detection import HouseWideDetectionChain
from chains.shade_analysis import ShadeAnalysisChain
from chains.execution_timing import ExecutionTimingChain
from chains.schedule_management import ScheduleManagementChain
from utils.solar import SolarUtils
from utils.hubitat_utils import HubitatUtils
from utils.blind_utils import BlindUtils
from utils.smart_scheduler import SmartScheduler
from utils.config_utils import ConfigManager

logger = logging.getLogger(__name__)


class SmartShadesAgent:
    """LangChain-based agent for intelligent shade control"""

    def __init__(self):
        self.llm = None
        self.config = None
        self.house_wide_chain = None
        self.shade_analysis_chain = None
        self.execution_timing_chain = None
        self.schedule_management_chain = None
        self.scheduler = None

    async def initialize(self):
        """Initialize the agent and LangChain components"""
        # Load environment variables
        ConfigManager.load_environment()

        # Validate environment
        if not ConfigManager.validate_environment():
            raise ValueError("Required environment variables are not set")

        # Load configuration
        self.config = await ConfigManager.load_blinds_config()

        # Override config with environment variables for Hubitat
        self.config = ConfigManager.override_hubitat_config(self.config)

        # Initialize LLM
        self.llm = ConfigManager.create_azure_llm()

        # Initialize chains
        self.house_wide_chain = HouseWideDetectionChain(self.llm)
        self.shade_analysis_chain = ShadeAnalysisChain(self.llm)
        self.execution_timing_chain = ExecutionTimingChain(self.llm)
        self.schedule_management_chain = ScheduleManagementChain(self.llm)

        # Initialize scheduler
        self.scheduler = SmartScheduler(agent_instance=self)
        self.scheduler.set_config(self.config)
        await self.scheduler.start()

        # Log configuration summary
        config_summary = ConfigManager.get_config_summary(self.config)
        logger.info(f"Smart Shades Agent initialized successfully: {config_summary}")

    async def shutdown(self):
        """Shutdown the agent and cleanup resources"""
        if self.scheduler:
            await self.scheduler.shutdown()
        logger.info("Smart Shades Agent shutdown completed")

    def get_schedules(self, room: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all schedules, optionally filtered by room"""
        if self.scheduler:
            return self.scheduler.get_schedules(room)
        return []

    async def _analyze_request(
        self, command: str, room: str, is_house_wide: bool = False
    ) -> ShadeAnalysis:
        """Analyze the incoming request and determine target position and affected blinds"""

        # Get available blinds for the current room
        room_blinds = []
        if room in self.config.rooms:
            room_blinds = self.config.rooms[room].blinds

        # Get current positions of blinds in this room
        try:
            current_positions = await HubitatUtils.get_room_current_positions(
                self.config, room
            )
        except Exception as e:
            logger.warning(f"Could not get current positions: {e}")
            current_positions = {blind.name: 50 for blind in room_blinds}

        # Get solar information and window sun exposure
        try:
            window_sun_info = SolarUtils.get_window_sun_exposure(self.config, room)
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
            "house_orientation": getattr(
                self.config.houseInformation, "orientation", ""
            ),
            "notes": getattr(self.config.houseInformation, "notes", ""),
        }

        try:
            analysis = await self.shade_analysis_chain.ainvoke(analysis_input)
            return analysis
        except Exception as e:
            logger.error(f"Error in shade analysis: {e}")
            # Simple fallback
            return ShadeAnalysis(
                position=50,
                scope="room",
                blind_filter=[],
                reasoning=f"Fallback parsing due to error: {e}",
            )

    async def _execute_action(
        self, analysis: ShadeAnalysis, room: str
    ) -> ExecutionResult:
        """Execute the shade position change via Hubitat API"""
        if not room or room not in self.config.rooms:
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
                            self.config, analysis.scope, operation.blind_filter, room
                        )
                    )

                    if target_blinds:
                        # Execute this operation
                        await HubitatUtils.control_blinds(
                            self.config, target_blinds, operation.position
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

    async def process_request(
        self, command: str, room: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user request through the enhanced LangChain pipeline with scheduling"""
        try:
            # Validate room
            if room not in self.config.rooms:
                return self._error_response(
                    f"Invalid room: {room}. Available rooms: {list(self.config.rooms.keys())}",
                    room,
                )

            # 1. Detect execution timing (current vs scheduled)
            timing = await self.execution_timing_chain.ainvoke({"command": command})

            if timing.execution_type == "current":
                # Current execution flow
                return await self._process_current_execution(command, room)
            else:
                # Scheduled execution flow
                return await self._process_scheduled_execution(command, room)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return self._error_response(f"Error processing command: {e}", room)

    async def _process_current_execution(
        self, command: str, room: str
    ) -> Dict[str, Any]:
        """Process immediate execution commands"""
        try:
            # 1. Detect if the command is house-wide using LLM prompt
            is_house_wide = await self.house_wide_chain.ainvoke({"command": command})

            # 2. Analyze the request
            analysis = await self._analyze_request(
                command, room, is_house_wide.is_house_wide
            )

            # 3. Execute the action
            execution_result = await self._execute_action(analysis, room)

            # 4. Build and return response
            return self._build_response_from_execution(execution_result, room)

        except Exception as e:
            logger.error(f"Error in current execution: {e}")
            return self._error_response(f"Error executing command: {e}", room)

    async def _process_scheduled_execution(
        self, command: str, room: str
    ) -> Dict[str, Any]:
        """Process scheduled execution commands"""
        try:
            # 1. Get existing schedules for context
            existing_schedules = self.scheduler.get_schedules(room)

            # 2. Analyze the schedule request
            schedule_op = await self.schedule_management_chain.ainvoke(
                {
                    "command": command,
                    "room": room,
                    "existing_schedules": existing_schedules,
                }
            )

            # 3. Execute the schedule operation
            if schedule_op.action_type == "create":
                result = await self.scheduler.create_schedule(schedule_op, room)
            elif schedule_op.action_type == "modify":
                result = await self.scheduler.modify_schedule(schedule_op, room)
            elif schedule_op.action_type == "delete":
                result = await self.scheduler.delete_schedule(schedule_op, room)
            else:
                return self._error_response(
                    f"Unknown schedule action: {schedule_op.action_type}", room
                )

            # 4. Build response
            if result.get("success"):
                return {
                    "message": f"Schedule {schedule_op.action_type}d successfully: {schedule_op.schedule_description}",
                    "room": room,
                    "schedule_id": result.get("job_id"),
                    "next_run": result.get("next_run"),
                    "operation": schedule_op.action_type,
                    "timestamp": datetime.now(),
                }
            else:
                return self._error_response(
                    result.get("error", "Unknown scheduling error"), room
                )

        except Exception as e:
            logger.error(f"Error in scheduled execution: {e}")
            return self._error_response(f"Error processing schedule: {e}", room)

    def _error_response(self, message: str, room: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "position": 0,
            "message": message,
            "room": room,
            "affected_blinds": [],
            "timestamp": datetime.now(),
        }

    def _build_response_from_execution(
        self, execution_result: ExecutionResult, room: str
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
            return self._error_response("Position updated successfully", room)

    async def get_current_status(self, room: str) -> Dict[str, Any]:
        """Get current shade status for a room"""
        if room not in self.config.rooms:
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
                self.config, room
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
            blind_names = [blind.name for blind in self.config.rooms[room].blinds]
            return {
                "position": 50,
                "message": f"Status for {room} ({len(blind_names)} blinds) - Could not retrieve current positions",
                "room": room,
                "affected_blinds": blind_names,
                "timestamp": datetime.now(),
            }

    async def shutdown(self):
        """Shutdown the agent"""
        logger.info("Smart Shades Agent shutdown complete")
