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
from utils.agent_response_utils import AgentResponseUtils
from utils.execution_utils import ExecutionUtils

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
        return await ExecutionUtils.analyze_request(
            self.shade_analysis_chain, self.config, command, room, is_house_wide
        )

    async def _execute_action(
        self, analysis: ShadeAnalysis, room: str
    ) -> ExecutionResult:
        """Execute the shade position change via Hubitat API"""
        return await ExecutionUtils.execute_action(self.config, analysis, room)

    async def process_request(
        self, command: str, room: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user request through the enhanced LangChain pipeline with scheduling"""
        try:
            # Validate room
            if not ExecutionUtils.validate_room(self.config, room):
                return AgentResponseUtils.create_error_response(
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
            return AgentResponseUtils.create_error_response(
                f"Error processing command: {e}", room
            )

    async def _process_current_execution(
        self, command: str, room: str
    ) -> Dict[str, Any]:
        """Process immediate execution commands"""
        return await ExecutionUtils.process_current_execution(
            self.shade_analysis_chain,
            self.house_wide_chain,
            self.config,
            command,
            room,
        )

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
                return AgentResponseUtils.create_error_response(
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
                return AgentResponseUtils.create_error_response(
                    result.get("error", "Unknown scheduling error"), room
                )

        except Exception as e:
            logger.error(f"Error in scheduled execution: {e}")
            return AgentResponseUtils.create_error_response(
                f"Error processing schedule: {e}", room
            )

    async def get_current_status(self, room: str) -> Dict[str, Any]:
        """Get current shade status for a room"""
        return await AgentResponseUtils.get_current_status(self.config, room)
