"""
Smart Shades Agent implementation using LangChain
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI

from models.config import HubitatConfig
from models.agent import (
    ShadeAnalysis,
    ExecutionResult,
    BlindOperation,
)
from chains.house_wide_detection import HouseWideDetectionChain
from chains.shade_analysis import ShadeAnalysisChain
from utils.solar_utils import SolarUtils
from utils.hubitat_utils import HubitatUtils
from utils.blind_utils import BlindUtils

logger = logging.getLogger(__name__)


class SmartShadesAgent:
    """LangChain-based agent for intelligent shade control"""

    def __init__(self):
        self.llm = None
        self.config = None
        self.house_wide_chain = None
        self.shade_analysis_chain = None

    async def initialize(self):
        """Initialize the agent and LangChain components"""
        # Load environment variables from .env file
        load_dotenv()

        # Load configuration
        await self._load_config()

        # Override config with environment variables for Hubitat
        hubitat_access_token = os.getenv("HUBITAT_ACCESS_TOKEN")
        hubitat_api_url = os.getenv("HUBITAT_API_URL")

        if hubitat_access_token:
            self.config.accessToken = hubitat_access_token
        if hubitat_api_url:
            self.config.hubitatUrl = hubitat_api_url
        if not self.config.makerApiId:
            self.config.makerApiId = "1"  # Default setup

        # Initialize LLM
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([api_key, azure_endpoint, deployment_name, api_version]):
            raise ValueError(
                "Azure OpenAI environment variables are required: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION"
            )

        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            deployment_name=deployment_name,
            api_version=api_version,
            temperature=0,
        )

        # Initialize chains
        self.house_wide_chain = HouseWideDetectionChain(self.llm)
        self.shade_analysis_chain = ShadeAnalysisChain(self.llm)

        logger.info("Smart Shades Agent initialized successfully")

    async def _load_config(self):
        """Load blinds configuration from JSON file"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "blinds_config.json",
        )

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            self.config = HubitatConfig(**config_data)
            logger.info(f"Loaded configuration for {len(self.config.rooms)} rooms")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

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
        """Process a user request through the LangChain pipeline"""
        try:
            # Validate room
            if room not in self.config.rooms:
                return self._error_response(
                    f"Invalid room: {room}. Available rooms: {list(self.config.rooms.keys())}",
                    room,
                )

            # 1. Detect if the command is house-wide using LLM prompt
            is_house_wide = await self.house_wide_chain.ainvoke({"command": command})

            # 2. Analyze the request
            analysis = await self._analyze_request(command, room, is_house_wide)

            # 3. Execute the action
            execution_result = await self._execute_action(analysis, room)

            # 4. Build and return response
            return self._build_response_from_execution(execution_result, room)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return self._error_response(f"Error processing command: {e}", room)

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
