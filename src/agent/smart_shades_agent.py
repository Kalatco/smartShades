"""
Smart Shades Agent implementation using LangGraph
"""

import asyncio
import logging
import os
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import pytz
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import httpx
from astral import LocationInfo
from astral.sun import sun, azimuth, elevation

# Use individual functions instead of combined sun() function
from astral.sun import sunrise, sunset

from models.requests import (
    AgentState,
    HubitatConfig,
    ShadeAnalysis,
    ExecutionResult,
    BlindOperation,
)

logger = logging.getLogger(__name__)


class SmartShadesAgent:
    """LangGraph-based agent for intelligent shade control"""

    def __init__(self):
        self.llm = None
        self.graph = None
        self.config = None

    async def initialize(self):
        """Initialize the agent and LangGraph"""
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
            self.config.makerApiId = "21"  # Default from your setup

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

        # Build the LangGraph
        self._build_graph()

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

    def _build_graph(self):
        """Build the LangGraph for shade control decision making"""

        def analyze_request(state: AgentState) -> AgentState:
            """Analyze the incoming request and determine target position and affected blinds"""
            messages = state.messages
            if not messages:
                return state

            user_message = messages[-1] if messages else ""

            # Get available blinds for the current room
            room_blinds = []
            if state.room in self.config.rooms:
                room_blinds = self.config.rooms[state.room].blinds

            # Get current positions of blinds in this room
            try:
                # Run async function in sync context (LangGraph limitation)
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                current_positions = loop.run_until_complete(
                    self._get_room_current_positions(state.room)
                )
            except Exception as e:
                logger.warning(f"Could not get current positions: {e}")
                current_positions = {blind.name: 50 for blind in room_blinds}

            # Get solar information and window sun exposure
            try:
                window_sun_info = self._get_window_sun_exposure(state.room)
            except Exception as e:
                logger.warning(f"Could not get solar info: {e}")
                window_sun_info = {"error": "Solar info unavailable"}

            # Create system prompt with current positions and solar info
            current_status = ", ".join(
                [f"{name}: {pos}%" for name, pos in current_positions.items()]
            )

            # Build solar context
            solar_context = ""
            if "error" not in window_sun_info:
                if "message" in window_sun_info:
                    solar_context = f"\nSUN STATUS: {window_sun_info['message']}"
                else:
                    sunny_windows = [
                        name
                        for name, info in window_sun_info.items()
                        if isinstance(info, dict) and info.get("is_sunny", False)
                    ]
                    if sunny_windows:
                        # Sort by intensity to identify the brightest
                        intensity_order = {"high": 3, "medium": 2, "low": 1}
                        sorted_windows = sorted(
                            sunny_windows,
                            key=lambda x: intensity_order.get(
                                window_sun_info[x]["sun_intensity"], 0
                            ),
                            reverse=True,
                        )

                        intensities = [
                            f"{name} ({window_sun_info[name]['sun_intensity']} intensity)"
                            for name in sorted_windows
                        ]
                        solar_context = f"\nSUN STATUS: Direct sunlight on: {', '.join(intensities)}"

                        # Highlight the brightest window
                        if len(sorted_windows) > 1:
                            brightest = sorted_windows[0]
                            brightest_intensity = window_sun_info[brightest][
                                "sun_intensity"
                            ]
                            solar_context += f"\nBRIGHTEST: {brightest} has {brightest_intensity} intensity (main glare source)"

                        if "solar_info" in window_sun_info:
                            solar = window_sun_info["solar_info"]
                            solar_context += f"\nSun direction: {solar['direction']}, elevation: {solar['elevation']:.1f}°"
                    else:
                        solar_context = f"\nSUN STATUS: No direct sunlight on any windows in this room"

            # Create system prompt with current positions and house layout
            current_status = ", ".join(
                [f"{name}: {pos}%" for name, pos in current_positions.items()]
            )

            # Add house orientation context
            house_info = ""
            if (
                hasattr(self.config, "house_orientation")
                and self.config.house_orientation
            ):
                house_info = (
                    f"\nHOUSE LAYOUT: {self.config.house_orientation} orientation"
                )
                if hasattr(self.config, "notes") and self.config.notes:
                    house_info += f". {self.config.notes}"

            system_prompt = f"""Analyze this shade control request for room: {state.room}

Available blinds in THIS room: {[f"{blind.name} ({getattr(blind, 'orientation', 'unknown')} facing)" for blind in room_blinds]}
CURRENT POSITIONS: {current_status}{solar_context}{house_info}

For commands that mention MULTIPLE specific blinds with DIFFERENT positions, use the 'operations' array.
For commands that set the SAME position for multiple/all blinds, use the legacy single operation format.

MULTIPLE OPERATIONS FORMAT (use when different positions needed):
- "open the side window halfway, and front window fully" → 
  operations: [
    {{blind_filter: ["side"], position: 50, reasoning: "side window to halfway"}},
    {{blind_filter: ["front"], position: 100, reasoning: "front window fully open"}}
  ]
  scope: "specific"

SINGLE OPERATION FORMAT (use when same position for all targets):
- "open all windows" → position: 100, scope: "room", blind_filter: []
- "close the front window" → position: 0, scope: "specific", blind_filter: ["front"]

Position guidelines:
- "open/up/fully"=100, "close/down"=0, "half/halfway"=50
- "a little more/bit more" = current + 10-15%
- "much more" = current + 25-30%

Scope rules:
- "house": ONLY if explicitly mentions "house", "all rooms", "entire house"
- "room": Default for "all windows", "all shades" (current room only)  
- "specific": When naming specific blinds OR multiple different operations

Examples:
- "Open all windows" → position: 100, scope: "room", blind_filter: []
- "Open the front window" → position: 100, scope: "specific", blind_filter: ["front"]
- "Open side halfway, front fully" → operations: [{{blind_filter: ["side"], position: 50}}, {{blind_filter: ["front"], position: 100}}], scope: "specific"

Provide structured response. Use operations array ONLY when different positions are needed for different blinds."""

            try:
                # Use structured output with Pydantic model
                structured_llm = self.llm.with_structured_output(ShadeAnalysis)

                analysis: ShadeAnalysis = structured_llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"User request: {user_message}"),
                    ]
                )

                # Set the target position and store analysis as dict
                # Handle both single and multi-operation formats
                if analysis.operations:
                    # Multi-operation format - use first operation's position as primary
                    state.target_position = analysis.operations[0].position
                else:
                    # Single operation format
                    state.target_position = analysis.position or 50

                state.reasoning = analysis.model_dump()

            except Exception as e:
                logger.error(f"Error in analyze_request: {e}")
                # Simple fallback
                fallback_analysis = ShadeAnalysis(
                    position=50,  # Default to 50%
                    scope="room",
                    blind_filter=[],
                    reasoning=f"Fallback parsing due to error: {e}",
                )
                state.target_position = fallback_analysis.position
                state.reasoning = fallback_analysis.model_dump()

            return state

        def execute_action(state: AgentState) -> AgentState:
            """Execute the shade position change via Hubitat API"""
            if not state.room or state.room not in self.config.rooms:
                state.reasoning = ExecutionResult(
                    executed_blinds=[],
                    affected_rooms=[],
                    total_blinds=0,
                    position=0,
                    scope="error",
                    reasoning=f"Invalid room: {state.room}",
                ).model_dump()
                return state

            try:
                # Validate that we have reasoning data from analyze_request
                if not state.reasoning:
                    state.reasoning = ExecutionResult(
                        executed_blinds=[],
                        affected_rooms=[],
                        total_blinds=0,
                        position=0,
                        scope="error",
                        reasoning="No analysis data available - analyze_request may have failed",
                    ).model_dump()
                    return state

                analysis = ShadeAnalysis.model_validate(state.reasoning)
                all_executed_blinds = []
                all_affected_rooms = []
                total_operations = 0

                # Handle multiple operations if present
                if analysis.operations:
                    # Multi-operation format: execute each operation separately
                    for operation in analysis.operations:
                        # Create temporary analysis for this operation
                        temp_analysis = ShadeAnalysis(
                            position=operation.position,
                            scope=analysis.scope,
                            blind_filter=operation.blind_filter,
                            reasoning=operation.reasoning,
                        )

                        target_blinds, affected_rooms = self._get_target_blinds(
                            temp_analysis, state.room
                        )

                        if target_blinds:
                            # Execute this operation
                            self._execute_blind_control_async(
                                target_blinds, operation.position
                            )
                            all_executed_blinds.extend(
                                [blind.name for blind in target_blinds]
                            )
                            all_affected_rooms.extend(affected_rooms)
                            total_operations += len(target_blinds)
                else:
                    # Single operation format: use legacy logic
                    target_blinds, affected_rooms = self._get_target_blinds(
                        analysis, state.room
                    )

                    if target_blinds:
                        self._execute_blind_control_async(
                            target_blinds, state.target_position
                        )
                        all_executed_blinds = [blind.name for blind in target_blinds]
                        all_affected_rooms = affected_rooms
                        total_operations = len(target_blinds)

                if not all_executed_blinds:
                    state.reasoning = ExecutionResult(
                        executed_blinds=[],
                        affected_rooms=[],
                        total_blinds=0,
                        position=state.target_position,
                        scope=analysis.scope,
                        reasoning=f"No matching blinds found",
                    ).model_dump()
                    return state

                # Remove duplicates while preserving order
                unique_blinds = list(dict.fromkeys(all_executed_blinds))
                unique_rooms = list(dict.fromkeys(all_affected_rooms))

                # Create success result
                state.reasoning = ExecutionResult(
                    executed_blinds=unique_blinds,
                    affected_rooms=unique_rooms,
                    total_blinds=len(unique_blinds),
                    position=state.target_position,
                    scope=analysis.scope,
                    reasoning=analysis.reasoning,
                ).model_dump()

                logger.info(
                    f"Executed: Set {len(target_blinds)} blinds to {state.target_position}%"
                )

            except Exception as e:
                logger.error(f"Error executing action: {e}")
                state.reasoning = ExecutionResult(
                    executed_blinds=[],
                    affected_rooms=[],
                    total_blinds=0,
                    position=state.target_position,
                    scope="error",
                    reasoning=f"Error controlling blinds: {e}",
                ).model_dump()

            return state

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze", analyze_request)
        workflow.add_node("execute", execute_action)

        # Add edges
        workflow.add_edge("analyze", "execute")
        workflow.add_edge("execute", END)

        # Set entry point
        workflow.set_entry_point("analyze")

        # Compile the graph
        self.graph = workflow.compile()

    def _get_target_blinds(self, analysis: ShadeAnalysis, room: str):
        """Get target blinds based on scope and filters"""
        target_blinds = []
        affected_rooms = []

        if analysis.scope == "house":
            # All rooms
            for room_name, room_config in self.config.rooms.items():
                if analysis.blind_filter:
                    filtered_blinds = self._filter_blinds(
                        room_config.blinds, analysis.blind_filter
                    )
                    if filtered_blinds:
                        target_blinds.extend(filtered_blinds)
                        affected_rooms.append(room_name)
                else:
                    target_blinds.extend(room_config.blinds)
                    affected_rooms.append(room_name)
        else:
            # Current room only (both "room" and "specific" scope)
            room_blinds = self.config.rooms[room].blinds
            if analysis.blind_filter:
                filtered_blinds = self._filter_blinds(
                    room_blinds, analysis.blind_filter
                )
                if filtered_blinds:
                    target_blinds = filtered_blinds
                    affected_rooms = [room]
            else:
                target_blinds = room_blinds
                affected_rooms = [room]

        return target_blinds, affected_rooms

    def _execute_blind_control_async(self, target_blinds, position: int):
        """Execute blind control in background thread"""

        def execute_control():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._control_blinds(target_blinds, position))
                loop.close()
            except Exception as e:
                logger.error(f"Error in background control execution: {e}")

        thread = threading.Thread(target=execute_control)
        thread.daemon = True
        thread.start()

    def _filter_blinds(self, blinds, filter_keywords):
        """Filter blinds based on keywords matching blind names"""
        if not filter_keywords:
            return blinds

        filtered = []
        for blind in blinds:
            blind_name_lower = blind.name.lower()
            if any(keyword.lower() in blind_name_lower for keyword in filter_keywords):
                filtered.append(blind)
        return filtered

    async def _control_blinds(self, blinds, position: int):
        """Send HTTP requests to control individual blinds"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            for blind in blinds:
                url = f"{self.config.hubitatUrl}/apps/api/{self.config.makerApiId}/devices/{blind.id}/setPosition/{position}?access_token={self.config.accessToken}"

                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        logger.info(f"Successfully set {blind.name} to {position}%")
                    else:
                        logger.error(
                            f"Failed to control {blind.name}: HTTP {response.status_code} - {response.text}"
                        )
                except Exception as e:
                    logger.error(f"Error controlling {blind.name}: {e}")

    async def _get_blind_current_position(self, blind_id: str) -> int:
        """Get current position of a specific blind from Hubitat"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{self.config.hubitatUrl}/apps/api/{self.config.makerApiId}/devices/{blind_id}?access_token={self.config.accessToken}"
                logger.info(f"Getting blind {blind_id} position from: {url}")
                response = await client.get(url)

                if response.status_code == 200:
                    device_data = response.json()
                    # Look for position attribute in the device attributes
                    for attr in device_data.get("attributes", []):
                        if attr.get("name") == "position":
                            return int(attr.get("currentValue", 50))
                    # Fallback to looking for 'level' attribute
                    for attr in device_data.get("attributes", []):
                        if attr.get("name") == "level":
                            return int(attr.get("currentValue", 50))
                    return 50  # Default if no position found
                else:
                    logger.warning(
                        f"Failed to get device {blind_id} status: HTTP {response.status_code}"
                    )
                    return 50
        except Exception as e:
            logger.error(f"Error getting blind {blind_id} position: {e}")
            return 50

    async def _get_room_current_positions(self, room: str) -> Dict[str, int]:
        """Get current positions of all blinds in a room"""
        if room not in self.config.rooms:
            return {}

        positions = {}
        for blind in self.config.rooms[room].blinds:
            positions[blind.name] = await self._get_blind_current_position(blind.id)

        return positions

    def _get_solar_info(self) -> Dict[str, Any]:
        """Get current solar position and sun-related information"""
        try:
            if not self.config.latitude or not self.config.longitude:
                return {"error": "Location coordinates not configured"}

            # Create location info with timezone from config
            location = LocationInfo(
                "Home",
                "Region",
                self.config.timezone or "UTC",
                self.config.latitude,
                self.config.longitude,
            )

            # Get current time in the configured timezone
            if self.config.timezone:
                pacific_tz = pytz.timezone(self.config.timezone)
                now = datetime.now(pacific_tz)
            else:
                now = datetime.now(timezone.utc)

            # Calculate sun position
            sun_azimuth = azimuth(
                location.observer, now
            )  # 0° = North, 90° = East, 180° = South, 270° = West
            sun_elevation_angle = elevation(location.observer, now)  # Above horizon

            # Get sun times for today - use LOCAL date, not UTC date
            if self.config.timezone:
                pacific_tz = pytz.timezone(self.config.timezone)
                local_now = datetime.now(pacific_tz)
                today_date = local_now.date()  # Use local date
            else:
                utc_now = datetime.now(pytz.UTC)
                today_date = utc_now.date()

            logger.info(f"Using local date for calculations: {today_date}")

            try:
                # Try calculating for today with explicit UTC date
                sunrise_utc = sunrise(location.observer, date=today_date)
                sunset_utc = sunset(location.observer, date=today_date)

                # Check if sunset is in the morning (wrong!) - if so, try tomorrow's date
                if sunset_utc.time().hour < 12:  # If sunset is in AM, it's wrong
                    logger.warning(
                        f"Sunset appears to be in AM ({sunset_utc.time()}), trying next day..."
                    )
                    tomorrow_date = today_date + timedelta(days=1)
                    sunset_utc = sunset(location.observer, date=tomorrow_date)
                    logger.info(f"  Corrected sunset UTC: {sunset_utc}")

                sun_times = {"sunrise": sunrise_utc, "sunset": sunset_utc}

            except Exception as e:
                logger.error(f"Error with individual sun calculations: {e}")
                # Fallback to original method
                sun_times = sun(location.observer, date=today_date)

            # Convert sunrise/sunset to the same timezone as 'now'
            if self.config.timezone:
                pacific_tz = pytz.timezone(self.config.timezone)
                sunrise_local = (
                    sun_times["sunrise"].replace(tzinfo=pytz.UTC).astimezone(pacific_tz)
                )
                sunset_local = (
                    sun_times["sunset"].replace(tzinfo=pytz.UTC).astimezone(pacific_tz)
                )

                logger.info(f"Converted sunrise: {sunrise_local}")
                logger.info(f"Converted sunset: {sunset_local}")
            else:
                sunrise_local = sun_times["sunrise"]
                sunset_local = sun_times["sunset"]

            # Determine if sun is up (comparing times in the same timezone)
            is_sun_up = sunrise_local <= now <= sunset_local

            # Convert azimuth to cardinal direction
            def azimuth_to_direction(azimuth_deg):
                directions = [
                    "north",
                    "northeast",
                    "east",
                    "southeast",
                    "south",
                    "southwest",
                    "west",
                    "northwest",
                ]
                idx = int((azimuth_deg + 22.5) // 45) % 8
                return directions[idx]

            sun_direction = azimuth_to_direction(sun_azimuth)

            return {
                "azimuth": sun_azimuth,
                "elevation": sun_elevation_angle,
                "direction": sun_direction,
                "is_up": is_sun_up,
                "sunrise": sunrise_local.strftime("%H:%M %Z"),
                "sunset": sunset_local.strftime("%H:%M %Z"),
                "current_time": now.strftime("%H:%M %Z"),
                "timezone": self.config.timezone or "UTC",
                "debug": f"Now: {now}, Sunrise: {sunrise_local}, Sunset: {sunset_local}, Sun up: {is_sun_up}",
            }

        except Exception as e:
            logger.error(f"Error calculating solar info: {e}")
            return {"error": f"Solar calculation failed: {e}"}

    def _get_window_sun_exposure(self, room: str) -> Dict[str, Dict[str, Any]]:
        """Determine which windows are currently exposed to direct sunlight"""
        solar_info = self._get_solar_info()

        if "error" in solar_info:
            return {"error": solar_info["error"]}

        if not solar_info.get("is_up", False):
            return {"message": "Sun is down - no direct sunlight"}

        window_exposure = {}

        # Check each blind in the room
        if room in self.config.rooms:
            for blind in self.config.rooms[room].blinds:
                orientation = getattr(blind, "orientation", "south")

                # Calculate if this window orientation is getting direct sun
                is_sunny = self._is_window_sunny(
                    orientation, solar_info["azimuth"], solar_info["elevation"]
                )

                window_exposure[blind.name] = {
                    "orientation": orientation,
                    "is_sunny": is_sunny,
                    "sun_intensity": self._calculate_sun_intensity(
                        orientation, solar_info["azimuth"], solar_info["elevation"]
                    ),
                }

        window_exposure["solar_info"] = solar_info
        return window_exposure

    def _is_window_sunny(
        self, window_orientation: str, sun_azimuth: float, sun_elevation: float
    ) -> bool:
        """Determine if a window with given orientation is getting direct sunlight"""
        if sun_elevation < 0:  # Sun is below horizon
            return False

        # Map orientations to azimuth ranges
        orientation_ranges = {
            "north": (315, 45),  # 315° to 45° (through 0°)
            "northeast": (0, 90),  # 0° to 90°
            "east": (45, 135),  # 45° to 135°
            "southeast": (90, 180),  # 90° to 180°
            "south": (135, 225),  # 135° to 225°
            "southwest": (180, 270),  # 180° to 270°
            "west": (225, 315),  # 225° to 315°
            "northwest": (270, 360),  # 270° to 360°
        }

        if window_orientation.lower() not in orientation_ranges:
            return False

        start, end = orientation_ranges[window_orientation.lower()]

        # Handle wrapping around 0° for north
        if window_orientation.lower() == "north":
            return sun_azimuth >= start or sun_azimuth <= end
        else:
            return start <= sun_azimuth <= end

    def _calculate_sun_intensity(
        self, window_orientation: str, sun_azimuth: float, sun_elevation: float
    ) -> str:
        """Calculate relative sun intensity for a window"""
        if not self._is_window_sunny(window_orientation, sun_azimuth, sun_elevation):
            return "none"

        # Higher elevation = more intense
        if sun_elevation > 60:
            return "high"
        elif sun_elevation > 30:
            return "medium"
        else:
            return "low"

    async def process_request(
        self, command: str, room: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user request through the LangGraph"""
        try:
            # Validate room
            if room not in self.config.rooms:
                return self._error_response(
                    f"Invalid room: {room}. Available rooms: {list(self.config.rooms.keys())}",
                    room,
                )

            # Handle house-wide commands - use first room as context
            if "house" in command.lower() or "all rooms" in command.lower():
                if room not in self.config.rooms:
                    room = list(self.config.rooms.keys())[0]

            # Prepare state and run through graph
            state = AgentState(room=room, messages=[command], config=self.config)
            result = await self.graph.ainvoke(state)

            # Extract execution details from result
            return self._build_response(result, room)

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

    def _build_response(self, result, room: str) -> Dict[str, Any]:
        """Build response from LangGraph result"""
        try:
            # Handle both object and dict formats for result
            if hasattr(result, "reasoning"):
                reasoning_data = result.reasoning
                target_position = getattr(result, "target_position", 50)
            elif isinstance(result, dict):
                reasoning_data = result.get("reasoning")
                target_position = result.get("target_position", 50)
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                return self._error_response("Unexpected response format", room)

            # Default values
            affected_blinds = [blind.name for blind in self.config.rooms[room].blinds]
            message = "Position updated successfully"

            # Try to extract better info from reasoning_data
            if reasoning_data and isinstance(reasoning_data, dict):
                # If it's ExecutionResult format
                if "executed_blinds" in reasoning_data:
                    affected_blinds = reasoning_data.get(
                        "executed_blinds", affected_blinds
                    )
                    reasoning_text = reasoning_data.get(
                        "reasoning", "Operation completed"
                    )
                    message = f"{reasoning_text}. Affected blinds: {', '.join(affected_blinds)}"
                # If it's ShadeAnalysis format
                elif "reasoning" in reasoning_data:
                    reasoning_text = reasoning_data.get("reasoning", "Position updated")
                    message = f"{reasoning_text}. Affected blinds: {', '.join(affected_blinds)}"

            return {
                "position": target_position,
                "message": message,
                "room": room,
                "affected_blinds": affected_blinds,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.warning(f"Error building response: {e}")
            logger.warning(f"Result type: {type(result)}, content: {result}")
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
            current_positions = await self._get_room_current_positions(room)
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
