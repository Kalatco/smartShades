"""
Smart Shades Agent implementation using LangGraph
"""

import asyncio
import logging
import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import httpx

from models.requests import AgentState, HubitatConfig, ShadeAnalysis, ExecutionResult

logger = logging.getLogger(__name__)


class SmartShadesAgent:
    """LangGraph-based agent for intelligent shade control"""

    def __init__(self):
        self.llm = None
        self.graph = None
        self.config = None

    async def initialize(self):
        """Initialize the agent and LangGraph"""
        # Load configuration
        await self._load_config()

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

            # Create system prompt with current positions
            current_status = ", ".join(
                [f"{name}: {pos}%" for name, pos in current_positions.items()]
            )

            system_prompt = f"""Analyze this shade control request for room: {state.room}

Available blinds in THIS room: {[blind.name for blind in room_blinds]}
CURRENT POSITIONS: {current_status}

Determine:
1. Position (0-100%): 
   - For absolute commands: "open/up"=100, "close/down"=0, "half"=50, or extract number
   - For relative commands: "a little more/bit more" = current + 10-15%, "much more" = current + 25-30%
   - For relative commands: "a little less/bit less" = current - 10-15%, "much less" = current - 25-30%
   - "just a little bit" typically means small adjustment: +/- 10%
2. Scope - IMPORTANT RULES:
   - "house": ONLY if explicitly mentions "house", "all rooms", "entire house", "everywhere" 
   - "room": Default for commands like "all windows", "all shades", "all blinds" (affects current room only)
   - "specific": When naming specific blinds like "front window", "bedroom shade"
3. Blind filters: keywords to match blind names (e.g., ["front"] for "front shade")

Examples:
- "Open all windows" → scope: "room" (current room only)
- "Close all the shades" → scope: "room" (current room only) 
- "Open all windows in the house" → scope: "house" (all rooms)
- "Open the front window" → scope: "specific", blind_filter: ["front"]
- "Open the side window just a little bit" (current: 50%) → position: 60%, scope: "specific", blind_filter: ["side"]

Provide structured response with position, scope, blind_filter list, and reasoning."""

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
                state.target_position = analysis.position
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
                target_blinds, affected_rooms = self._get_target_blinds(
                    analysis, state.room
                )

                if not target_blinds:
                    state.reasoning = ExecutionResult(
                        executed_blinds=[],
                        affected_rooms=[],
                        total_blinds=0,
                        position=state.target_position,
                        scope=analysis.scope,
                        reasoning=f"No matching blinds found for: {analysis.blind_filter}",
                    ).model_dump()
                    return state

                # Execute blind control in background thread (LangGraph is synchronous)
                self._execute_blind_control_async(target_blinds, state.target_position)

                # Create success result
                state.reasoning = ExecutionResult(
                    executed_blinds=[blind.name for blind in target_blinds],
                    affected_rooms=affected_rooms,
                    total_blinds=len(target_blinds),
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
        async with httpx.AsyncClient() as client:
            for blind in blinds:
                url = f"{self.config.hubitatUrl}/apps/api/{self.config.makerApiId}/devices/{blind.id}/setPosition/{position}?access_token={self.config.accessToken}"

                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        logger.info(f"Successfully set {blind.name} to {position}%")
                    else:
                        logger.error(
                            f"Failed to control {blind.name}: HTTP {response.status_code}"
                        )
                except Exception as e:
                    logger.error(f"Error controlling {blind.name}: {e}")

    async def _get_blind_current_position(self, blind_id: str) -> int:
        """Get current position of a specific blind from Hubitat"""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.config.hubitatUrl}/apps/api/{self.config.makerApiId}/devices/{blind_id}?access_token={self.config.accessToken}"
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
