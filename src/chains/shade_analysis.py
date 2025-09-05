"""
Shade analysis chain for parsing commands and determining actions
"""

from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from models.requests import ShadeAnalysis, BlindConfig
import logging

logger = logging.getLogger(__name__)


class ShadeAnalysisChain:
    """Chain for analyzing shade commands and determining positions/scope"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

    async def ainvoke(self, input_data: Dict[str, Any]) -> ShadeAnalysis:
        """LangChain-style ainvoke method"""
        command = input_data.get("command", "")
        room = input_data.get("room", "")
        room_blinds = input_data.get("room_blinds", [])
        current_positions = input_data.get("current_positions", {})
        window_sun_info = input_data.get("window_sun_info", {})
        house_orientation = input_data.get("house_orientation", "")
        notes = input_data.get("notes", "")

        # Build context strings
        solar_context = self._build_solar_context(window_sun_info)
        house_info = self._build_house_info(house_orientation, notes)

        return await self.analyze_shade_request(
            command, room, room_blinds, current_positions, solar_context, house_info
        )

    def _build_solar_context(self, window_sun_info: Dict[str, Any]) -> str:
        """Build solar context string from window sun info"""
        if "error" in window_sun_info:
            return ""

        if "message" in window_sun_info:
            return f"\nSUN STATUS: {window_sun_info['message']}"

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
            solar_context = (
                f"\nSUN STATUS: Direct sunlight on: {', '.join(intensities)}"
            )

            # Highlight the brightest window
            if len(sorted_windows) > 1:
                brightest = sorted_windows[0]
                brightest_intensity = window_sun_info[brightest]["sun_intensity"]
                solar_context += f"\nBRIGHTEST: {brightest} has {brightest_intensity} intensity (main glare source)"

            if "solar_info" in window_sun_info:
                solar = window_sun_info["solar_info"]
                solar_context += f"\nSun direction: {solar['direction']}, elevation: {solar['elevation']:.1f}°"

            return solar_context
        else:
            return f"\nSUN STATUS: No direct sunlight on any windows in this room"

    def _build_house_info(self, house_orientation: str, notes: str) -> str:
        """Build house information context string"""
        house_info = ""
        if house_orientation:
            house_info = f"\nHOUSE LAYOUT: {house_orientation} orientation"
            if notes:
                house_info += f". {notes}"
        return house_info

    async def analyze_shade_request(
        self,
        user_message: str,
        room: str,
        room_blinds: List[BlindConfig],
        current_positions: Dict[str, int],
        solar_context: str = "",
        house_info: str = "",
    ) -> ShadeAnalysis:
        """Analyze a shade control request and return structured analysis"""

        # Create system prompt with current positions and context
        current_status = ", ".join(
            [f"{name}: {pos}%" for name, pos in current_positions.items()]
        )

        system_prompt = f"""Analyze this shade control request for room: {room}

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
        - "house": Use when command starts with [HOUSE-WIDE COMMAND] OR explicitly mentions "house", "all rooms", "entire house"
        - "room": Default for "all windows", "all shades" (current room only)  
        - "specific": When naming specific blinds OR multiple different operations

        IMPORTANT: If command starts with [HOUSE-WIDE COMMAND], automatically set scope: "house"

        Examples:
        - "Open all windows" → position: 100, scope: "room", blind_filter: []
        - "Open the front window" → position: 100, scope: "specific", blind_filter: ["front"]
        - "[HOUSE-WIDE COMMAND] close all blinds" → position: 0, scope: "house", blind_filter: []
        - "Open side halfway, front fully" → operations: [{{blind_filter: ["side"], position: 50}}, {{blind_filter: ["front"], position: 100}}], scope: "specific"

        Provide structured response. Use operations array ONLY when different positions are needed for different blinds.
        """

        try:
            # Use structured output with Pydantic model
            structured_llm = self.llm.with_structured_output(ShadeAnalysis)

            analysis: ShadeAnalysis = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"User request: {user_message}"),
                ]
            )

            return analysis

        except Exception as e:
            logger.error(f"Error in shade analysis: {e}")
            # Return fallback analysis
            return ShadeAnalysis(
                position=50,  # Default to 50%
                scope="room",
                blind_filter=[],
                reasoning=f"Fallback parsing due to error: {e}",
            )
