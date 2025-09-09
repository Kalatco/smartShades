"""
Blind execution planning chain for generating BlindExecutionRequest from user commands
"""

from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.agent import BlindExecutionRequest
from models.config import HubitatConfig, BlindConfig, RoomConfig
from utils.hubitat_utils import HubitatUtils
import logging

logger = logging.getLogger(__name__)


class BlindExecutionPlanningChain:
    """Chain for analyzing user commands and generating BlindExecutionRequest"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

        # Define the system prompt template
        system_template = """You are a smart home assistant that converts natural language commands into specific blind control instructions.

        Given a user command and available rooms/blinds, generate a BlindExecutionRequest that specifies exactly which blinds to control and their target positions.

        AVAILABLE ROOMS AND BLINDS:
        {rooms_info}

        HOUSE INFORMATION:
        {house_information}

        CURRENT ROOM CONTEXT: {current_room}

        CURRENT BLIND POSITIONS:
        {current_positions}

        POSITION GUIDELINES:
        - "open/up/fully/all the way" = 100
        - "close/down/shut" = 0  
        - "half/halfway/middle" = 50
        - "quarter/little bit" = 25
        - "most of the way/almost" = 75
        
        RELATIVE POSITIONING (based on current position):
        - "open more/open it more" = current + 15
        - "open much more/open way more" = current + 30
        - "close more/close it more" = current - 15
        - "close much more/close way more" = current - 30
        - "open halfway" when already open = move halfway between current and 100
        - "close halfway" when already closed = move halfway between current and 0
        
        CONTEXTUAL POSITIONING:
        - If blind is currently closed (0-25%) and command says "open halfway" = 50%
        - If blind is currently open (75-100%) and command says "close halfway" = 50%
        - If blind is at 50% and command says "open halfway" = 75% (halfway to fully open)
        - If blind is at 50% and command says "close halfway" = 25% (halfway to fully closed)
        - If blind is at 30% and command says "open halfway" = 65% (halfway between 30% and 100%)
        - If blind is at 70% and command says "close halfway" = 35% (halfway between 70% and 0%)
        
        ABSOLUTE VS RELATIVE INTERPRETATION:
        - "set to 50%" or "make it 50%" = absolute position 50%
        - "open halfway" = relative to current state (move halfway toward open)
        - "close halfway" = relative to current state (move halfway toward closed)

        SCOPE DETECTION:
        - ROOM SCOPE: "all windows", "all blinds", "open the room", "close everything here"
        - HOUSE SCOPE: "entire house", "all rooms", "house-wide", "everywhere", "all blinds in the house"
        - SPECIFIC SCOPE: Mentions specific blind names, orientations, or locations

        BLIND MATCHING:
        - Match by name keywords: "front window" matches blinds with "front" in name
        - Match by orientation: "south windows" matches blinds with orientation="south"
        - Match by location descriptors: "living room windows" when in living room context

        RESPONSE FORMAT:
        Generate a BlindExecutionRequest with:
        - rooms: Dictionary mapping room names to RoomBlindsExecution objects
        - Each RoomBlindsExecution contains blinds: Dictionary mapping blind IDs to target positions (0-100)

        EXAMPLES:

        Command: "close all blinds"
        Current Room: "living_room"
        Result: Close all blinds in living_room only

        Command: "open all blinds in the house"  
        Current Room: "bedroom"
        Result: Open all blinds in ALL rooms

        Command: "close the front window"
        Current Room: "kitchen" 
        Result: Close only blinds in kitchen that match "front"

        Command: "set living room to 50%"
        Current Room: "bedroom"
        Result: Set all blinds in living_room to 50

        Command: "open more" (when current position is 60%)
        Current Room: "living_room"
        Result: Set blinds to 75% (current + 15)

        Command: "open the side halfway" (when side window is at 50%)
        Current Room: "living_room"
        Result: Set side window to 75% (halfway between 50% and 100%)

        Command: "close the front halfway" (when front window is at 80%)
        Current Room: "living_room"
        Result: Set front window to 40% (halfway between 80% and 0%)

        Command: "open halfway" (when blinds are closed at 10%)
        Current Room: "living_room"
        Result: Set blinds to 50% (halfway point, since they're mostly closed)

        {format_instructions}"""

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_template), ("human", "User command: {user_command}")]
        )

        # Create output parser for structured output
        self.output_parser = PydanticOutputParser(pydantic_object=BlindExecutionRequest)

        # Create the chain
        self.chain = self.prompt | self.llm | self.output_parser

    async def ainvoke(self, input_data: Dict[str, Any]) -> BlindExecutionRequest:
        """
        Analyze user command and generate BlindExecutionRequest

        Args:
            input_data: Dictionary containing:
                - command: User's natural language command
                - current_room: Name of the current room context
                - config: HubitatConfig object with room/blind data

        Returns:
            BlindExecutionRequest with specific blind control instructions
        """
        command = input_data.get("command", "")
        current_room = input_data.get("current_room", "")
        config = input_data.get("config")

        if not config or not isinstance(config, HubitatConfig):
            logger.error("Invalid or missing HubitatConfig in input_data")
            # Return empty request
            from models.agent import RoomBlindsExecution

            return BlindExecutionRequest(rooms={})

        try:
            # Get current positions for all rooms to support relative commands
            current_positions = {}
            for room_name in config.rooms.keys():
                try:
                    room_positions = await HubitatUtils.get_room_current_positions(
                        config, room_name
                    )
                    if room_positions:
                        current_positions[room_name] = room_positions
                except Exception as e:
                    logger.warning(
                        f"Could not get current positions for room {room_name}: {e}"
                    )
                    current_positions[room_name] = {}

            # Format current positions for the prompt
            positions_text = ""
            if current_positions:
                positions_text = "Current blind positions by room:\n"
                for room_name, positions in current_positions.items():
                    positions_text += f"  {room_name}:\n"
                    for blind_name, position in positions.items():
                        positions_text += f"    - {blind_name}: {position}%\n"
            else:
                positions_text = "Current positions unavailable"

            # Use the chain to get structured output
            execution_request = await self.chain.ainvoke(
                {
                    "user_command": command,
                    "current_room": current_room,
                    "rooms_info": getattr(config, "rooms", {}),
                    "house_information": getattr(config, "houseInformation", {}),
                    "current_positions": positions_text,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            )

            logger.info(f"Generated BlindExecutionRequest for command: '{command}'")
            logger.info(f"Rooms affected: {list(execution_request.rooms.keys())}")
            logger.info(f"Execution request details: {execution_request}")

            return execution_request

        except Exception as e:
            logger.error(f"Error in blind execution planning: {e}")
            # Return empty request as fallback
            from models.agent import RoomBlindsExecution

            return BlindExecutionRequest(rooms={})
