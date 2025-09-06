"""
House-wide command detection chain
"""

from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.agent import HouseWideDetection
import logging

logger = logging.getLogger(__name__)


class HouseWideDetectionChain:
    """Chain for detecting if a command is meant to be house-wide"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

        # Create the parser
        self.parser = PydanticOutputParser(pydantic_object=HouseWideDetection)

        # Define the prompt template
        system_prompt = """Analyze this shade/blind control command to determine if it's meant to affect the entire house or just a specific room.

        Return True ONLY if the command explicitly mentions:
        - "house" or "entire house" or "whole house"
        - "all rooms" or "every room"
        - "everywhere" in the context of the entire house

        Return False for:
        - Commands mentioning specific rooms (even if multiple rooms)
        - Commands saying "all windows" or "all blinds" (these refer to current room)
        - General commands without house-wide indicators
        - Commands with specific blind names or orientations

        Examples:
        - "close all blinds in the house" → True
        - "open all windows in every room" → True
        - "close all blinds" → False (current room only)
        - "open the living room and bedroom windows" → False (specific rooms)
        - "close all windows" → False (current room only)

        {format_instructions}"""

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "Command: {command}")]
        )

        # Create the chain
        self.chain = self.prompt | self.llm | self.parser

    async def ainvoke(self, input_data: Dict[str, Any]) -> HouseWideDetection:
        """LangChain-style ainvoke method for detecting house-wide commands"""
        command = input_data.get("command", "")

        try:
            # Include format instructions in the prompt
            prompt_input = {
                "command": command,
                "format_instructions": self.parser.get_format_instructions(),
            }

            response = await self.chain.ainvoke(prompt_input)

            # Log the detection result
            if response.is_house_wide:
                logger.info(f"Command detected as HOUSE-WIDE: '{command}'")
            else:
                logger.info(f"Command detected as ROOM-SPECIFIC: '{command}'")

            # Return the HouseWideDetection object
            return response

        except Exception as e:
            logger.error(f"Error detecting house-wide command: {e}")
            # Default to False (room-specific) on error
            return HouseWideDetection(is_house_wide=False)
