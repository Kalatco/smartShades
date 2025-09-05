"""
House-wide command detection chain
"""

from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)


class HouseWideDetectionChain:
    """Chain for detecting if a command is meant to be house-wide"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

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

        Respond with only "true" or "false"."""

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "Command: {command}")]
        )

        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def ainvoke(self, input_data: Dict[str, Any]) -> bool:
        """LangChain-style ainvoke method for detecting house-wide commands"""
        command = input_data.get("command", "")

        try:
            response = await self.chain.ainvoke({"command": command})

            # Extract boolean from response
            response_text = response.lower().strip()
            return response_text == "true"

        except Exception as e:
            logger.error(f"Error detecting house-wide command: {e}")
            # Default to False (room-specific) on error
            return False
