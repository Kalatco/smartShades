"""
Duration parsing chain for converting natural language durations to structured data
"""

import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI
from models.agent import DurationInfo

logger = logging.getLogger(__name__)


class DurationParsingChain:
    """Chain for parsing natural language duration expressions into structured data"""

    def __init__(self, llm: AzureChatOpenAI):
        """Initialize the duration parsing chain"""
        self.llm = llm

        # Define the system prompt template
        system_template = """Parse natural language duration expressions into structured duration information.

        DURATION PARSING RULES:
        - Extract the numeric value and time unit from duration expressions
        - Normalize time units to standard forms: "days", "weeks", "months"
        - Handle various natural language expressions for durations

        EXAMPLES:
        Input: "for the next week"
        Output: duration_value=1, duration_unit="weeks", total_days=7

        Input: "for 3 days"
        Output: duration_value=3, duration_unit="days", total_days=3

        Input: "for two weeks"
        Output: duration_value=2, duration_unit="weeks", total_days=14

        Input: "for a month"
        Output: duration_value=1, duration_unit="months", total_days=30

        Input: "for the next 5 days"
        Output: duration_value=5, duration_unit="days", total_days=5

        NUMERIC WORD CONVERSION:
        - "a", "one" → 1
        - "two" → 2
        - "three" → 3
        - "four" → 4
        - "five" → 5

        UNIT NORMALIZATION:
        - "day", "days" → "days"
        - "week", "weeks" → "weeks"
        - "month", "months" → "months"

        TOTAL DAYS CALCULATION:
        - days: duration_value × 1
        - weeks: duration_value × 7
        - months: duration_value × 30 (approximate)

        If duration cannot be parsed, set all values to null.

        {format_instructions}"""

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", "Duration expression: {duration_text}"),
            ]
        )

        # Create output parser for structured output
        self.output_parser = PydanticOutputParser(pydantic_object=DurationInfo)

        # Create the chain
        self.chain = self.prompt | self.llm | self.output_parser

    async def ainvoke(self, input_data: Dict[str, Any]) -> DurationInfo:
        """LangChain-style ainvoke method"""
        duration_text = input_data.get("duration_text", "")

        try:
            # Use the chain to get structured output
            duration_info = await self.chain.ainvoke(
                {
                    "duration_text": duration_text,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            )

            logger.info(f"Parsed duration '{duration_text}' -> {duration_info}")
            return duration_info

        except Exception as e:
            logger.error(f"Error in duration parsing: {e}")
            # Return fallback duration info
            return DurationInfo(
                duration_value=None,
                duration_unit=None,
                total_days=None,
                is_valid=False,
                reasoning=f"Failed to parse duration: {e}",
            )
