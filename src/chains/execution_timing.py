"""
Execution timing detection chain for determining immediate vs scheduled execution
"""

from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.agent import ExecutionTiming
import logging

logger = logging.getLogger(__name__)


class ExecutionTimingChain:
    """Chain for detecting if command should be executed immediately or scheduled"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

        # Define the system prompt template
        system_template = """Analyze this shade control command to determine if it should be executed immediately or scheduled for later.

        CURRENT EXECUTION indicators:
        - Commands with no time references: "close the blinds", "open all windows"
        - Commands with "now": "close the blinds now"
        - Present tense without future time: "turn off the lights"
        - Immediate adjustments: "close it a bit more", "open the side window"

        SCHEDULED EXECUTION indicators:
        - Specific times: "close the blinds at 9pm", "open at sunrise"
        - Future dates: "close tomorrow", "open next week"
        - Recurring patterns: "close every day at 8pm", "open weekdays at 7am"
        - Relative future time: "close after sunset", "open in 2 hours"
        - Conditional future: "when I'm out of town", "while I'm away"

        Examples:
        - "close the blinds" → CURRENT (no time specified)
        - "close the blinds at 9pm today" → SCHEDULED (specific time)
        - "close the blinds after sunset" → SCHEDULED (relative future time)
        - "open all windows" → CURRENT (immediate action)
        - "stop closing the blinds everyday" → SCHEDULED (modifying recurring schedule)
        - "close the blinds every day at 8pm" → SCHEDULED (recurring pattern)
        - "open and close the blinds everyday while I'm out" → SCHEDULED (conditional future)

        Provide reasoning for your decision.

        {format_instructions}"""

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_template), ("human", "Command: {command}")]
        )

        # Create output parser for structured output
        self.output_parser = PydanticOutputParser(pydantic_object=ExecutionTiming)

        # Create the chain
        self.chain = self.prompt | self.llm | self.output_parser

    async def ainvoke(self, input_data: Dict[str, Any]) -> ExecutionTiming:
        """LangChain-style ainvoke method"""
        command = input_data.get("command", "")

        try:
            # Use the chain to get structured output
            timing = await self.chain.ainvoke(
                {
                    "command": command,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            )

            return timing

        except Exception as e:
            logger.error(f"Error in execution timing detection: {e}")
            # Return fallback timing (assume current execution)
            return ExecutionTiming(
                execution_type="current",
                reasoning=f"Fallback to current execution due to error: {e}",
            )
