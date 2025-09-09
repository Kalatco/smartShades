"""
Schedule management chain for creating, modifying, and deleting scheduled shade operations
"""

from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.agent import ScheduleOperation
import logging

logger = logging.getLogger(__name__)


class ScheduleManagementChain:
    """Chain for parsing schedule commands and determining schedule operations"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

        # Define the system prompt template
        system_template = """Parse this scheduling command and determine the appropriate schedule operation.

        EXISTING SCHEDULES: {existing_schedules}

        ACTION TYPES:
        - CREATE: New schedule that doesn't conflict with existing ones
        - MODIFY: Update existing schedule with new time/command (when similar schedule exists)
        - DELETE: Remove existing schedule (when user says "stop", "cancel", "remove")

        TIME PARSING:
        - "9pm", "21:00", "9 PM" → "21:00"
        - "sunrise", "sunset" → ALWAYS keep as "sunrise" or "sunset" (system will resolve)
        - "after sunset" → "sunset+0" (system will add offset)
        - "15 minutes after sunset" → "sunset+15m"
        - "30 minutes before sunrise" → "sunrise-30m"
        - "in 2 hours" → "now+2h" (system will calculate)
        
        CRITICAL: When the command contains "sunset" or "sunrise", ALWAYS preserve these exact words in schedule_time.
        Do NOT set schedule_time to None when sunset/sunrise is mentioned.

        DATE PARSING:
        - "today" → current date
        - "tomorrow" → next day
        - "next week" → date range
        - "everyday", "daily" → recurrence pattern

        RECURRENCE PATTERNS:
        - "everyday", "daily" → "daily"
        - "weekdays" → "weekdays"
        - "weekends" → "weekends"
        - "every Monday" → "weekly"
        - No mention → "once"

        DURATION PARSING:
        - "for the next week", "for a week" → "week" 
        - "for 3 days", "for the next 3 days" → "3 days"
        - "for 2 weeks" → "2 weeks"
        - "for a month" → "month"
        - No mention → None (indefinite)

        IMPORTANT: When duration is specified (like "for the next week"), the recurrence should be "daily" unless explicitly stated otherwise.

        DECISION LOGIC:
        1. If user says "stop", "cancel", "remove" → DELETE existing schedule
        2. If similar schedule exists (same recurrence + similar time) → MODIFY
        3. If no conflicts → CREATE new schedule

        EXAMPLES:
        - "close blinds at 9pm today" + existing "close at 7pm today" → MODIFY (update time)
        - "stop closing blinds everyday" + existing daily close → DELETE
        - "open blinds at sunrise daily" + no existing sunrise schedule → CREATE
        - "close at 10pm today" + existing "close at 7pm today" → MODIFY (same day, different time)
        - "close the front shade to 40% at sunset" → CREATE with schedule_time="sunset"
        - "for the next week, close the shades 15 minutes after sunset" → CREATE with schedule_time="sunset+15m", recurrence="daily", duration="week"
        - "for 3 days, open blinds 30 minutes before sunrise" → CREATE with schedule_time="sunrise-30m", recurrence="daily", duration="3 days"
        - "open blinds at sunset tomorrow" → CREATE with schedule_time="sunset", schedule_date="tomorrow"
        
        SOLAR TIME EXAMPLES:
        Input: "close the front shade to 40% at sunset"
        Output: schedule_time="sunset", command_to_execute="close the front shade to 40%"
        
        Input: "open at sunrise daily" 
        Output: schedule_time="sunrise", recurrence="daily", command_to_execute="open"

        Input: "close shades 15 minutes after sunset"
        Output: schedule_time="sunset+15m", command_to_execute="close shades"
        
        Input: "open blinds 30 minutes before sunrise"
        Output: schedule_time="sunrise-30m", command_to_execute="open blinds"

        DURATION EXAMPLES:
        Input: "close the shades at 9pm daily for the next week"
        Output: schedule_time="21:00", recurrence="daily", duration="week", command_to_execute="close the shades"
        
        Input: "open blinds every morning for 3 days"
        Output: schedule_time="sunrise", recurrence="daily", duration="3 days", command_to_execute="open blinds"

        Extract the core shade command (without timing): "close the blinds at 9pm" → "close the blinds"

        {format_instructions}"""

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_template), ("human", "Schedule command: {command}")]
        )

        # Create output parser for structured output
        self.output_parser = PydanticOutputParser(pydantic_object=ScheduleOperation)

        # Create the chain
        self.chain = self.prompt | self.llm | self.output_parser

    async def ainvoke(self, input_data: Dict[str, Any]) -> ScheduleOperation:
        """LangChain-style ainvoke method"""
        command = input_data.get("command", "")
        existing_schedules = input_data.get("existing_schedules", [])
        room = input_data.get("room", "")

        # Format existing schedules for context
        schedules_text = self._format_existing_schedules(existing_schedules)

        try:
            # Use the chain to get structured output
            schedule_op = await self.chain.ainvoke(
                {
                    "command": command,
                    "existing_schedules": schedules_text,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            )

            return schedule_op

        except Exception as e:
            logger.error(f"Error in schedule management: {e}")
            # Return fallback schedule operation
            return ScheduleOperation(
                action_type="create",
                schedule_time=None,
                schedule_date="today",
                recurrence="once",
                command_to_execute=command,
                schedule_description=f"Fallback schedule for: {command}",
                reasoning=f"Fallback parsing due to error: {e}",
            )

    def _format_existing_schedules(self, schedules: List[Dict[str, Any]]) -> str:
        """Format existing schedules for prompt context"""
        if not schedules:
            return "No existing schedules"

        formatted = []
        for i, schedule in enumerate(schedules):
            schedule_info = (
                f"{i+1}. ID: {schedule.get('id', 'unknown')} - "
                f"{schedule.get('description', 'No description')} - "
                f"Next run: {schedule.get('next_run_time', 'unknown')}"
            )
            formatted.append(schedule_info)

        return "\n".join(formatted)
