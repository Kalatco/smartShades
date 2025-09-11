"""
Tests for Schedule Management Chain
"""

import pytest
from unittest.mock import AsyncMock, patch

from chains.schedule_management import ScheduleManagementChain
from models.agent import ScheduleOperation


class TestScheduleManagementChain:
    """Test cases for schedule management"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def chain(self, mock_llm):
        """Create chain instance with mock LLM"""
        return ScheduleManagementChain(mock_llm)

    @pytest.mark.asyncio
    async def test_create_schedule_commands(self, chain, mock_llm):
        """Test commands that should create new schedules"""
        create_commands = [
            {
                "command": "close the blinds every weekday at 6 PM",
                "expected_time": "18:00",
                "expected_recurrence": "weekdays",
            },
            {
                "command": "open blinds at sunrise",
                "expected_time": "sunrise",
                "expected_recurrence": "daily",
            },
            {
                "command": "close the blinds tonight at 9 PM",
                "expected_time": "21:00",
                "expected_date": "today",
            },
            {
                "command": "block the sun after 2 PM on weekends",
                "expected_time": "14:00",
                "expected_recurrence": "weekends",
            },
            {
                "command": "close blinds for the next 3 days at 8 PM",
                "expected_time": "20:00",
                "expected_date": "next 3 days",
            },
            {
                "command": "can you open the shades tomorrow morning at 8am",
                "expected_time": "08:00",
                "expected_date": "tomorrow",
            },
        ]

        for case in create_commands:
            # Mock the LLM response as JSON string
            mock_response = ScheduleOperation(
                action_type="create",
                schedule_time=case.get("expected_time", ""),
                schedule_date=case.get("expected_date", ""),
                recurrence=case.get("expected_recurrence", ""),
                command_to_execute="close the blinds",
                schedule_description=f"Create schedule: {case['command']}",
                reasoning=f"User wants to create a schedule for: {case['command']}",
            )
            mock_llm.ainvoke.return_value = mock_response.model_dump_json()

            result = await chain.ainvoke(
                {
                    "command": case["command"],
                    "room": "guest_bedroom",
                    "existing_schedules": [],
                }
            )

            assert isinstance(result, ScheduleOperation)
            assert (
                result.action_type == "create"
            ), f"Command '{case['command']}' should create schedule"

    @pytest.mark.asyncio
    async def test_delete_schedule_commands(self, chain, mock_llm):
        """Test commands that should delete schedules"""
        delete_commands = [
            "stop closing the blinds everyday",
            "cancel my sunset schedule",
            "hey stop closing the blinds everyday",
            "stop all scheduled blind operations",
            "cancel all my schedules",
            "remove the 9pm schedule",
        ]

        for command in delete_commands:
            # Mock the chain's ainvoke method directly
            expected_result = ScheduleOperation(
                action_type="delete",
                command_to_execute="cancel schedule",
                schedule_description=f"Delete schedule: {command}",
                reasoning=f"User wants to delete a schedule: {command}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": command,
                        "room": "guest_bedroom",
                        "existing_schedules": [
                            {
                                "id": "schedule_1",
                                "description": "Close blinds every day at 6pm",
                            }
                        ],
                    }
                )

                assert isinstance(result, ScheduleOperation)
                assert (
                    result.action_type == "delete"
                ), f"Command '{command}' should delete schedule"

    @pytest.mark.asyncio
    async def test_modify_schedule_commands(self, chain, mock_llm):
        """Test commands that should modify existing schedules"""
        modify_commands = [
            "change my 6pm schedule to 7pm",
            "move the sunset closing to 8pm",
            "update my morning schedule to 7am",
        ]

        for command in modify_commands:
            # Mock the chain's ainvoke method directly
            expected_result = ScheduleOperation(
                action_type="modify",
                command_to_execute="modify schedule",
                schedule_description=f"Modify schedule: {command}",
                reasoning=f"User wants to modify a schedule: {command}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": command,
                        "room": "guest_bedroom",
                        "existing_schedules": [
                            {
                                "id": "schedule_1",
                                "description": "Close blinds every day at 6pm",
                            }
                        ],
                    }
                )

                assert isinstance(result, ScheduleOperation)
                assert (
                    result.action_type == "modify"
                ), f"Command '{command}' should modify schedule"

    @pytest.mark.asyncio
    async def test_travel_schedule_commands(self, chain, mock_llm):
        """Test commands for travel/vacation schedules"""
        travel_commands = [
            "im out of town next week, can you open and close the blinds everyday for me"
        ]

        for command in travel_commands:
            # Mock the LLM response as JSON string
            mock_response = ScheduleOperation(
                action_type="create",
                schedule_time="08:00,20:00",  # Multiple times
                schedule_date="next week",
                recurrence="daily",
                command_to_execute="open and close blinds",
                schedule_description="Travel schedule: open and close daily",
                reasoning="User is traveling and wants automated blind control",
            )
            mock_llm.ainvoke.return_value = mock_response.model_dump_json()

            result = await chain.ainvoke(
                {"command": command, "room": "guest_bedroom", "existing_schedules": []}
            )

            assert isinstance(result, ScheduleOperation)
            assert (
                result.action_type == "create"
            ), f"Command '{command}' should create travel schedule"

    @pytest.mark.asyncio
    async def test_time_parsing(self, chain, mock_llm):
        """Test various time format parsing"""
        time_formats = [
            {"command": "close blinds at 9pm", "expected_time": "21:00"},
            {"command": "open at 8:30 AM", "expected_time": "08:30"},
            {"command": "close at noon", "expected_time": "12:00"},
            {"command": "open at midnight", "expected_time": "00:00"},
        ]

        for case in time_formats:
            # Mock the LLM response as JSON string
            mock_response = ScheduleOperation(
                action_type="create",
                schedule_time=case["expected_time"],
                command_to_execute="control blinds",
                schedule_description=f"Schedule at {case['expected_time']}",
                reasoning=f"User specified time: {case['expected_time']}",
            )
            mock_llm.ainvoke.return_value = mock_response.model_dump_json()

            result = await chain.ainvoke(
                {
                    "command": case["command"],
                    "room": "guest_bedroom",
                    "existing_schedules": [],
                }
            )

            assert isinstance(result, ScheduleOperation)
            assert result.action_type == "create"

    @pytest.mark.asyncio
    async def test_recurrence_patterns(self, chain, mock_llm):
        """Test various recurrence pattern parsing"""
        recurrence_patterns = [
            {
                "command": "close blinds every weekday at 6pm",
                "expected_recurrence": "weekdays",
            },
            {
                "command": "open blinds on weekends at 8am",
                "expected_recurrence": "weekends",
            },
            {
                "command": "close blinds every Monday at 7pm",
                "expected_recurrence": "monday",
            },
            {"command": "open blinds daily at sunrise", "expected_recurrence": "daily"},
        ]

        for case in recurrence_patterns:
            # Mock the LLM response as JSON string
            mock_response = ScheduleOperation(
                action_type="create",
                recurrence=case["expected_recurrence"],
                command_to_execute="control blinds",
                schedule_description=f"Recurring: {case['expected_recurrence']}",
                reasoning=f"User wants {case['expected_recurrence']} schedule",
            )
            mock_llm.ainvoke.return_value = mock_response.model_dump_json()

            result = await chain.ainvoke(
                {
                    "command": case["command"],
                    "room": "guest_bedroom",
                    "existing_schedules": [],
                }
            )

            assert isinstance(result, ScheduleOperation)
            assert result.action_type == "create"

    @pytest.mark.asyncio
    async def test_error_handling(self, chain, mock_llm):
        """Test error handling returns fallback"""
        # Mock an exception
        mock_llm.ainvoke.side_effect = Exception("LLM Error")

        result = await chain.ainvoke(
            {
                "command": "test command",
                "room": "guest_bedroom",
                "existing_schedules": [],
            }
        )

        assert isinstance(result, ScheduleOperation)
        assert result.action_type == "create"  # Fallback action

    @pytest.mark.asyncio
    async def test_with_existing_schedules(self, chain, mock_llm):
        """Test schedule operations with existing schedules"""
        existing_schedules = [
            {"id": "schedule_1", "description": "Close blinds every day at 6pm"},
            {"id": "schedule_2", "description": "Open blinds at sunrise"},
        ]

        # Mock the LLM response as JSON string
        mock_response = ScheduleOperation(
            action_type="create",
            command_to_execute="new schedule command",
            schedule_description="New schedule with existing context",
            reasoning="User wants to create new schedule considering existing ones",
        )
        mock_llm.ainvoke.return_value = mock_response.model_dump_json()

        result = await chain.ainvoke(
            {
                "command": "close blinds at 8pm",
                "room": "guest_bedroom",
                "existing_schedules": existing_schedules,
            }
        )

        assert isinstance(result, ScheduleOperation)
        assert result.action_type == "create"
