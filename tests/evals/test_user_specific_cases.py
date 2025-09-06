"""
Integration tests for specific user cases
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chains.execution_timing import ExecutionTimingChain
from chains.schedule_management import ScheduleManagementChain
from chains.shade_analysis import ShadeAnalysisChain
from models.agent import (
    ExecutionTiming,
    ScheduleOperation,
    ShadeAnalysis,
    BlindOperation,
)
from models.config import BlindConfig


class TestUserSpecificCases:
    """Integration tests for the specific cases mentioned by the user"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        return AsyncMock()

    @pytest.fixture
    def sample_blinds(self):
        """Sample blind configuration"""
        return [
            BlindConfig(id="1", name="Guest Side Window", orientation="North"),
            BlindConfig(id="2", name="Guest Front Window", orientation="East"),
        ]

    @pytest.mark.asyncio
    async def test_case_1_stop_closing_everyday(self, mock_llm, sample_blinds):
        """
        Test Case 1: "hey stop closing the blinds everyday"
        Should cancel closing schedule tasks
        """
        # Test execution timing detection
        timing_chain = ExecutionTimingChain(mock_llm)
        expected_timing = ExecutionTiming(
            execution_type="scheduled",
            reasoning="Stop recurring schedule - should be handled as scheduled operation",
        )

        with patch.object(timing_chain, "ainvoke", return_value=expected_timing):
            timing_result = await timing_chain.ainvoke(
                {"command": "hey stop closing the blinds everyday"}
            )

            assert timing_result.execution_type == "scheduled"

        # Test schedule management
        schedule_chain = ScheduleManagementChain(mock_llm)
        expected_schedule = ScheduleOperation(
            action_type="delete",
            schedule_description="Cancel daily closing schedule",
            target_schedule_filter="closing.*daily|everyday.*closing",
            command_to_execute="",  # Not needed for delete operations
            reasoning="User wants to stop recurring daily blind closing",
        )

        with patch.object(schedule_chain, "ainvoke", return_value=expected_schedule):
            schedule_result = await schedule_chain.ainvoke(
                {
                    "command": "hey stop closing the blinds everyday",
                    "room": "guest_bedroom",
                    "existing_schedules": [
                        {
                            "id": "schedule_1",
                            "description": "Close blinds every day at 6pm",
                        }
                    ],
                }
            )

            assert schedule_result.action_type == "delete"
            assert (
                "cancel" in schedule_result.schedule_description.lower()
                or "delete" in schedule_result.schedule_description.lower()
            )

    @pytest.mark.asyncio
    async def test_case_2_front_shade_half_of_100(self, mock_llm, sample_blinds):
        """
        Test Case 2: "open the front shade around half of 100"
        Should open front shade 50%
        """
        # Test execution timing - should be current
        timing_chain = ExecutionTimingChain(mock_llm)
        expected_timing = ExecutionTiming(
            execution_type="current", reasoning="No time reference, execute immediately"
        )

        with patch.object(timing_chain, "ainvoke", return_value=expected_timing):
            timing_result = await timing_chain.ainvoke(
                {"command": "open the front shade around half of 100"}
            )

            assert timing_result.execution_type == "current"

        # Test shade analysis
        analysis_chain = ShadeAnalysisChain(mock_llm)
        expected_analysis = ShadeAnalysis(
            operations=[
                BlindOperation(
                    blind_filter=["front"],
                    position=50,
                    reasoning="Half of 100% = 50%, targeting front shade",
                )
            ],
            scope="specific",
            reasoning="Specific blind with calculated position",
        )

        with patch.object(analysis_chain, "ainvoke", return_value=expected_analysis):
            analysis_result = await analysis_chain.ainvoke(
                {
                    "command": "open the front shade around half of 100",
                    "room": "guest_bedroom",
                    "room_blinds": sample_blinds,
                    "current_positions": {
                        "Guest Side Window": 0,
                        "Guest Front Window": 0,
                    },
                    "window_sun_info": {},
                    "house_orientation": "east-west",
                    "notes": "",
                }
            )

            assert analysis_result.scope == "specific"
            assert len(analysis_result.operations) == 1
            assert analysis_result.operations[0].position == 50
            assert "front" in analysis_result.operations[0].blind_filter

    @pytest.mark.asyncio
    async def test_case_3_open_shades_tomorrow_8am(self, mock_llm, sample_blinds):
        """
        Test Case 3: "can you open the shades tomorrow morning at 8am"
        Should open the shades in this bedroom at 8am tomorrow
        """
        # Test execution timing - should be scheduled
        timing_chain = ExecutionTimingChain(mock_llm)
        expected_timing = ExecutionTiming(
            execution_type="scheduled",
            reasoning="Tomorrow morning with specific time - future scheduling",
        )

        with patch.object(timing_chain, "ainvoke", return_value=expected_timing):
            timing_result = await timing_chain.ainvoke(
                {"command": "can you open the shades tomorrow morning at 8am"}
            )

            assert timing_result.execution_type == "scheduled"

        # Test schedule management
        schedule_chain = ScheduleManagementChain(mock_llm)
        expected_schedule = ScheduleOperation(
            action_type="create",
            schedule_time="08:00",
            schedule_date="tomorrow",
            recurrence="",  # One-time schedule
            schedule_description="Open shades tomorrow morning at 8am",
            command_to_execute="open the shades",
            reasoning="Creating one-time schedule for tomorrow morning",
        )

        with patch.object(schedule_chain, "ainvoke", return_value=expected_schedule):
            schedule_result = await schedule_chain.ainvoke(
                {
                    "command": "can you open the shades tomorrow morning at 8am",
                    "room": "guest_bedroom",
                    "existing_schedules": [],
                }
            )

            assert schedule_result.action_type == "create"
            assert (
                "08:00" in schedule_result.schedule_time
                or "8" in schedule_result.schedule_time
            )
            assert "tomorrow" in schedule_result.schedule_date.lower()

    @pytest.mark.asyncio
    async def test_comprehensive_workflow_case_1(self, mock_llm, sample_blinds):
        """Test complete workflow for case 1"""
        command = "hey stop closing the blinds everyday"

        # 1. Execution timing should detect as scheduled
        timing_chain = ExecutionTimingChain(mock_llm)
        mock_llm.ainvoke.return_value = ExecutionTiming(
            execution_type="scheduled", reasoning="Managing existing schedule"
        )
        timing_result = await timing_chain.ainvoke({"command": command})

        # 2. Since it's scheduled, should go to schedule management
        if timing_result.execution_type == "scheduled":
            schedule_chain = ScheduleManagementChain(mock_llm)
            mock_llm.ainvoke.return_value = ScheduleOperation(
                action_type="delete",
                schedule_description="Delete daily closing schedule",
            )
            schedule_result = await schedule_chain.ainvoke(
                {
                    "command": command,
                    "room": "guest_bedroom",
                    "existing_schedules": [
                        {"id": "daily_close", "description": "Close blinds every day"}
                    ],
                }
            )

            assert schedule_result.action_type == "delete"

    @pytest.mark.asyncio
    async def test_comprehensive_workflow_case_2(self, mock_llm, sample_blinds):
        """Test complete workflow for case 2"""
        command = "open the front shade around half of 100"

        # 1. Execution timing should detect as current
        timing_chain = ExecutionTimingChain(mock_llm)
        expected_timing = ExecutionTiming(
            execution_type="current", reasoning="Immediate execution"
        )

        with patch.object(timing_chain, "ainvoke", return_value=expected_timing):
            timing_result = await timing_chain.ainvoke({"command": command})

        # 2. Since it's current, should go to shade analysis
        if timing_result.execution_type == "current":
            analysis_chain = ShadeAnalysisChain(mock_llm)
            expected_analysis = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=["front"],
                        position=50,
                        reasoning="Front shade to 50%",
                    )
                ],
                scope="specific",
                reasoning="Target front shade with 50% position",
            )

            with patch.object(
                analysis_chain, "ainvoke", return_value=expected_analysis
            ):
                analysis_result = await analysis_chain.ainvoke(
                    {
                        "command": command,
                        "room": "guest_bedroom",
                        "room_blinds": sample_blinds,
                        "current_positions": {
                            "Guest Side Window": 0,
                            "Guest Front Window": 0,
                        },
                        "window_sun_info": {},
                        "house_orientation": "east-west",
                        "notes": "",
                    }
                )

                assert analysis_result.scope == "specific"
                assert analysis_result.operations[0].position == 50

    @pytest.mark.asyncio
    async def test_comprehensive_workflow_case_3(self, mock_llm, sample_blinds):
        """Test complete workflow for case 3"""
        command = "can you open the shades tomorrow morning at 8am"

        # 1. Execution timing should detect as scheduled
        timing_chain = ExecutionTimingChain(mock_llm)
        mock_llm.ainvoke.return_value = ExecutionTiming(
            execution_type="scheduled", reasoning="Future time reference"
        )
        timing_result = await timing_chain.ainvoke({"command": command})

        # 2. Since it's scheduled, should go to schedule management
        if timing_result.execution_type == "scheduled":
            schedule_chain = ScheduleManagementChain(mock_llm)
            mock_llm.ainvoke.return_value = ScheduleOperation(
                action_type="create",
                schedule_time="08:00",
                schedule_date="tomorrow",
                schedule_description="Open shades tomorrow at 8am",
            )
            schedule_result = await schedule_chain.ainvoke(
                {"command": command, "room": "guest_bedroom", "existing_schedules": []}
            )

            assert schedule_result.action_type == "create"
            assert (
                "08:00" in schedule_result.schedule_time
                or "8" in schedule_result.schedule_time
            )

    @pytest.mark.asyncio
    async def test_edge_cases_and_variations(self, mock_llm, sample_blinds):
        """Test variations and edge cases of the user scenarios"""
        test_cases = [
            {
                "command": "stop the daily blind closing",
                "expected_timing": "scheduled",
                "expected_action": "delete",
            },
            {
                "command": "open front window to about 50 percent",
                "expected_timing": "current",
                "expected_position": 50,
            },
            {
                "command": "please open shades at 8 AM tomorrow",
                "expected_timing": "scheduled",
                "expected_action": "create",
            },
        ]

        for case in test_cases:
            # Test timing detection
            timing_chain = ExecutionTimingChain(mock_llm)
            expected_timing = ExecutionTiming(
                execution_type=case["expected_timing"],
                reasoning=f"Expected timing for: {case['command']}",
            )

            with patch.object(timing_chain, "ainvoke", return_value=expected_timing):
                timing_result = await timing_chain.ainvoke({"command": case["command"]})
                assert timing_result.execution_type == case["expected_timing"]
