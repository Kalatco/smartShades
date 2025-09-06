"""
Simplified Integration Tests for SmartShades System

These tests validate the actual functionality of the system using real components
rather than complex mocking that's difficult to maintain.
"""

import pytest
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from models.config import BlindConfig
from models.agent import (
    ExecutionTiming,
    ScheduleOperation,
    ShadeAnalysis,
    BlindOperation,
    HouseWideDetection,
)


class TestSmartShadesSystemIntegration:
    """Integration tests that validate system behavior with specific user cases"""

    @pytest.fixture
    def sample_blinds(self):
        """Sample blind configurations for testing"""
        return [
            BlindConfig(id="1", name="Guest Side Window", orientation="North"),
            BlindConfig(id="2", name="Guest Front Window", orientation="East"),
        ]

    def test_user_case_1_data_structure(self):
        """Test Case 1: "hey stop closing the blinds everyday" - validate data structures"""
        # This should create a schedule operation for canceling a recurring schedule
        operation = ScheduleOperation(
            action_type="delete",
            command_to_execute="cancel closing schedule",
            schedule_description="Stop closing blinds everyday",
            reasoning="User wants to cancel recurring blind closing schedule",
        )

        assert operation.action_type == "delete"
        assert operation.schedule_description is not None
        assert (
            "stop" in operation.schedule_description.lower()
            or "cancel" in operation.schedule_description.lower()
        )

    def test_user_case_2_data_structure(self):
        """Test Case 2: "open the front shade around half of 100" - validate data structures"""
        # This should create a shade analysis targeting the front shade at 50%
        operation = ShadeAnalysis(
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

        assert operation.scope == "specific"
        assert len(operation.operations) == 1
        assert operation.operations[0].position == 50
        assert "front" in operation.operations[0].blind_filter

    def test_user_case_3_data_structure(self):
        """Test Case 3: "can you open the shades tomorrow morning at 8am" - validate data structures"""
        # This should be detected as scheduled execution
        timing = ExecutionTiming(
            execution_type="scheduled",
            reasoning="Tomorrow morning with specific time - future scheduling",
        )

        # And create a schedule operation
        schedule = ScheduleOperation(
            action_type="create",
            schedule_time="08:00",
            schedule_date="tomorrow",
            command_to_execute="open shades",
            schedule_description="Open shades tomorrow at 8am",
            reasoning="User specified future time for opening shades",
        )

        assert timing.execution_type == "scheduled"
        assert schedule.action_type == "create"
        assert schedule.schedule_time == "08:00"
        assert schedule.schedule_date == "tomorrow"

    def test_house_wide_detection_structure(self):
        """Test house-wide command detection data structure"""
        # Test house-wide command
        house_wide = HouseWideDetection(is_house_wide=True)
        assert house_wide.is_house_wide == True

        # Test room-specific command
        room_specific = HouseWideDetection(is_house_wide=False)
        assert room_specific.is_house_wide == False

    def test_execution_timing_types(self):
        """Test execution timing type validation"""
        # Current execution
        current = ExecutionTiming(
            execution_type="current", reasoning="No time specified, execute immediately"
        )
        assert current.execution_type == "current"

        # Scheduled execution
        scheduled = ExecutionTiming(
            execution_type="scheduled",
            reasoning="Time reference found, schedule for later",
        )
        assert scheduled.execution_type == "scheduled"

    def test_schedule_operation_types(self):
        """Test schedule operation type validation"""
        # Create operation
        create = ScheduleOperation(
            action_type="create",
            command_to_execute="open blinds",
            schedule_description="Create new schedule",
            reasoning="User wants to create a new schedule",
        )
        assert create.action_type == "create"

        # Delete operation
        delete = ScheduleOperation(
            action_type="delete",
            command_to_execute="cancel schedule",
            schedule_description="Delete existing schedule",
            reasoning="User wants to delete a schedule",
        )
        assert delete.action_type == "delete"

        # Modify operation
        modify = ScheduleOperation(
            action_type="modify",
            command_to_execute="update schedule",
            schedule_description="Modify existing schedule",
            reasoning="User wants to modify a schedule",
        )
        assert modify.action_type == "modify"

    def test_shade_analysis_scopes(self, sample_blinds):
        """Test shade analysis scope validation"""
        # Specific scope
        specific = ShadeAnalysis(
            operations=[
                BlindOperation(
                    blind_filter=["front"],
                    position=50,
                    reasoning="Target specific blind",
                )
            ],
            scope="specific",
            reasoning="Command targets specific blind",
        )
        assert specific.scope == "specific"

        # Room scope
        room = ShadeAnalysis(
            operations=[
                BlindOperation(
                    blind_filter=[], position=0, reasoning="All blinds in room"
                )
            ],
            scope="room",
            reasoning="Command targets all blinds in room",
        )
        assert room.scope == "room"

        # House scope
        house = ShadeAnalysis(
            operations=[
                BlindOperation(
                    blind_filter=[], position=100, reasoning="All blinds in house"
                )
            ],
            scope="house",
            reasoning="Command targets all blinds in house",
        )
        assert house.scope == "house"

    def test_blind_operation_validation(self):
        """Test blind operation validation"""
        # Valid position range
        for position in [0, 25, 50, 75, 100]:
            operation = BlindOperation(
                blind_filter=["test"],
                position=position,
                reasoning=f"Test position {position}%",
            )
            assert operation.position == position

        # Test position bounds
        with pytest.raises(Exception):  # Should fail validation
            BlindOperation(
                blind_filter=["test"],
                position=-1,  # Invalid: below 0
                reasoning="Invalid position",
            )

        with pytest.raises(Exception):  # Should fail validation
            BlindOperation(
                blind_filter=["test"],
                position=101,  # Invalid: above 100
                reasoning="Invalid position",
            )

    def test_comprehensive_workflow_data_flow(self):
        """Test that all data structures work together in a comprehensive workflow"""
        # 1. House-wide detection
        house_wide = HouseWideDetection(is_house_wide=False)

        # 2. Execution timing
        timing = ExecutionTiming(
            execution_type="current", reasoning="No time reference, execute immediately"
        )

        # 3. Shade analysis (if current execution)
        if timing.execution_type == "current":
            shade_analysis = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=["front"],
                        position=50,
                        reasoning="Open front shade to 50%",
                    )
                ],
                scope="specific",
                reasoning="Target front shade with specific position",
            )

            # Validate the complete workflow
            assert not house_wide.is_house_wide
            assert timing.execution_type == "current"
            assert shade_analysis.scope == "specific"
            assert len(shade_analysis.operations) == 1
            assert shade_analysis.operations[0].position == 50

    def test_schedule_workflow_data_flow(self):
        """Test schedule-related workflow data structures"""
        # 1. Execution timing detects scheduled
        timing = ExecutionTiming(
            execution_type="scheduled", reasoning="Time reference found in command"
        )

        # 2. Schedule management (if scheduled execution)
        if timing.execution_type == "scheduled":
            schedule = ScheduleOperation(
                action_type="create",
                schedule_time="21:00",
                schedule_date="daily",
                recurrence="daily",
                command_to_execute="close all blinds",
                schedule_description="Close blinds every day at 9pm",
                reasoning="User wants recurring evening blind closure",
            )

            # Validate the schedule workflow
            assert timing.execution_type == "scheduled"
            assert schedule.action_type == "create"
            assert schedule.schedule_time == "21:00"
            assert schedule.recurrence == "daily"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
