"""
Simple Functional Tests for SmartShades Chains

These tests focus on testing the chain initialization and basic functionality
without complex LangChain mocking, which is difficult to maintain.

For comprehensive validation, use:
1. test_integration_simplified.py - Data structure validation
2. API testing - End-to-end validation via FastAPI endpoints
3. Manual testing - Real LLM integration testing
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chains.execution_timing import ExecutionTimingChain
from chains.house_wide_detection import HouseWideDetectionChain
from chains.schedule_management import ScheduleManagementChain
from chains.shade_analysis import ShadeAnalysisChain
from models.agent import (
    ExecutionTiming,
    HouseWideDetection,
    ScheduleOperation,
    ShadeAnalysis,
    BlindOperation,
)
from models.config import BlindConfig
from unittest.mock import AsyncMock


class TestChainInitialization:
    """Test that all chains can be initialized properly"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for initialization"""
        return AsyncMock()

    def test_execution_timing_chain_init(self, mock_llm):
        """Test ExecutionTimingChain initialization"""
        chain = ExecutionTimingChain(mock_llm)
        assert chain.llm is not None
        assert chain.prompt is not None
        assert chain.output_parser is not None
        assert chain.chain is not None

    def test_house_wide_detection_chain_init(self, mock_llm):
        """Test HouseWideDetectionChain initialization"""
        chain = HouseWideDetectionChain(mock_llm)
        assert chain.llm is not None
        assert chain.prompt is not None
        assert (
            chain.parser is not None
        )  # This chain uses 'parser' instead of 'output_parser'
        assert chain.chain is not None

    def test_schedule_management_chain_init(self, mock_llm):
        """Test ScheduleManagementChain initialization"""
        chain = ScheduleManagementChain(mock_llm)
        assert chain.llm is not None
        assert chain.prompt is not None
        assert chain.output_parser is not None
        assert chain.chain is not None

    def test_shade_analysis_chain_init(self, mock_llm):
        """Test ShadeAnalysisChain initialization"""
        chain = ShadeAnalysisChain(mock_llm)
        assert chain.llm is not None
        assert chain.prompt is not None
        assert chain.output_parser is not None
        assert chain.chain is not None


class TestUserCaseDataValidation:
    """Test specific user cases through data structure validation"""

    def test_case_1_stop_closing_everyday_data_flow(self):
        """
        User Case 1: "hey stop closing the blinds everyday"
        Validates the expected data flow for canceling schedules
        """
        # Expected timing detection
        timing = ExecutionTiming(
            execution_type="scheduled",
            reasoning="Schedule management command requires scheduled processing",
        )
        assert timing.execution_type == "scheduled"

        # Expected schedule operation
        schedule_op = ScheduleOperation(
            action_type="delete",
            command_to_execute="cancel closing schedule",
            schedule_description="Stop closing blinds everyday",
            reasoning="User wants to cancel recurring blind closing schedule",
        )
        assert schedule_op.action_type == "delete"
        assert "stop" in schedule_op.schedule_description.lower()

    def test_case_2_front_shade_50_percent_data_flow(self):
        """
        User Case 2: "open the front shade around half of 100"
        Validates the expected data flow for specific blind positioning
        """
        # Expected timing detection
        timing = ExecutionTiming(
            execution_type="current", reasoning="No time reference, execute immediately"
        )
        assert timing.execution_type == "current"

        # Expected shade analysis
        shade_analysis = ShadeAnalysis(
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
        assert shade_analysis.scope == "specific"
        assert len(shade_analysis.operations) == 1
        assert shade_analysis.operations[0].position == 50
        assert "front" in shade_analysis.operations[0].blind_filter

    def test_case_3_tomorrow_8am_data_flow(self):
        """
        User Case 3: "can you open the shades tomorrow morning at 8am"
        Validates the expected data flow for future scheduling
        """
        # Expected timing detection
        timing = ExecutionTiming(
            execution_type="scheduled",
            reasoning="Tomorrow morning with specific time - future scheduling",
        )
        assert timing.execution_type == "scheduled"

        # Expected schedule operation
        schedule_op = ScheduleOperation(
            action_type="create",
            schedule_time="08:00",
            schedule_date="tomorrow",
            command_to_execute="open shades",
            schedule_description="Open shades tomorrow at 8am",
            reasoning="User specified future time for opening shades",
        )
        assert schedule_op.action_type == "create"
        assert schedule_op.schedule_time == "08:00"
        assert schedule_op.schedule_date == "tomorrow"


class TestDataStructureValidation:
    """Test comprehensive data structure validation"""

    def test_execution_timing_values(self):
        """Test ExecutionTiming enum values"""
        # Test current
        current = ExecutionTiming(
            execution_type="current", reasoning="Immediate execution"
        )
        assert current.execution_type == "current"

        # Test scheduled
        scheduled = ExecutionTiming(
            execution_type="scheduled", reasoning="Future execution"
        )
        assert scheduled.execution_type == "scheduled"

    def test_house_wide_detection_values(self):
        """Test HouseWideDetection boolean values"""
        house_wide = HouseWideDetection(is_house_wide=True)
        assert house_wide.is_house_wide == True

        room_specific = HouseWideDetection(is_house_wide=False)
        assert room_specific.is_house_wide == False

    def test_schedule_operation_types(self):
        """Test ScheduleOperation action types"""
        for action_type in ["create", "modify", "delete"]:
            operation = ScheduleOperation(
                action_type=action_type,
                command_to_execute="test command",
                schedule_description=f"Test {action_type} operation",
                reasoning=f"Testing {action_type} action type",
            )
            assert operation.action_type == action_type

    def test_shade_analysis_scopes(self):
        """Test ShadeAnalysis scope values"""
        for scope in ["specific", "room", "house"]:
            analysis = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=["test"], position=50, reasoning="Test operation"
                    )
                ],
                scope=scope,
                reasoning=f"Test {scope} scope",
            )
            assert analysis.scope == scope

    def test_blind_operation_position_validation(self):
        """Test BlindOperation position constraints"""
        # Valid positions
        for position in [0, 25, 50, 75, 100]:
            operation = BlindOperation(
                blind_filter=["test"],
                position=position,
                reasoning=f"Test position {position}%",
            )
            assert operation.position == position

        # Invalid positions should raise validation errors
        with pytest.raises(Exception):
            BlindOperation(
                blind_filter=["test"], position=-1, reasoning="Invalid position"
            )

        with pytest.raises(Exception):
            BlindOperation(
                blind_filter=["test"], position=101, reasoning="Invalid position"
            )


class TestIntegrationWorkflows:
    """Test complete workflow integration"""

    @pytest.fixture
    def sample_blinds(self):
        """Sample blind configurations"""
        return [
            BlindConfig(id="1", name="Guest Side Window", orientation="North"),
            BlindConfig(id="2", name="Guest Front Window", orientation="East"),
        ]

    def test_current_execution_workflow(self, sample_blinds):
        """Test complete current execution workflow"""
        # 1. House-wide detection (room-specific)
        house_wide = HouseWideDetection(is_house_wide=False)

        # 2. Execution timing (current)
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

            # Validate complete workflow
            assert not house_wide.is_house_wide
            assert timing.execution_type == "current"
            assert shade_analysis.scope == "specific"
            assert len(shade_analysis.operations) == 1
            assert shade_analysis.operations[0].position == 50

    def test_scheduled_execution_workflow(self):
        """Test complete scheduled execution workflow"""
        # 1. House-wide detection (room-specific)
        house_wide = HouseWideDetection(is_house_wide=False)

        # 2. Execution timing (scheduled)
        timing = ExecutionTiming(
            execution_type="scheduled", reasoning="Future time reference found"
        )

        # 3. Schedule management (if scheduled execution)
        if timing.execution_type == "scheduled":
            schedule_op = ScheduleOperation(
                action_type="create",
                schedule_time="21:00",
                schedule_date="daily",
                recurrence="daily",
                command_to_execute="close all blinds",
                schedule_description="Close blinds every day at 9pm",
                reasoning="User wants recurring evening blind closure",
            )

            # Validate complete workflow
            assert not house_wide.is_house_wide
            assert timing.execution_type == "scheduled"
            assert schedule_op.action_type == "create"
            assert schedule_op.schedule_time == "21:00"
            assert schedule_op.recurrence == "daily"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
