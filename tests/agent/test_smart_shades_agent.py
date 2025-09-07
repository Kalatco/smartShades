"""
Tests for SmartShadesAgent
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent.smart_shades_agent import SmartShadesAgent
from models.config import (
    HubitatConfig,
    RoomConfig,
    BlindConfig,
    LocationConfig,
    HouseInformationConfig,
)
from models.agent import (
    ShadeAnalysis,
    ExecutionResult,
    BlindOperation,
    ExecutionTiming,
    ScheduleOperation,
    HouseWideDetection,
)


class TestSmartShadesAgent:
    """Test cases for SmartShadesAgent"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return HubitatConfig(
            accessToken="test-token",
            hubitatUrl="http://test.hub",
            makerApiId="123",
            location=LocationConfig(city="Seattle", timezone="America/Los_Angeles"),
            houseInformation=HouseInformationConfig(orientation="north-south"),
            rooms={
                "test_room": RoomConfig(
                    blinds=[
                        BlindConfig(id="1", name="Test Blind 1", orientation="North"),
                        BlindConfig(id="2", name="Test Blind 2", orientation="South"),
                    ]
                )
            },
        )

    @pytest.fixture
    def agent(self):
        """Create a SmartShadesAgent instance for testing"""
        return SmartShadesAgent()

    @pytest.mark.asyncio
    async def test_agent_initialization_success(self, agent):
        """Test successful agent initialization"""
        with patch("utils.config_utils.ConfigManager.load_environment"), patch(
            "utils.config_utils.ConfigManager.validate_environment", return_value=True
        ), patch(
            "utils.config_utils.ConfigManager.load_blinds_config"
        ) as mock_load_config, patch(
            "utils.config_utils.ConfigManager.override_hubitat_config"
        ) as mock_override, patch(
            "utils.config_utils.ConfigManager.create_azure_llm"
        ) as mock_create_llm, patch(
            "utils.config_utils.ConfigManager.get_config_summary", return_value={}
        ), patch(
            "agent.smart_shades_agent.SmartScheduler"
        ) as mock_scheduler_class:

            # Setup mocks
            mock_config = Mock()
            mock_config.rooms = {"test_room": Mock()}
            mock_load_config.return_value = mock_config
            mock_override.return_value = mock_config
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm

            mock_scheduler = AsyncMock()
            mock_scheduler.start = AsyncMock()
            mock_scheduler.set_config = Mock()
            mock_scheduler_class.return_value = mock_scheduler

            # Initialize agent
            await agent.initialize()

            # Verify initialization
            assert agent.config == mock_config
            assert agent.llm == mock_llm
            assert agent.house_wide_chain is not None
            assert agent.shade_analysis_chain is not None
            assert agent.execution_timing_chain is not None
            assert agent.schedule_management_chain is not None
            assert agent.scheduler == mock_scheduler
            mock_scheduler.start.assert_called_once()
            mock_scheduler.set_config.assert_called_once_with(
                mock_config
            )  # Verify scheduler was started
            mock_scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_initialization_env_validation_failure(self, agent):
        """Test agent initialization with environment validation failure"""
        with patch("utils.config_utils.ConfigManager.load_environment"), patch(
            "utils.config_utils.ConfigManager.validate_environment", return_value=False
        ):

            with pytest.raises(
                ValueError, match="Required environment variables are not set"
            ):
                await agent.initialize()

    @pytest.mark.asyncio
    async def test_agent_shutdown(self, agent):
        """Test agent shutdown"""

        # Create a simple test scheduler that tracks calls
        class TestScheduler:
            def __init__(self):
                self.shutdown_called = False

            async def shutdown(self):
                self.shutdown_called = True

        test_scheduler = TestScheduler()
        agent.scheduler = test_scheduler

        # Call shutdown
        await agent.shutdown()

        # Verify shutdown was called
        assert (
            test_scheduler.shutdown_called
        ), "Scheduler shutdown method was not called"

    def test_get_schedules_with_scheduler(self, agent):
        """Test getting schedules when scheduler is available"""
        mock_scheduler = Mock()
        mock_schedules = [{"id": "1", "description": "Test schedule"}]
        mock_scheduler.get_schedules.return_value = mock_schedules
        agent.scheduler = mock_scheduler

        result = agent.get_schedules("test_room")

        assert result == mock_schedules
        mock_scheduler.get_schedules.assert_called_once_with("test_room")

    def test_get_schedules_without_scheduler(self, agent):
        """Test getting schedules when scheduler is not available"""
        agent.scheduler = None

        result = agent.get_schedules("test_room")

        assert result == []

    @pytest.mark.asyncio
    async def test_process_request_invalid_room(self, agent, mock_config):
        """Test processing request with invalid room"""
        agent.config = mock_config

        result = await agent.process_request("test command", "invalid_room")

        assert result["position"] == 0
        assert "Invalid room" in result["message"]
        assert result["room"] == "invalid_room"

    @pytest.mark.asyncio
    async def test_process_current_execution_flow(self, agent, mock_config):
        """Test current execution flow"""
        agent.config = mock_config

        # Setup mocks
        mock_timing_chain = AsyncMock()
        mock_timing_chain.ainvoke.return_value = ExecutionTiming(
            execution_type="current", reasoning="Immediate execution"
        )
        agent.execution_timing_chain = mock_timing_chain

        mock_house_wide_chain = AsyncMock()
        mock_house_wide_chain.ainvoke.return_value = HouseWideDetection(
            is_house_wide=False, reasoning="Room specific"
        )
        agent.house_wide_chain = mock_house_wide_chain

        with patch.object(agent, "_analyze_request") as mock_analyze, patch.object(
            agent, "_execute_action"
        ) as mock_execute, patch.object(
            agent, "_build_response_from_execution"
        ) as mock_build_response:

            mock_analysis = ShadeAnalysis(
                operations=[
                    BlindOperation(blind_filter=["test"], position=50, reasoning="Test")
                ],
                scope="room",
                reasoning="Test analysis",
            )
            mock_analyze.return_value = mock_analysis

            mock_execution = ExecutionResult(
                executed_blinds=["Test Blind 1"],
                affected_rooms=["test_room"],
                total_blinds=1,
                position=50,
                scope="room",
                reasoning="Test execution",
            )
            mock_execute.return_value = mock_execution

            mock_response = {"message": "Success", "position": 50}
            mock_build_response.return_value = mock_response

            result = await agent.process_request("close blinds", "test_room")

            assert result == mock_response
            mock_analyze.assert_called_once_with("close blinds", "test_room", False)
            mock_execute.assert_called_once_with(mock_analysis, "test_room")

    @pytest.mark.asyncio
    async def test_process_scheduled_execution_flow(self, agent, mock_config):
        """Test scheduled execution flow"""
        agent.config = mock_config

        # Setup mocks
        mock_timing_chain = AsyncMock()
        mock_timing_chain.ainvoke.return_value = ExecutionTiming(
            execution_type="scheduled", reasoning="Future execution"
        )
        agent.execution_timing_chain = mock_timing_chain

        mock_schedule_chain = AsyncMock()
        mock_schedule_chain.ainvoke.return_value = ScheduleOperation(
            action_type="create",
            schedule_time="9pm",
            command_to_execute="close blinds",
            schedule_description="Close blinds at 9pm",
            reasoning="Evening schedule",
        )
        agent.schedule_management_chain = mock_schedule_chain

        mock_scheduler = AsyncMock()
        mock_scheduler.get_schedules.return_value = []
        mock_scheduler.create_schedule.return_value = {
            "success": True,
            "job_id": "job_123",
            "next_run": "2025-09-06 21:00:00",
        }
        agent.scheduler = mock_scheduler

        result = await agent.process_request("close blinds at 9pm", "test_room")

        assert result["operation"] == "create"
        assert result["schedule_id"] == "job_123"
        assert "Schedule created successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_analyze_request_with_solar_info(self, agent, mock_config):
        """Test request analysis with solar information"""
        agent.config = mock_config
        agent.shade_analysis_chain = AsyncMock()

        mock_analysis = ShadeAnalysis(
            operations=[
                BlindOperation(
                    blind_filter=["north"], position=0, reasoning="Block sun"
                )
            ],
            scope="specific",
            reasoning="Solar analysis",
        )
        agent.shade_analysis_chain.ainvoke.return_value = mock_analysis

        with patch(
            "utils.hubitat_utils.HubitatUtils.get_room_current_positions"
        ) as mock_positions, patch(
            "utils.solar.SolarUtils.get_window_sun_exposure"
        ) as mock_solar:

            mock_positions.return_value = {"Test Blind 1": 50, "Test Blind 2": 75}
            mock_solar.return_value = {
                "Test Blind 1": {"is_sunny": True, "sun_intensity": "high"}
            }

            result = await agent._analyze_request("block the sun", "test_room", False)

            assert result == mock_analysis

            # Verify the chain was called with proper inputs
            call_args = agent.shade_analysis_chain.ainvoke.call_args[0][0]
            assert call_args["command"] == "block the sun"
            assert call_args["room"] == "test_room"
            assert len(call_args["room_blinds"]) == 2
            assert call_args["current_positions"] == {
                "Test Blind 1": 50,
                "Test Blind 2": 75,
            }

    @pytest.mark.asyncio
    async def test_execute_action_success(self, agent, mock_config):
        """Test successful action execution"""
        agent.config = mock_config

        analysis = ShadeAnalysis(
            operations=[
                BlindOperation(
                    blind_filter=["Test"], position=25, reasoning="Quarter close"
                )
            ],
            scope="specific",
            reasoning="Test execution",
        )

        with patch(
            "utils.blind_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds, patch(
            "utils.hubitat_utils.HubitatUtils.control_blinds"
        ) as mock_control:

            mock_blinds = [
                BlindConfig(id="1", name="Test Blind 1", orientation="North")
            ]
            mock_get_blinds.return_value = (mock_blinds, ["test_room"])
            mock_control.return_value = None

            result = await agent._execute_action(analysis, "test_room")

            assert result.executed_blinds == ["Test Blind 1"]
            assert result.affected_rooms == ["test_room"]
            assert result.total_blinds == 1
            assert result.position == 25
            assert result.scope == "specific"

    @pytest.mark.asyncio
    async def test_execute_action_no_matching_blinds(self, agent, mock_config):
        """Test action execution with no matching blinds"""
        agent.config = mock_config

        analysis = ShadeAnalysis(
            operations=[
                BlindOperation(
                    blind_filter=["nonexistent"], position=50, reasoning="No match"
                )
            ],
            scope="specific",
            reasoning="Test",
        )

        with patch(
            "utils.blind_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds:
            mock_get_blinds.return_value = ([], [])

            result = await agent._execute_action(analysis, "test_room")

            assert result.executed_blinds == []
            assert result.total_blinds == 0
            assert "No matching blinds found" in result.reasoning

    @pytest.mark.asyncio
    async def test_get_current_status_success(self, agent, mock_config):
        """Test getting current status successfully"""
        agent.config = mock_config

        with patch(
            "utils.hubitat_utils.HubitatUtils.get_room_current_positions"
        ) as mock_positions:
            mock_positions.return_value = {"Test Blind 1": 30, "Test Blind 2": 70}

            result = await agent.get_current_status("test_room")

            assert result["position"] == 50  # Average of 30 and 70
            assert result["room"] == "test_room"
            assert "Test Blind 1: 30%" in result["message"]
            assert "Test Blind 2: 70%" in result["message"]
            assert result["affected_blinds"] == ["Test Blind 1", "Test Blind 2"]

    @pytest.mark.asyncio
    async def test_get_current_status_hubitat_error(self, agent, mock_config):
        """Test getting current status with Hubitat error"""
        agent.config = mock_config

        with patch(
            "utils.hubitat_utils.HubitatUtils.get_room_current_positions"
        ) as mock_positions:
            mock_positions.side_effect = Exception("Hubitat connection error")

            result = await agent.get_current_status("test_room")

            assert result["position"] == 50  # Fallback value
            assert result["room"] == "test_room"
            assert "Could not retrieve current positions" in result["message"]

    def test_error_response(self, agent):
        """Test error response creation"""
        result = agent._error_response("Test error message", "test_room")

        assert result["position"] == 0
        assert result["message"] == "Test error message"
        assert result["room"] == "test_room"
        assert result["affected_blinds"] == []
        assert isinstance(result["timestamp"], datetime)

    def test_build_response_from_execution(self, agent):
        """Test response building from execution result"""
        execution_result = ExecutionResult(
            executed_blinds=["Blind 1", "Blind 2"],
            affected_rooms=["test_room"],
            total_blinds=2,
            position=75,
            scope="room",
            reasoning="Test execution successful",
        )

        result = agent._build_response_from_execution(execution_result, "test_room")

        assert result["position"] == 75
        assert result["room"] == "test_room"
        assert result["affected_blinds"] == ["Blind 1", "Blind 2"]
        assert "Test execution successful" in result["message"]
        assert "Blind 1, Blind 2" in result["message"]
