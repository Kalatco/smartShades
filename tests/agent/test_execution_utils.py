"""
Tests for ExecutionUtils
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from utils.agent.smart_shades.execution_utils import ExecutionUtils
from models.config import (
    HubitatConfig,
    RoomConfig,
    BlindConfig,
    LocationConfig,
    HouseInformationConfig,
)
from models.agent import ShadeAnalysis, ExecutionResult, BlindOperation


class TestExecutionUtils:
    """Test cases for ExecutionUtils"""

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
                "living_room": RoomConfig(
                    blinds=[
                        BlindConfig(id="1", name="main_blind"),
                        BlindConfig(id="2", name="side_blind"),
                    ]
                ),
                "bedroom": RoomConfig(
                    blinds=[BlindConfig(id="3", name="bedroom_blind")]
                ),
            },
        )

    @pytest.fixture
    def mock_shade_analysis_chain(self):
        """Create a mock shade analysis chain"""
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock()
        return mock_chain

    def test_validate_room_valid(self, mock_config):
        """Test room validation with valid room"""
        assert ExecutionUtils.validate_room(mock_config, "living_room") is True
        assert ExecutionUtils.validate_room(mock_config, "bedroom") is True

    def test_validate_room_invalid(self, mock_config):
        """Test room validation with invalid room"""
        assert ExecutionUtils.validate_room(mock_config, "kitchen") is False
        assert ExecutionUtils.validate_room(mock_config, "") is False
        assert ExecutionUtils.validate_room(mock_config, None) is False

    def test_get_room_blinds_valid(self, mock_config):
        """Test getting blinds for valid room"""
        blinds = ExecutionUtils.get_room_blinds(mock_config, "living_room")
        assert len(blinds) == 2
        assert blinds[0].name == "main_blind"
        assert blinds[1].name == "side_blind"

    def test_get_room_blinds_invalid(self, mock_config):
        """Test getting blinds for invalid room"""
        blinds = ExecutionUtils.get_room_blinds(mock_config, "kitchen")
        assert blinds == []

    @pytest.mark.asyncio
    async def test_get_room_current_positions_success(self, mock_config):
        """Test getting current positions successfully"""
        with patch(
            "utils.agent.smart_shades.execution_utils.HubitatUtils.get_room_current_positions"
        ) as mock_get_positions:
            mock_get_positions.return_value = {"main_blind": 75, "side_blind": 50}

            positions = await ExecutionUtils.get_room_current_positions(
                mock_config, "living_room"
            )

            assert positions == {"main_blind": 75, "side_blind": 50}
            mock_get_positions.assert_called_once_with(mock_config, "living_room")

    @pytest.mark.asyncio
    async def test_get_room_current_positions_error_fallback(self, mock_config):
        """Test getting current positions with error fallback"""
        with patch(
            "utils.agent.smart_shades.execution_utils.HubitatUtils.get_room_current_positions"
        ) as mock_get_positions:
            mock_get_positions.side_effect = Exception("Connection error")

            positions = await ExecutionUtils.get_room_current_positions(
                mock_config, "living_room"
            )

            # Should return fallback positions (50 for each blind)
            assert positions == {"main_blind": 50, "side_blind": 50}

    @pytest.mark.asyncio
    async def test_analyze_request_success(
        self, mock_config, mock_shade_analysis_chain
    ):
        """Test successful request analysis"""
        # Mock the chain response
        mock_analysis = ShadeAnalysis(
            scope="room", reasoning="Set blinds to 75% for optimal light"
        )
        mock_shade_analysis_chain.ainvoke.return_value = mock_analysis

        with patch(
            "utils.agent.smart_shades.execution_utils.HubitatUtils.get_room_current_positions"
        ) as mock_get_positions, patch(
            "utils.agent.smart_shades.execution_utils.SolarUtils.get_window_sun_exposure"
        ) as mock_solar:

            mock_get_positions.return_value = {"main_blind": 50, "side_blind": 60}
            mock_solar.return_value = {"sun_angle": 45, "exposure": "direct"}

            result = await ExecutionUtils.analyze_request(
                mock_shade_analysis_chain,
                mock_config,
                "close blinds halfway",
                "living_room",
                False,
            )

            assert result.scope == "room"
            assert result.reasoning == "Set blinds to 75% for optimal light"

            # Verify chain was called with correct parameters
            call_args = mock_shade_analysis_chain.ainvoke.call_args[0][0]
            assert call_args["command"] == "close blinds halfway"
            assert call_args["room"] == "living_room"
            assert "main_blind" in [blind.name for blind in call_args["room_blinds"]]

    @pytest.mark.asyncio
    async def test_analyze_request_house_wide(
        self, mock_config, mock_shade_analysis_chain
    ):
        """Test request analysis with house-wide command"""
        mock_analysis = ShadeAnalysis(
            scope="house", reasoning="Close all blinds house-wide"
        )
        mock_shade_analysis_chain.ainvoke.return_value = mock_analysis

        with patch(
            "utils.agent.smart_shades.execution_utils.HubitatUtils.get_room_current_positions"
        ) as mock_get_positions, patch(
            "utils.agent.smart_shades.execution_utils.SolarUtils.get_window_sun_exposure"
        ) as mock_solar:

            mock_get_positions.return_value = {"main_blind": 50}
            mock_solar.return_value = {"sun_angle": 45}

            result = await ExecutionUtils.analyze_request(
                mock_shade_analysis_chain,
                mock_config,
                "close all blinds",
                "living_room",
                True,  # is_house_wide
            )

            # Verify the command was enhanced with house-wide prefix
            call_args = mock_shade_analysis_chain.ainvoke.call_args[0][0]
            assert call_args["command"] == "[HOUSE-WIDE COMMAND] close all blinds"

    @pytest.mark.asyncio
    async def test_analyze_request_chain_error(
        self, mock_config, mock_shade_analysis_chain
    ):
        """Test request analysis with chain error"""
        mock_shade_analysis_chain.ainvoke.side_effect = Exception("Chain error")

        with patch(
            "utils.agent.smart_shades.execution_utils.HubitatUtils.get_room_current_positions"
        ) as mock_get_positions, patch(
            "utils.agent.smart_shades.execution_utils.SolarUtils.get_window_sun_exposure"
        ) as mock_solar:

            mock_get_positions.return_value = {"main_blind": 50}
            mock_solar.return_value = {"sun_angle": 45}

            result = await ExecutionUtils.analyze_request(
                mock_shade_analysis_chain,
                mock_config,
                "test command",
                "living_room",
                False,
            )

            # Should return fallback analysis
            assert result.scope == "room"
            assert "Fallback parsing due to error: Chain error" in result.reasoning

    @pytest.mark.asyncio
    async def test_execute_action_invalid_room(self, mock_config):
        """Test execution with invalid room"""
        analysis = ShadeAnalysis(
            position=50, scope="room", blind_filter=[], reasoning="Test analysis"
        )

        result = await ExecutionUtils.execute_action(
            mock_config, analysis, "invalid_room"
        )

        assert result.executed_blinds == []
        assert result.total_blinds == 0
        assert result.position == 0
        assert result.scope == "error"
        assert "Invalid room: invalid_room" in result.reasoning

    @pytest.mark.asyncio
    async def test_execute_action_success(self, mock_config):
        """Test successful action execution"""
        # Create analysis with operations
        analysis = ShadeAnalysis(
            scope="room",
            reasoning="Set blinds to 75%",
            operations=[
                BlindOperation(
                    blind_filter=["main_blind"],
                    position=75,
                    reasoning="Close main blind to 75%",
                )
            ],
        )

        with patch(
            "utils.agent.smart_shades.execution_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds, patch(
            "utils.agent.smart_shades.execution_utils.HubitatUtils.control_blinds"
        ) as mock_control:

            # Mock getting target blinds
            mock_blind = MagicMock()
            mock_blind.name = "main_blind"
            mock_get_blinds.return_value = ([mock_blind], ["living_room"])

            result = await ExecutionUtils.execute_action(
                mock_config, analysis, "living_room"
            )

            assert result.executed_blinds == ["main_blind"]
            assert result.total_blinds == 1
            assert result.position == 75
            assert result.scope == "room"
            assert result.reasoning == "Set blinds to 75%"

            # Verify control was called
            mock_control.assert_called_once_with(mock_config, [mock_blind], 75)

    @pytest.mark.asyncio
    async def test_execute_action_no_operations(self, mock_config):
        """Test execution with no operations"""
        analysis = ShadeAnalysis(
            scope="room", reasoning="Test analysis", operations=[]  # No operations
        )

        result = await ExecutionUtils.execute_action(
            mock_config, analysis, "living_room"
        )

        assert result.executed_blinds == []
        assert result.total_blinds == 0
        assert result.position == 0
        assert result.scope == "room"
        assert result.reasoning == "No operations to execute"

    @pytest.mark.asyncio
    async def test_execute_action_no_matching_blinds(self, mock_config):
        """Test execution with no matching blinds"""
        analysis = ShadeAnalysis(
            scope="room",
            reasoning="Test analysis",
            operations=[
                BlindOperation(
                    blind_filter=["nonexistent_blind"],
                    position=75,
                    reasoning="Test operation",
                )
            ],
        )

        with patch(
            "utils.agent.smart_shades.execution_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds:
            # Return empty list for no matching blinds
            mock_get_blinds.return_value = ([], [])

            result = await ExecutionUtils.execute_action(
                mock_config, analysis, "living_room"
            )

            assert result.executed_blinds == []
            assert result.total_blinds == 0
            assert result.position == 0  # Should be 0 when no operations executed
            assert result.reasoning == "No matching blinds found"

    @pytest.mark.asyncio
    async def test_process_current_execution_success(
        self, mock_config, mock_shade_analysis_chain
    ):
        """Test successful current execution processing"""
        # Mock house-wide chain
        mock_house_wide_chain = AsyncMock()
        mock_house_wide_result = MagicMock()
        mock_house_wide_result.is_house_wide = False
        mock_house_wide_chain.ainvoke.return_value = mock_house_wide_result

        # Mock analysis result
        mock_analysis = ShadeAnalysis(
            scope="room",
            reasoning="Test execution",
            operations=[
                BlindOperation(
                    blind_filter=["main_blind"], position=50, reasoning="Test operation"
                )
            ],
        )

        with patch(
            "utils.agent.smart_shades.execution_utils.ExecutionUtils.analyze_request"
        ) as mock_analyze, patch(
            "utils.agent.smart_shades.execution_utils.ExecutionUtils.execute_action"
        ) as mock_execute, patch(
            "utils.agent.smart_shades.agent_response_utils.AgentResponseUtils.build_response_from_execution"
        ) as mock_build_response:

            mock_analyze.return_value = mock_analysis
            mock_execute.return_value = ExecutionResult(
                executed_blinds=["main_blind"],
                affected_rooms=["living_room"],
                total_blinds=1,
                position=50,
                scope="room",
                reasoning="Test execution",
            )
            mock_build_response.return_value = {"status": "success"}

            result = await ExecutionUtils.process_current_execution(
                mock_shade_analysis_chain,
                mock_house_wide_chain,
                mock_config,
                "test command",
                "living_room",
            )

            assert result == {"status": "success"}
            mock_house_wide_chain.ainvoke.assert_called_once_with(
                {"command": "test command"}
            )
            mock_analyze.assert_called_once()
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_current_execution_error(
        self, mock_config, mock_shade_analysis_chain
    ):
        """Test current execution processing with error"""
        # Mock house-wide chain to raise exception
        mock_house_wide_chain = AsyncMock()
        mock_house_wide_chain.ainvoke.side_effect = Exception("Chain error")

        with patch(
            "utils.agent.smart_shades.agent_response_utils.AgentResponseUtils.create_error_response"
        ) as mock_error_response:
            mock_error_response.return_value = {"error": "Chain error"}

            result = await ExecutionUtils.process_current_execution(
                mock_shade_analysis_chain,
                mock_house_wide_chain,
                mock_config,
                "test command",
                "living_room",
            )

            assert result == {"error": "Chain error"}
            mock_error_response.assert_called_once()
