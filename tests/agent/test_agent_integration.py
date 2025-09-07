"""
Integration tests for SmartShadesAgent
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from agent.smart_shades_agent import SmartShadesAgent
from models.config import (
    HubitatConfig,
    RoomConfig,
    BlindConfig,
    HouseInformationConfig,
    LocationConfig,
)
from models.agent import (
    ShadeAnalysis,
    ExecutionResult,
    BlindOperation,
    ExecutionTiming,
    ScheduleOperation,
    HouseWideDetection,
)


class TestSmartShadesAgentIntegration:
    """Integration tests for SmartShadesAgent with realistic scenarios"""

    @pytest.fixture
    def full_config(self):
        """Create a comprehensive configuration for testing"""
        return HubitatConfig(
            accessToken="test-token-123",
            hubitatUrl="http://192.168.1.100",
            makerApiId="21",
            rooms={
                "living_room": RoomConfig(
                    blinds=[
                        BlindConfig(
                            id="5", name="Living Room Front Window", orientation="South"
                        ),
                        BlindConfig(
                            id="6", name="Living Room Side Window", orientation="East"
                        ),
                    ]
                ),
                "bedroom": RoomConfig(
                    blinds=[
                        BlindConfig(id="7", name="Bedroom Window", orientation="North"),
                    ]
                ),
                "office": RoomConfig(
                    blinds=[
                        BlindConfig(
                            id="8", name="Office East Window", orientation="East"
                        ),
                        BlindConfig(
                            id="9", name="Office West Window", orientation="West"
                        ),
                    ]
                ),
            },
            houseInformation=HouseInformationConfig(
                orientation="north-south", notes="Smart home with automated blinds"
            ),
            location=LocationConfig(city="Seattle, WA", timezone="America/Los_Angeles"),
        )

    @pytest.fixture
    async def initialized_agent(self, full_config):
        """Create and initialize a SmartShadesAgent for testing"""
        agent = SmartShadesAgent()

        with patch("utils.config_utils.ConfigManager.load_environment"), patch(
            "utils.config_utils.ConfigManager.validate_environment", return_value=True
        ), patch(
            "utils.config_utils.ConfigManager.load_blinds_config",
            return_value=full_config,
        ), patch(
            "utils.config_utils.ConfigManager.override_hubitat_config",
            return_value=full_config,
        ), patch(
            "utils.config_utils.ConfigManager.create_azure_llm"
        ) as mock_create_llm, patch(
            "utils.config_utils.ConfigManager.get_config_summary", return_value={}
        ), patch(
            "utils.smart_scheduler.SmartScheduler"
        ) as mock_scheduler_class:

            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm

            mock_scheduler = AsyncMock()
            mock_scheduler_class.return_value = mock_scheduler

            await agent.initialize()

            yield agent

            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_user_case_1_stop_daily_schedule(self, initialized_agent):
        """Test Case 1: 'hey stop closing the blinds everyday'"""
        agent = initialized_agent

        # Mock the chain responses
        with patch.object(
            agent.execution_timing_chain, "ainvoke"
        ) as mock_timing, patch.object(
            agent.schedule_management_chain, "ainvoke"
        ) as mock_schedule, patch.object(
            agent.scheduler, "get_schedules"
        ) as mock_get_schedules, patch.object(
            agent.scheduler, "delete_schedule"
        ) as mock_delete_schedule:

            # Setup timing detection
            mock_timing.return_value = ExecutionTiming(
                execution_type="scheduled", reasoning="Managing existing schedule"
            )

            # Setup schedule operation
            mock_schedule.return_value = ScheduleOperation(
                action_type="delete",
                schedule_description="Cancel daily closing schedule",
                command_to_execute="",
                reasoning="User wants to stop recurring daily blind closing",
            )

            # Setup existing schedules
            mock_get_schedules.return_value = [
                {"id": "daily_close", "description": "Close blinds every day at 6pm"}
            ]

            # Setup delete operation
            mock_delete_schedule.return_value = {
                "success": True,
                "job_id": "daily_close",
            }

            result = await agent.process_request(
                "hey stop closing the blinds everyday", "living_room"
            )

            assert result["operation"] == "delete"
            assert "Schedule deleted successfully" in result["message"]
            mock_delete_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_case_2_specific_blind_percentage(self, initialized_agent):
        """Test Case 2: 'open the front shade around half of 100'"""
        agent = initialized_agent

        with patch.object(
            agent.execution_timing_chain, "ainvoke"
        ) as mock_timing, patch.object(
            agent.house_wide_chain, "ainvoke"
        ) as mock_house_wide, patch.object(
            agent.shade_analysis_chain, "ainvoke"
        ) as mock_analysis, patch(
            "utils.hubitat_utils.HubitatUtils.get_room_current_positions"
        ) as mock_positions, patch(
            "utils.solar.SolarUtils.get_window_sun_exposure"
        ) as mock_solar, patch(
            "utils.blind_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds, patch(
            "utils.hubitat_utils.HubitatUtils.control_blinds"
        ) as mock_control:

            # Setup timing detection
            mock_timing.return_value = ExecutionTiming(
                execution_type="current", reasoning="Immediate execution"
            )

            # Setup house-wide detection
            mock_house_wide.return_value = HouseWideDetection(
                is_house_wide=False, reasoning="Specific blind mentioned"
            )

            # Setup current positions and solar info
            mock_positions.return_value = {
                "Living Room Front Window": 0,
                "Living Room Side Window": 25,
            }
            mock_solar.return_value = {
                "Living Room Front Window": {"is_sunny": False, "sun_intensity": "none"}
            }

            # Setup analysis result
            mock_analysis.return_value = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=["front"],
                        position=50,
                        reasoning="Half of 100% = 50%, targeting front window",
                    )
                ],
                scope="specific",
                reasoning="Specific blind with calculated position",
            )

            # Setup blind targeting and control
            target_blind = BlindConfig(
                id="5", name="Living Room Front Window", orientation="South"
            )
            mock_get_blinds.return_value = ([target_blind], ["living_room"])
            mock_control.return_value = None

            result = await agent.process_request(
                "open the front shade around half of 100", "living_room"
            )

            assert result["position"] == 50
            assert "Living Room Front Window" in result["affected_blinds"]
            mock_control.assert_called_once_with(agent.config, [target_blind], 50)

    @pytest.mark.asyncio
    async def test_user_case_3_future_schedule(self, initialized_agent):
        """Test Case 3: 'can you open the shades tomorrow morning at 8am'"""
        agent = initialized_agent

        with patch.object(
            agent.execution_timing_chain, "ainvoke"
        ) as mock_timing, patch.object(
            agent.schedule_management_chain, "ainvoke"
        ) as mock_schedule, patch.object(
            agent.scheduler, "get_schedules"
        ) as mock_get_schedules, patch.object(
            agent.scheduler, "create_schedule"
        ) as mock_create_schedule:

            # Setup timing detection
            mock_timing.return_value = ExecutionTiming(
                execution_type="scheduled",
                reasoning="Future time reference - tomorrow morning",
            )

            # Setup schedule operation
            mock_schedule.return_value = ScheduleOperation(
                action_type="create",
                schedule_time="08:00",
                schedule_date="tomorrow",
                command_to_execute="open the shades",
                schedule_description="Open shades tomorrow morning at 8am",
                reasoning="Creating one-time schedule for tomorrow morning",
            )

            # Setup existing schedules
            mock_get_schedules.return_value = []

            # Setup create operation
            mock_create_schedule.return_value = {
                "success": True,
                "job_id": "tomorrow_8am_open",
                "next_run": "2025-09-07 08:00:00",
            }

            result = await agent.process_request(
                "can you open the shades tomorrow morning at 8am", "bedroom"
            )

            assert result["operation"] == "create"
            assert result["schedule_id"] == "tomorrow_8am_open"
            assert "2025-09-07 08:00:00" in result["next_run"]
            mock_create_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_house_wide_command(self, initialized_agent):
        """Test house-wide command: 'close all blinds'"""
        agent = initialized_agent

        with patch.object(
            agent.execution_timing_chain, "ainvoke"
        ) as mock_timing, patch.object(
            agent.house_wide_chain, "ainvoke"
        ) as mock_house_wide, patch.object(
            agent.shade_analysis_chain, "ainvoke"
        ) as mock_analysis, patch(
            "utils.hubitat_utils.HubitatUtils.get_room_current_positions"
        ) as mock_positions, patch(
            "utils.solar.SolarUtils.get_window_sun_exposure"
        ) as mock_solar, patch(
            "utils.blind_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds, patch(
            "utils.hubitat_utils.HubitatUtils.control_blinds"
        ) as mock_control:

            # Setup timing detection
            mock_timing.return_value = ExecutionTiming(
                execution_type="current", reasoning="Immediate execution"
            )

            # Setup house-wide detection
            mock_house_wide.return_value = HouseWideDetection(
                is_house_wide=True, reasoning="'all blinds' indicates house-wide scope"
            )

            # Setup current positions and solar info
            mock_positions.return_value = {
                "Living Room Front Window": 75,
                "Living Room Side Window": 50,
            }
            mock_solar.return_value = {}

            # Setup analysis result
            mock_analysis.return_value = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=[],
                        position=0,
                        reasoning="Close all blinds in the house",
                    )
                ],
                scope="house",
                reasoning="House-wide operation to close all blinds",
            )

            # Setup blind targeting and control
            all_blinds = [
                BlindConfig(
                    id="5", name="Living Room Front Window", orientation="South"
                ),
                BlindConfig(id="6", name="Living Room Side Window", orientation="East"),
            ]
            mock_get_blinds.return_value = (all_blinds, ["living_room"])
            mock_control.return_value = None

            result = await agent.process_request("close all blinds", "living_room")

            assert result["position"] == 0
            assert len(result["affected_blinds"]) == 2
            mock_control.assert_called_once_with(agent.config, all_blinds, 0)

    @pytest.mark.asyncio
    async def test_solar_aware_command(self, initialized_agent):
        """Test solar-aware command: 'block the sun'"""
        agent = initialized_agent

        with patch.object(
            agent.execution_timing_chain, "ainvoke"
        ) as mock_timing, patch.object(
            agent.house_wide_chain, "ainvoke"
        ) as mock_house_wide, patch.object(
            agent.shade_analysis_chain, "ainvoke"
        ) as mock_analysis, patch(
            "utils.hubitat_utils.HubitatUtils.get_room_current_positions"
        ) as mock_positions, patch(
            "utils.solar.SolarUtils.get_window_sun_exposure"
        ) as mock_solar, patch(
            "utils.blind_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds, patch(
            "utils.hubitat_utils.HubitatUtils.control_blinds"
        ) as mock_control:

            # Setup timing detection
            mock_timing.return_value = ExecutionTiming(
                execution_type="current", reasoning="Immediate sun blocking needed"
            )

            # Setup house-wide detection
            mock_house_wide.return_value = HouseWideDetection(
                is_house_wide=False, reasoning="Context suggests room-specific action"
            )

            # Setup current positions
            mock_positions.return_value = {
                "Office East Window": 100,
                "Office West Window": 75,
            }

            # Setup solar info - east window is sunny
            mock_solar.return_value = {
                "Office East Window": {"is_sunny": True, "sun_intensity": "high"},
                "Office West Window": {"is_sunny": False, "sun_intensity": "none"},
            }

            # Setup analysis result - should target sunny window
            mock_analysis.return_value = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=["east"],
                        position=0,
                        reasoning="Block sun on east-facing window with high intensity",
                    )
                ],
                scope="specific",
                reasoning="Solar-aware operation targeting sunny windows",
            )

            # Setup blind targeting
            target_blind = BlindConfig(
                id="8", name="Office East Window", orientation="East"
            )
            mock_get_blinds.return_value = ([target_blind], ["office"])
            mock_control.return_value = None

            result = await agent.process_request("block the sun", "office")

            assert result["position"] == 0
            assert "Office East Window" in result["affected_blinds"]

            # Verify that solar information was provided to the analysis
            call_args = mock_analysis.call_args[0][0]
            assert "window_sun_info" in call_args
            assert (
                call_args["window_sun_info"]["Office East Window"]["is_sunny"] is True
            )

    @pytest.mark.asyncio
    async def test_error_handling_chain_failure(self, initialized_agent):
        """Test error handling when chain operations fail"""
        agent = initialized_agent

        with patch.object(agent.execution_timing_chain, "ainvoke") as mock_timing:
            # Simulate chain failure
            mock_timing.side_effect = Exception("LLM service unavailable")

            result = await agent.process_request("close blinds", "living_room")

            assert result["position"] == 0
            assert "Error processing command" in result["message"]
            assert result["room"] == "living_room"

    @pytest.mark.asyncio
    async def test_error_handling_hubitat_failure(self, initialized_agent):
        """Test error handling when Hubitat operations fail"""
        agent = initialized_agent

        with patch.object(
            agent.execution_timing_chain, "ainvoke"
        ) as mock_timing, patch.object(
            agent.house_wide_chain, "ainvoke"
        ) as mock_house_wide, patch.object(
            agent.shade_analysis_chain, "ainvoke"
        ) as mock_analysis, patch(
            "utils.hubitat_utils.HubitatUtils.get_room_current_positions"
        ) as mock_positions, patch(
            "utils.solar.SolarUtils.get_window_sun_exposure"
        ) as mock_solar, patch(
            "utils.blind_utils.BlindUtils.get_target_blinds_for_operation"
        ) as mock_get_blinds, patch(
            "utils.hubitat_utils.HubitatUtils.control_blinds"
        ) as mock_control:

            # Setup normal chain responses
            mock_timing.return_value = ExecutionTiming(
                execution_type="current", reasoning="Test"
            )
            mock_house_wide.return_value = HouseWideDetection(
                is_house_wide=False, reasoning="Test"
            )
            mock_positions.return_value = {"Test Blind": 50}
            mock_solar.return_value = {}
            mock_analysis.return_value = ShadeAnalysis(
                operations=[
                    BlindOperation(blind_filter=["test"], position=25, reasoning="Test")
                ],
                scope="room",
                reasoning="Test",
            )

            target_blind = BlindConfig(id="5", name="Test Blind", orientation="South")
            mock_get_blinds.return_value = ([target_blind], ["living_room"])

            # Simulate Hubitat failure
            mock_control.side_effect = Exception("Hubitat connection timeout")

            result = await agent.process_request("close blinds", "living_room")

            assert result["position"] == 0
            assert "Error controlling blinds" in result["message"]
