"""
Tests for AgentResponseUtils
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock, PropertyMock

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from utils.agent_response_utils import AgentResponseUtils
from models.config import (
    HubitatConfig,
    RoomConfig,
    BlindConfig,
    LocationConfig,
    HouseInformationConfig,
)
from models.agent import ExecutionResult


class TestAgentResponseUtils:
    """Test cases for AgentResponseUtils"""

    def test_create_error_response(self):
        """Test error response creation"""
        message = "Test error"
        room = "living_room"

        result = AgentResponseUtils.create_error_response(message, room)

        assert result["position"] == 0
        assert result["message"] == message
        assert result["room"] == room
        assert result["affected_blinds"] == []
        assert isinstance(result["timestamp"], datetime)

    def test_build_response_from_execution(self):
        """Test building response from execution result"""
        execution_result = ExecutionResult(
            executed_blinds=["blind1", "blind2"],
            position=75,
            scope="room",
            reasoning="Successfully adjusted blinds for optimal lighting",
        )
        room = "bedroom"

        result = AgentResponseUtils.build_response_from_execution(
            execution_result, room
        )

        assert result["position"] == 75
        assert result["room"] == room
        assert result["affected_blinds"] == ["blind1", "blind2"]
        assert "Successfully adjusted blinds for optimal lighting" in result["message"]
        assert "blind1, blind2" in result["message"]
        assert isinstance(result["timestamp"], datetime)

    def test_build_response_from_execution_with_exception(self):
        """Test building response from execution result that raises exception"""
        # Create a mock execution result that will cause an exception
        execution_result = ExecutionResult(
            executed_blinds=["test"],  # Valid list to avoid validation error
            position=50,
            scope="room",
            reasoning="Test",
        )
        room = "test_room"

        # Mock the execution_result.executed_blinds to be None after creation
        with patch.object(execution_result, "executed_blinds", None):
            with patch("utils.agent_response_utils.logger") as mock_logger:
                result = AgentResponseUtils.build_response_from_execution(
                    execution_result, room
                )

                # Should return error response when exception occurs
                assert result["position"] == 0
                assert result["message"] == "Position updated successfully"
                assert result["room"] == room
                assert result["affected_blinds"] == []
                mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_status_invalid_room(self):
        """Test getting status for invalid room"""
        config = HubitatConfig(
            accessToken="test-token",
            hubitatUrl="http://test.hub",
            makerApiId="123",
            location=LocationConfig(city="Test", timezone="UTC"),
            houseInformation=HouseInformationConfig(orientation="north-south"),
            rooms={},
        )

        result = await AgentResponseUtils.get_current_status(config, "invalid_room")

        assert result["position"] == 0
        assert "Invalid room: invalid_room" in result["message"]
        assert result["room"] == "invalid_room"
        assert result["affected_blinds"] == []

    @pytest.mark.asyncio
    async def test_get_current_status_success(self):
        """Test getting current status successfully"""
        config = HubitatConfig(
            accessToken="test-token",
            hubitatUrl="http://test.hub",
            makerApiId="123",
            location=LocationConfig(city="Test", timezone="UTC"),
            houseInformation=HouseInformationConfig(orientation="north-south"),
            rooms={
                "living_room": RoomConfig(
                    blinds=[
                        BlindConfig(id="1", name="blind1"),
                        BlindConfig(id="2", name="blind2"),
                    ]
                )
            },
        )

        with patch(
            "utils.agent_response_utils.HubitatUtils.get_room_current_positions"
        ) as mock_get_positions:
            mock_get_positions.return_value = {"blind1": 60, "blind2": 80}

            result = await AgentResponseUtils.get_current_status(config, "living_room")

            assert result["position"] == 70  # Average of 60 and 80
            assert result["room"] == "living_room"
            assert result["affected_blinds"] == ["blind1", "blind2"]
            assert "blind1: 60%" in result["message"]
            assert "blind2: 80%" in result["message"]

    @pytest.mark.asyncio
    async def test_get_current_status_exception(self):
        """Test getting current status with exception"""
        config = HubitatConfig(
            accessToken="test-token",
            hubitatUrl="http://test.hub",
            makerApiId="123",
            location=LocationConfig(city="Test", timezone="UTC"),
            houseInformation=HouseInformationConfig(orientation="north-south"),
            rooms={
                "living_room": RoomConfig(
                    blinds=[
                        BlindConfig(id="1", name="blind1"),
                        BlindConfig(id="2", name="blind2"),
                    ]
                )
            },
        )

        with patch(
            "utils.agent_response_utils.HubitatUtils.get_room_current_positions"
        ) as mock_get_positions:
            mock_get_positions.side_effect = Exception("Connection error")

            result = await AgentResponseUtils.get_current_status(config, "living_room")

            assert result["position"] == 50  # Fallback position
            assert result["room"] == "living_room"
            assert result["affected_blinds"] == ["blind1", "blind2"]
            assert "Could not retrieve current positions" in result["message"]

    def test_create_standard_response(self):
        """Test creating standard response"""
        result = AgentResponseUtils.create_standard_response(
            position=25,
            message="Test message",
            room="kitchen",
            affected_blinds=["blind3", "blind4"],
        )

        assert result["position"] == 25
        assert result["message"] == "Test message"
        assert result["room"] == "kitchen"
        assert result["affected_blinds"] == ["blind3", "blind4"]
        assert isinstance(result["timestamp"], datetime)

    def test_create_schedule_response(self):
        """Test creating schedule response"""
        result = AgentResponseUtils.create_schedule_response(
            operation_type="created",
            description="Daily morning routine",
            room="bedroom",
            schedule_id="schedule123",
        )

        assert result["position"] == 0
        assert "Schedule created: Daily morning routine" in result["message"]
        assert "(ID: schedule123)" in result["message"]
        assert result["room"] == "bedroom"
        assert result["affected_blinds"] == []
        assert result["schedule_id"] == "schedule123"
        assert result["operation_type"] == "created"
        assert isinstance(result["timestamp"], datetime)

    def test_create_schedule_response_no_id(self):
        """Test creating schedule response without ID"""
        result = AgentResponseUtils.create_schedule_response(
            operation_type="deleted", description="Remove old schedule", room="office"
        )

        assert result["position"] == 0
        assert result["message"] == "Schedule deleted: Remove old schedule"
        assert result["room"] == "office"
        assert result["affected_blinds"] == []
        assert result["schedule_id"] is None
        assert result["operation_type"] == "deleted"
