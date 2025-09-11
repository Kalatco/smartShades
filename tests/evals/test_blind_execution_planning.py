"""
Tests for Blind Execution Planning Chain
"""

import pytest
from unittest.mock import AsyncMock, patch

from chains.blind_execution_planning_v2 import BlindExecutionPlanningChain
from models.agent import BlindExecutionRequest, RoomBlindsExecution
from models.config import (
    HubitatConfig,
    RoomConfig,
    BlindConfig,
    LocationConfig,
    HouseInformationConfig,
)


class TestBlindExecutionPlanningChain:
    """Test cases for blind execution planning"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing"""
        return HubitatConfig(
            rooms={
                "living_room": RoomConfig(
                    blinds=[
                        BlindConfig(
                            id="lr_front",
                            name="Living Room Front Window",
                            orientation="south",
                        ),
                        BlindConfig(
                            id="lr_side",
                            name="Living Room Side Window",
                            orientation="east",
                        ),
                    ]
                ),
                "bedroom": RoomConfig(
                    blinds=[
                        BlindConfig(
                            id="br_main",
                            name="Bedroom Main Window",
                            orientation="north",
                        ),
                        BlindConfig(
                            id="br_back", name="Bedroom Back Window", orientation="west"
                        ),
                    ]
                ),
                "kitchen": RoomConfig(
                    blinds=[
                        BlindConfig(
                            id="k_window", name="Kitchen Window", orientation="east"
                        ),
                    ]
                ),
            },
            location=LocationConfig(
                city="San Francisco", timezone="America/Los_Angeles"
            ),
            houseInformation=HouseInformationConfig(
                orientation="east-west",
                notes="Two-story house with morning sun in kitchen",
            ),
        )

    @pytest.fixture
    def chain(self, mock_llm):
        """Create chain instance with mock LLM"""
        return BlindExecutionPlanningChain(mock_llm)

    @pytest.mark.asyncio
    async def test_room_scope_commands(self, chain, mock_llm, sample_config):
        """Test commands that should affect all blinds in current room"""
        test_cases = [
            {
                "command": "close all blinds",
                "current_room": "living_room",
                "expected_rooms": ["living_room"],
                "expected_blinds": ["lr_front", "lr_side"],
                "expected_position": 0,
            },
            {
                "command": "open all windows",
                "current_room": "bedroom",
                "expected_rooms": ["bedroom"],
                "expected_blinds": ["br_main", "br_back"],
                "expected_position": 100,
            },
            {
                "command": "set to half",
                "current_room": "kitchen",
                "expected_rooms": ["kitchen"],
                "expected_blinds": ["k_window"],
                "expected_position": 50,
            },
        ]

        for case in test_cases:
            # Mock the LLM response for room scope
            expected_result = BlindExecutionRequest(
                rooms={
                    case["expected_rooms"][0]: RoomBlindsExecution(
                        blinds={
                            blind_id: case["expected_position"]
                            for blind_id in case["expected_blinds"]
                        }
                    )
                }
            )

            with patch.object(
                chain, "ainvoke", return_value=expected_result
            ) as mock_ainvoke:
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "current_room": case["current_room"],
                        "config": sample_config,
                    }
                )

                assert isinstance(result, BlindExecutionRequest)
                assert list(result.rooms.keys()) == case["expected_rooms"]

                room_execution = result.rooms[case["expected_rooms"][0]]
                assert set(room_execution.blinds.keys()) == set(case["expected_blinds"])

                # Check that all blinds have the expected position
                for blind_id in case["expected_blinds"]:
                    assert room_execution.blinds[blind_id] == case["expected_position"]

    @pytest.mark.asyncio
    async def test_house_wide_commands(self, chain, mock_llm, sample_config):
        """Test commands that should affect the entire house"""
        test_cases = [
            {
                "command": "close all blinds in the house",
                "current_room": "living_room",
                "expected_position": 0,
            },
            {
                "command": "open everything",
                "current_room": "bedroom",
                "expected_position": 100,
            },
            {
                "command": "set entire house to 75%",
                "current_room": "kitchen",
                "expected_position": 75,
            },
        ]

        for case in test_cases:
            # Mock the LLM response for house-wide scope
            expected_result = BlindExecutionRequest(
                rooms={
                    "living_room": RoomBlindsExecution(
                        blinds={
                            "lr_front": case["expected_position"],
                            "lr_side": case["expected_position"],
                        }
                    ),
                    "bedroom": RoomBlindsExecution(
                        blinds={
                            "br_main": case["expected_position"],
                            "br_back": case["expected_position"],
                        }
                    ),
                    "kitchen": RoomBlindsExecution(
                        blinds={"k_window": case["expected_position"]}
                    ),
                }
            )

            with patch.object(
                chain, "ainvoke", return_value=expected_result
            ) as mock_ainvoke:
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "current_room": case["current_room"],
                        "config": sample_config,
                    }
                )

                assert isinstance(result, BlindExecutionRequest)
                assert len(result.rooms) == 3  # All rooms
                assert "living_room" in result.rooms
                assert "bedroom" in result.rooms
                assert "kitchen" in result.rooms

    @pytest.mark.asyncio
    async def test_specific_blind_commands(self, chain, mock_llm, sample_config):
        """Test commands that target specific blinds"""
        test_cases = [
            {
                "command": "close the front window",
                "current_room": "living_room",
                "expected_rooms": ["living_room"],
                "expected_blinds": ["lr_front"],
                "expected_position": 0,
            },
            {
                "command": "open the side window",
                "current_room": "living_room",
                "expected_rooms": ["living_room"],
                "expected_blinds": ["lr_side"],
                "expected_position": 100,
            },
            {
                "command": "set bedroom main window to 25%",
                "current_room": "kitchen",
                "expected_rooms": ["bedroom"],
                "expected_blinds": ["br_main"],
                "expected_position": 25,
            },
        ]

        for case in test_cases:
            # Mock the LLM response for specific blind scope
            expected_result = BlindExecutionRequest(
                rooms={
                    case["expected_rooms"][0]: RoomBlindsExecution(
                        blinds={
                            blind_id: case["expected_position"]
                            for blind_id in case["expected_blinds"]
                        }
                    )
                }
            )

            with patch.object(
                chain, "ainvoke", return_value=expected_result
            ) as mock_ainvoke:
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "current_room": case["current_room"],
                        "config": sample_config,
                    }
                )

                assert isinstance(result, BlindExecutionRequest)
                assert list(result.rooms.keys()) == case["expected_rooms"]

                room_execution = result.rooms[case["expected_rooms"][0]]
                assert set(room_execution.blinds.keys()) == set(case["expected_blinds"])

    @pytest.mark.asyncio
    async def test_orientation_based_commands(self, chain, mock_llm, sample_config):
        """Test commands that use window orientation"""
        test_cases = [
            {
                "command": "close all south windows",
                "current_room": "living_room",
                "expected_rooms": ["living_room"],
                "expected_blinds": ["lr_front"],  # Only south-facing blind
                "expected_position": 0,
            },
            {
                "command": "open east-facing windows",
                "current_room": "living_room",
                "expected_rooms": ["living_room"],
                "expected_blinds": ["lr_side"],  # Only east-facing blind
                "expected_position": 100,
            },
        ]

        for case in test_cases:
            # Mock the LLM response for orientation-based scope
            expected_result = BlindExecutionRequest(
                rooms={
                    case["expected_rooms"][0]: RoomBlindsExecution(
                        blinds={
                            blind_id: case["expected_position"]
                            for blind_id in case["expected_blinds"]
                        }
                    )
                }
            )

            with patch.object(
                chain, "ainvoke", return_value=expected_result
            ) as mock_ainvoke:
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "current_room": case["current_room"],
                        "config": sample_config,
                    }
                )

                assert isinstance(result, BlindExecutionRequest)
                room_execution = result.rooms[case["expected_rooms"][0]]
                assert set(room_execution.blinds.keys()) == set(case["expected_blinds"])

    @pytest.mark.asyncio
    async def test_position_variations(self, chain, mock_llm, sample_config):
        """Test different position expressions"""
        test_cases = [
            {"command": "open blinds fully", "expected_position": 100},
            {"command": "close blinds completely", "expected_position": 0},
            {"command": "set blinds to halfway", "expected_position": 50},
            {"command": "set blinds to quarter", "expected_position": 25},
            {"command": "set blinds to three quarters", "expected_position": 75},
        ]

        for case in test_cases:
            # Mock the LLM response
            expected_result = BlindExecutionRequest(
                rooms={
                    "living_room": RoomBlindsExecution(
                        blinds={
                            "lr_front": case["expected_position"],
                            "lr_side": case["expected_position"],
                        }
                    )
                }
            )

            with patch.object(
                chain, "ainvoke", return_value=expected_result
            ) as mock_ainvoke:
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "current_room": "living_room",
                        "config": sample_config,
                    }
                )

                assert isinstance(result, BlindExecutionRequest)
                room_execution = result.rooms["living_room"]
                for blind_id, position in room_execution.blinds.items():
                    assert position == case["expected_position"]

    @pytest.mark.asyncio
    async def test_error_handling(self, chain, mock_llm, sample_config):
        """Test error handling scenarios"""

        # Test with invalid config
        result = await chain.ainvoke(
            {"command": "close blinds", "current_room": "living_room", "config": None}
        )

        assert isinstance(result, BlindExecutionRequest)
        assert len(result.rooms) == 0  # Empty result for invalid config

        # Test with missing command
        result = await chain.ainvoke(
            {"command": "", "current_room": "living_room", "config": sample_config}
        )

        assert isinstance(result, BlindExecutionRequest)

    @pytest.mark.asyncio
    async def test_llm_exception_handling(self, chain, mock_llm, sample_config):
        """Test handling of LLM exceptions"""

        # Mock LLM to raise an exception
        mock_llm.ainvoke.side_effect = Exception("LLM error")

        result = await chain.ainvoke(
            {
                "command": "close blinds",
                "current_room": "living_room",
                "config": sample_config,
            }
        )

        assert isinstance(result, BlindExecutionRequest)
        assert len(result.rooms) == 0  # Empty result on exception

    @pytest.mark.asyncio
    async def test_cross_room_commands(self, chain, mock_llm, sample_config):
        """Test commands that specify different rooms"""
        test_cases = [
            {
                "command": "close kitchen blinds",
                "current_room": "living_room",
                "expected_rooms": ["kitchen"],
                "expected_blinds": ["k_window"],
            },
            {
                "command": "open bedroom windows",
                "current_room": "kitchen",
                "expected_rooms": ["bedroom"],
                "expected_blinds": ["br_main", "br_back"],
            },
        ]

        for case in test_cases:
            # Mock the LLM response for cross-room commands
            expected_result = BlindExecutionRequest(
                rooms={
                    case["expected_rooms"][0]: RoomBlindsExecution(
                        blinds={blind_id: 100 for blind_id in case["expected_blinds"]}
                    )
                }
            )

            with patch.object(
                chain, "ainvoke", return_value=expected_result
            ) as mock_ainvoke:
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "current_room": case["current_room"],
                        "config": sample_config,
                    }
                )

                assert isinstance(result, BlindExecutionRequest)
                assert list(result.rooms.keys()) == case["expected_rooms"]

                room_execution = result.rooms[case["expected_rooms"][0]]
                assert set(room_execution.blinds.keys()) == set(case["expected_blinds"])


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
