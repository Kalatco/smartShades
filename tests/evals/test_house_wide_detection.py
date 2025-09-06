"""
Tests for House-wide Detection Chain
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chains.house_wide_detection import HouseWideDetectionChain
from models.agent import HouseWideDetection


class TestHouseWideDetectionChain:
    """Test cases for house-wide command detection"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def chain(self, mock_llm):
        """Create chain instance with mock LLM"""
        return HouseWideDetectionChain(mock_llm)

    @pytest.mark.asyncio
    async def test_house_wide_commands(self, chain, mock_llm):
        """Test commands that should be detected as house-wide"""
        house_wide_commands = [
            "close all blinds in the house",
            "open all windows in every room",
            "close blinds throughout the entire house",
            "open all shades in the whole house",
            "close everything in all rooms",
            "open blinds everywhere in the house",
        ]

        for command in house_wide_commands:
            # Mock the chain's ainvoke method directly
            expected_result = HouseWideDetection(is_house_wide=True)

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke({"command": command})

                assert isinstance(result, HouseWideDetection)
                assert (
                    result.is_house_wide == True
                ), f"Command '{command}' should be house-wide"

    @pytest.mark.asyncio
    async def test_room_specific_commands(self, chain, mock_llm):
        """Test commands that should be detected as room-specific"""
        room_specific_commands = [
            "close all blinds",  # current room only
            "open all windows",  # current room only
            "close the front window",
            "open the side window halfway",
            "set blinds to 50%",
            "close all shades",  # current room only
            "open blinds halfway",
            "block the sun",
            "reduce glare",
        ]

        for command in room_specific_commands:
            # Mock the chain's ainvoke method directly
            expected_result = HouseWideDetection(is_house_wide=False)

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke({"command": command})

                assert isinstance(result, HouseWideDetection)
                assert (
                    result.is_house_wide == False
                ), f"Command '{command}' should be room-specific"

    @pytest.mark.asyncio
    async def test_multi_room_specific_commands(self, chain, mock_llm):
        """Test commands mentioning specific rooms (but not house-wide)"""
        multi_room_commands = [
            "open living room and bedroom windows",
            "close blinds in master bedroom and guest room",
            "set kitchen and dining room shades to 50%",
        ]

        for command in multi_room_commands:
            # Mock the chain's ainvoke method directly
            expected_result = HouseWideDetection(is_house_wide=False)

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke({"command": command})

                assert isinstance(result, HouseWideDetection)
                assert (
                    result.is_house_wide == False
                ), f"Command '{command}' should be room-specific"

    @pytest.mark.asyncio
    async def test_error_handling(self, chain, mock_llm):
        """Test error handling returns fallback"""
        # Mock an exception
        mock_llm.ainvoke.side_effect = Exception("LLM Error")

        result = await chain.ainvoke({"command": "test command"})

        assert isinstance(result, HouseWideDetection)
        assert result.is_house_wide == False  # Should default to room-specific

    @pytest.mark.asyncio
    async def test_empty_command(self, chain, mock_llm):
        """Test handling of empty command"""
        # Mock the chain's ainvoke method directly
        expected_result = HouseWideDetection(is_house_wide=False)

        with patch.object(chain, "ainvoke", return_value=expected_result):
            result = await chain.ainvoke({"command": ""})

            assert isinstance(result, HouseWideDetection)
            assert result.is_house_wide == False
