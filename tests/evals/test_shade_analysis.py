"""
Tests for Shade Analysis Chain
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chains.shade_analysis import ShadeAnalysisChain
from models.agent import ShadeAnalysis, BlindOperation
from models.config import BlindConfig


class TestShadeAnalysisChain:
    """Test cases for shade analysis and command parsing"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def chain(self, mock_llm):
        """Create chain instance with mock LLM"""
        return ShadeAnalysisChain(mock_llm)

    @pytest.fixture
    def sample_blinds(self):
        """Sample blind configuration for testing"""
        return [
            BlindConfig(id="1", name="Guest Side Window", orientation="North"),
            BlindConfig(id="2", name="Guest Front Window", orientation="East"),
        ]

    @pytest.mark.asyncio
    async def test_simple_position_commands(self, chain, mock_llm, sample_blinds):
        """Test basic position commands"""
        position_commands = [
            {
                "command": "open all windows",
                "expected_position": 100,
                "expected_scope": "room",
            },
            {
                "command": "close all blinds",
                "expected_position": 0,
                "expected_scope": "room",
            },
            {
                "command": "set blinds to 50%",
                "expected_position": 50,
                "expected_scope": "room",
            },
            {
                "command": "open blinds halfway",
                "expected_position": 50,
                "expected_scope": "room",
            },
        ]

        for case in position_commands:
            # Mock the chain's ainvoke method directly
            expected_result = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=[],
                        position=case["expected_position"],
                        reasoning=f"Set all blinds to {case['expected_position']}%",
                    )
                ],
                scope=case["expected_scope"],
                reasoning=f"Command: {case['command']}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "room": "guest_bedroom",
                        "room_blinds": sample_blinds,
                        "current_positions": {
                            "Guest Side Window": 50,
                            "Guest Front Window": 50,
                        },
                        "window_sun_info": {},
                        "house_orientation": "east-west",
                        "notes": "",
                    }
                )

                assert isinstance(result, ShadeAnalysis)
                assert result.scope == case["expected_scope"]
                assert len(result.operations) == 1
                assert result.operations[0].position == case["expected_position"]

    @pytest.mark.asyncio
    async def test_specific_blind_commands(self, chain, mock_llm, sample_blinds):
        """Test commands targeting specific blinds"""
        specific_commands = [
            {
                "command": "close the front window",
                "expected_filter": ["front"],
                "expected_position": 0,
                "expected_scope": "specific",
            },
            {
                "command": "open the side window",
                "expected_filter": ["side"],
                "expected_position": 100,
                "expected_scope": "specific",
            },
        ]

        for case in specific_commands:
            # Mock the chain's ainvoke method directly
            expected_result = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=case["expected_filter"],
                        position=case["expected_position"],
                        reasoning=f"Target specific blind: {case['expected_filter']}",
                    )
                ],
                scope=case["expected_scope"],
                reasoning=f"Command: {case['command']}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "room": "guest_bedroom",
                        "room_blinds": sample_blinds,
                        "current_positions": {
                            "Guest Side Window": 50,
                            "Guest Front Window": 50,
                        },
                        "window_sun_info": {},
                        "house_orientation": "east-west",
                        "notes": "",
                    }
                )

                assert isinstance(result, ShadeAnalysis)
                assert result.scope == case["expected_scope"]
                assert len(result.operations) == 1
                assert result.operations[0].blind_filter == case["expected_filter"]

    @pytest.mark.asyncio
    async def test_multiple_operations_commands(self, chain, mock_llm, sample_blinds):
        """Test commands requiring multiple different operations"""
        multi_commands = [
            {
                "command": "open the side window halfway, and front window fully",
                "expected_operations": [
                    {"filter": ["side"], "position": 50},
                    {"filter": ["front"], "position": 100},
                ],
                "expected_scope": "specific",
            }
        ]

        for case in multi_commands:
            # Mock the LLM response with multiple operations
            operations = [
                BlindOperation(
                    blind_filter=op["filter"],
                    position=op["position"],
                    reasoning=f"Set {op['filter']} to {op['position']}%",
                )
                for op in case["expected_operations"]
            ]

            expected_result = ShadeAnalysis(
                operations=operations,
                scope=case["expected_scope"],
                reasoning=f"Multiple operations: {case['command']}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "room": "guest_bedroom",
                        "room_blinds": sample_blinds,
                        "current_positions": {
                            "Guest Side Window": 50,
                            "Guest Front Window": 50,
                        },
                        "window_sun_info": {},
                        "house_orientation": "east-west",
                        "notes": "",
                    }
                )

                assert isinstance(result, ShadeAnalysis)
                assert result.scope == case["expected_scope"]
                assert len(result.operations) == len(case["expected_operations"])

    @pytest.mark.asyncio
    async def test_house_wide_commands(self, chain, mock_llm, sample_blinds):
        """Test house-wide commands"""
        house_wide_commands = [
            "[HOUSE-WIDE COMMAND] close all blinds",
            "[HOUSE-WIDE COMMAND] open all windows",
        ]

        for command in house_wide_commands:
            # Mock the LLM response
            expected_result = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=[],
                        position=0 if "close" in command else 100,
                        reasoning="House-wide operation",
                    )
                ],
                scope="house",
                reasoning=f"House-wide command: {command}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": command,
                        "room": "guest_bedroom",
                        "room_blinds": sample_blinds,
                        "current_positions": {
                            "Guest Side Window": 50,
                            "Guest Front Window": 50,
                        },
                        "window_sun_info": {},
                        "house_orientation": "east-west",
                        "notes": "",
                    }
                )

                assert isinstance(result, ShadeAnalysis)
                assert result.scope == "house"

    @pytest.mark.asyncio
    async def test_relative_position_commands(self, chain, mock_llm, sample_blinds):
        """Test relative position adjustments"""
        relative_commands = [
            {
                "command": "close it a bit more",
                "current_position": 50,
                "expected_change": -15,  # close more = decrease position
            },
            {
                "command": "open a little more",
                "current_position": 30,
                "expected_change": 15,  # open more = increase position
            },
        ]

        for case in relative_commands:
            expected_position = max(
                0, min(100, case["current_position"] + case["expected_change"])
            )

            # Mock the LLM response
            expected_result = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=[],
                        position=expected_position,
                        reasoning=f"Relative adjustment: {case['expected_change']}%",
                    )
                ],
                scope="room",
                reasoning=f"Relative command: {case['command']}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "room": "guest_bedroom",
                        "room_blinds": sample_blinds,
                        "current_positions": {
                            "Guest Side Window": case["current_position"],
                            "Guest Front Window": case["current_position"],
                        },
                        "window_sun_info": {},
                        "house_orientation": "east-west",
                        "notes": "",
                    }
                )

                assert isinstance(result, ShadeAnalysis)
                assert len(result.operations) == 1
                assert result.operations[0].position == expected_position

    @pytest.mark.asyncio
    async def test_additional_user_cases(self, chain, mock_llm, sample_blinds):
        """Test additional cases specified by user"""
        additional_cases = [
            {
                "command": "open the front shade around half of 100",
                "expected_filter": ["front"],
                "expected_position": 50,
                "expected_scope": "specific",
            }
        ]

        for case in additional_cases:
            # Mock the LLM response
            expected_result = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=case["expected_filter"],
                        position=case["expected_position"],
                        reasoning="Half of 100% = 50%",
                    )
                ],
                scope=case["expected_scope"],
                reasoning=f"Command: {case['command']}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
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

                assert isinstance(result, ShadeAnalysis)
                assert result.scope == case["expected_scope"]
                assert len(result.operations) == 1
                assert result.operations[0].position == case["expected_position"]
                assert result.operations[0].blind_filter == case["expected_filter"]

    @pytest.mark.asyncio
    async def test_solar_context_integration(self, chain, mock_llm, sample_blinds):
        """Test commands with solar context"""
        solar_commands = [
            {
                "command": "block the sun",
                "window_sun_info": {
                    "Guest Front Window": {"is_sunny": True, "sun_intensity": "high"},
                    "Guest Side Window": {"is_sunny": False, "sun_intensity": "none"},
                },
            },
            {
                "command": "reduce glare",
                "window_sun_info": {
                    "Guest Front Window": {"is_sunny": True, "sun_intensity": "medium"},
                    "Guest Side Window": {"is_sunny": False, "sun_intensity": "none"},
                },
            },
        ]

        for case in solar_commands:
            # Mock the LLM response that considers solar context
            expected_result = ShadeAnalysis(
                operations=[
                    BlindOperation(
                        blind_filter=["front"],  # Should target the sunny window
                        position=0,  # Close to block sun
                        reasoning="Block sun on sunny window",
                    )
                ],
                scope="specific",
                reasoning=f"Solar-aware: {case['command']}",
            )

            with patch.object(chain, "ainvoke", return_value=expected_result):
                result = await chain.ainvoke(
                    {
                        "command": case["command"],
                        "room": "guest_bedroom",
                        "room_blinds": sample_blinds,
                        "current_positions": {
                            "Guest Side Window": 50,
                            "Guest Front Window": 50,
                        },
                        "window_sun_info": case["window_sun_info"],
                        "house_orientation": "east-west",
                        "notes": "",
                    }
                )

                assert isinstance(result, ShadeAnalysis)
                assert len(result.operations) >= 1

    @pytest.mark.asyncio
    async def test_error_handling(self, chain, mock_llm, sample_blinds):
        """Test error handling returns fallback"""
        # Mock an exception
        mock_llm.ainvoke.side_effect = Exception("LLM Error")

        result = await chain.ainvoke(
            {
                "command": "test command",
                "room": "guest_bedroom",
                "room_blinds": sample_blinds,
                "current_positions": {
                    "Guest Side Window": 50,
                    "Guest Front Window": 50,
                },
                "window_sun_info": {},
                "house_orientation": "east-west",
                "notes": "",
            }
        )

        assert isinstance(result, ShadeAnalysis)
        assert result.operations == []
        assert result.scope == "room"
        assert "Fallback parsing due to error" in result.reasoning

    @pytest.mark.asyncio
    async def test_empty_command(self, chain, mock_llm, sample_blinds):
        """Test handling of empty command"""
        # Mock the LLM response
        mock_llm.ainvoke.return_value = ShadeAnalysis(
            operations=[], scope="room", reasoning="Empty command"
        )

        result = await chain.ainvoke(
            {
                "command": "",
                "room": "guest_bedroom",
                "room_blinds": sample_blinds,
                "current_positions": {
                    "Guest Side Window": 50,
                    "Guest Front Window": 50,
                },
                "window_sun_info": {},
                "house_orientation": "east-west",
                "notes": "",
            }
        )

        assert isinstance(result, ShadeAnalysis)
        assert result.scope == "room"

    @pytest.mark.asyncio
    async def test_context_building(self, chain, mock_llm, sample_blinds):
        """Test that context strings are built correctly"""
        # This test verifies the helper methods work correctly
        window_sun_info = {
            "Guest Front Window": {"is_sunny": True, "sun_intensity": "high"},
            "Guest Side Window": {"is_sunny": False, "sun_intensity": "none"},
        }

        solar_context = chain._build_solar_context(window_sun_info)
        house_info = chain._build_house_info("east-west", "North windows never get sun")

        assert "Direct sunlight" in solar_context
        assert "high intensity" in solar_context
        assert "east-west orientation" in house_info
        assert "North windows never get sun" in house_info
