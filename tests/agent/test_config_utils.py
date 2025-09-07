"""
Tests for ConfigManager utility
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, mock_open, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from utils.config_utils import ConfigManager
from models.config import HubitatConfig


class TestConfigManager:
    """Test cases for ConfigManager utility class"""

    def test_load_environment(self):
        """Test environment loading"""
        with patch("utils.config_utils.load_dotenv") as mock_load_dotenv:
            ConfigManager.load_environment()
            mock_load_dotenv.assert_called_once()

    def test_validate_environment_success(self):
        """Test successful environment validation"""
        env_vars = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
            "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
        }

        with patch.dict(os.environ, env_vars):
            result = ConfigManager.validate_environment()
            assert result is True

    def test_validate_environment_missing_vars(self):
        """Test environment validation with missing variables"""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = ConfigManager.validate_environment()
            assert result is False

    def test_validate_environment_partial_vars(self):
        """Test environment validation with some missing variables"""
        partial_env = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com",
            # Missing DEPLOYMENT_NAME and API_VERSION
        }

        with patch.dict(os.environ, partial_env, clear=True):
            result = ConfigManager.validate_environment()
            assert result is False

    @pytest.mark.asyncio
    async def test_load_blinds_config_success(self):
        """Test successful blinds configuration loading"""
        mock_config_data = {
            "accessToken": "test-token",
            "hubitatUrl": "http://test.hub",
            "makerApiId": "123",
            "rooms": {
                "test_room": {
                    "blinds": [
                        {"id": "1", "name": "Test Blind", "orientation": "North"}
                    ]
                }
            },
            "location": {"city": "Seattle", "timezone": "America/Los_Angeles"},
            "houseInformation": {"orientation": "north-south"},
        }

        mock_file_content = json.dumps(mock_config_data)

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            with patch("os.path.join") as mock_join:
                mock_join.return_value = "test_path/blinds_config.json"

                config = await ConfigManager.load_blinds_config()

                assert isinstance(config, HubitatConfig)
                assert config.accessToken == "test-token"
                assert len(config.rooms) == 1
                assert "test_room" in config.rooms

    @pytest.mark.asyncio
    async def test_load_blinds_config_file_not_found(self):
        """Test blinds configuration loading with missing file"""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                await ConfigManager.load_blinds_config()

    @pytest.mark.asyncio
    async def test_load_blinds_config_invalid_json(self):
        """Test blinds configuration loading with invalid JSON"""
        with patch("builtins.open", mock_open(read_data="invalid json {")):
            with pytest.raises(json.JSONDecodeError):
                await ConfigManager.load_blinds_config()

    def test_override_hubitat_config_with_env_vars(self):
        """Test Hubitat config override with environment variables"""
        # Create a basic config
        config_data = {
            "accessToken": "original-token",
            "hubitatUrl": "http://original.hub",
            "makerApiId": "",
            "rooms": {},
            "location": {"city": "Seattle", "timezone": "America/Los_Angeles"},
            "houseInformation": {"orientation": "north-south"},
        }
        config = HubitatConfig(**config_data)

        env_vars = {
            "HUBITAT_ACCESS_TOKEN": "new-token",
            "HUBITAT_API_URL": "http://new.hub",
        }

        with patch.dict(os.environ, env_vars):
            updated_config = ConfigManager.override_hubitat_config(config)

            assert updated_config.accessToken == "new-token"
            assert updated_config.hubitatUrl == "http://new.hub"
            assert updated_config.makerApiId == "1"  # Default value

    def test_override_hubitat_config_no_env_vars(self):
        """Test Hubitat config override without environment variables"""
        config_data = {
            "accessToken": "original-token",
            "hubitatUrl": "http://original.hub",
            "makerApiId": "456",
            "rooms": {},
            "location": {"city": "Seattle", "timezone": "America/Los_Angeles"},
            "houseInformation": {"orientation": "north-south"},
        }
        config = HubitatConfig(**config_data)

        with patch.dict(os.environ, {}, clear=True):
            updated_config = ConfigManager.override_hubitat_config(config)

            # Should remain unchanged
            assert updated_config.accessToken == "original-token"
            assert updated_config.hubitatUrl == "http://original.hub"
            assert updated_config.makerApiId == "456"

    def test_create_azure_llm_success(self):
        """Test successful Azure LLM creation"""
        env_vars = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
            "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
        }

        with patch.dict(os.environ, env_vars):
            with patch("utils.config_utils.AzureChatOpenAI") as mock_llm:
                mock_llm_instance = MagicMock()
                mock_llm.return_value = mock_llm_instance

                llm = ConfigManager.create_azure_llm()

                mock_llm.assert_called_once_with(
                    api_key="test-key",
                    azure_endpoint="https://test.endpoint.com",
                    deployment_name="test-deployment",
                    api_version="2024-12-01-preview",
                    temperature=0,
                )
                assert llm == mock_llm_instance

    def test_create_azure_llm_missing_env_vars(self):
        """Test Azure LLM creation with missing environment variables"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                ConfigManager.create_azure_llm()

            assert "Azure OpenAI environment variables are required" in str(
                exc_info.value
            )

    def test_get_config_summary(self):
        """Test configuration summary generation"""
        config_data = {
            "accessToken": "test-token",
            "hubitatUrl": "http://test.hub",
            "makerApiId": "123",
            "rooms": {
                "room1": {
                    "blinds": [
                        {"id": "1", "name": "Blind 1", "orientation": "North"},
                        {"id": "2", "name": "Blind 2", "orientation": "South"},
                    ]
                },
                "room2": {
                    "blinds": [{"id": "3", "name": "Blind 3", "orientation": "East"}]
                },
            },
            "location": {"city": "Seattle", "timezone": "America/Los_Angeles"},
            "houseInformation": {"orientation": "north-south"},
        }
        config = HubitatConfig(**config_data)

        summary = ConfigManager.get_config_summary(config)

        assert summary["total_rooms"] == 2
        assert summary["room_names"] == ["room1", "room2"]
        assert summary["total_blinds"] == 3
        assert summary["hubitat_configured"] is True
        assert summary["maker_api_id"] == "123"
        assert summary["house_orientation"] == "north-south"

    def test_get_config_summary_minimal_config(self):
        """Test configuration summary with minimal config"""
        config_data = {
            "accessToken": "",
            "hubitatUrl": "",
            "makerApiId": "",
            "rooms": {},
            "location": {"city": "Seattle", "timezone": "UTC"},
            "houseInformation": {"orientation": None},
        }
        config = HubitatConfig(**config_data)

        summary = ConfigManager.get_config_summary(config)

        assert summary["total_rooms"] == 0
        assert summary["room_names"] == []
        assert summary["total_blinds"] == 0
        assert summary["hubitat_configured"] is False
        assert summary["house_orientation"] == "not set"

    def test_get_config_summary_no_house_info(self):
        """Test configuration summary without house information"""
        config_data = {
            "accessToken": "test-token",
            "hubitatUrl": "http://test.hub",
            "makerApiId": "123",
            "rooms": {"room1": {"blinds": []}},
            "location": {"city": "Seattle", "timezone": "America/Los_Angeles"},
            "houseInformation": {"orientation": None},
        }
        config = HubitatConfig(**config_data)

        # Remove houseInformation if it exists
        if hasattr(config, "houseInformation"):
            delattr(config, "houseInformation")

        summary = ConfigManager.get_config_summary(config)

        assert summary["house_orientation"] == "not configured"
