"""
Configuration management utilities for Smart Shades Agent
"""

import logging
import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from models.config import HubitatConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Utility class for managing configuration and environment setup"""

    @staticmethod
    def load_environment():
        """Load environment variables from .env file"""
        load_dotenv()

    @staticmethod
    async def load_blinds_config() -> HubitatConfig:
        """Load blinds configuration from JSON file"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "blinds_config.json",
        )

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            config = HubitatConfig(**config_data)
            logger.info(f"Loaded configuration for {len(config.rooms)} rooms")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    @staticmethod
    def override_hubitat_config(config: HubitatConfig) -> HubitatConfig:
        """Override config with environment variables for Hubitat"""
        hubitat_access_token = os.getenv("HUBITAT_ACCESS_TOKEN")
        hubitat_api_url = os.getenv("HUBITAT_API_URL")

        if hubitat_access_token:
            config.accessToken = hubitat_access_token
        if hubitat_api_url:
            config.hubitatUrl = hubitat_api_url
        if not config.makerApiId:
            config.makerApiId = "1"  # Default setup

        return config

    @staticmethod
    def create_azure_llm() -> AzureChatOpenAI:
        """Create and configure Azure OpenAI LLM instance"""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([api_key, azure_endpoint, deployment_name, api_version]):
            raise ValueError(
                "Azure OpenAI environment variables are required: "
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION"
            )

        return AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            deployment_name=deployment_name,
            api_version=api_version,
            temperature=0,
        )

    @staticmethod
    def validate_environment() -> bool:
        """Validate that all required environment variables are set"""
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_VERSION",
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        return True

    @staticmethod
    def get_config_summary(config: HubitatConfig) -> dict:
        """Get a summary of the current configuration for logging/debugging"""
        return {
            "total_rooms": len(config.rooms),
            "room_names": list(config.rooms.keys()),
            "total_blinds": sum(len(room.blinds) for room in config.rooms.values()),
            "hubitat_configured": bool(config.accessToken and config.hubitatUrl),
            "maker_api_id": config.makerApiId,
            "house_orientation": (
                getattr(config.houseInformation, "orientation", "not set") or "not set"
                if hasattr(config, "houseInformation")
                else "not configured"
            ),
        }
