"""
Hubitat API utilities for the Smart Shades Agent
"""

import logging
from typing import Dict
import httpx

logger = logging.getLogger(__name__)


class HubitatUtils:
    """Utility class for Hubitat API interactions"""

    @staticmethod
    async def control_blinds(config, blinds, position: int):
        """Send HTTP requests to control individual blinds"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            for blind in blinds:
                url = f"{config.hubitatUrl}/apps/api/{config.makerApiId}/devices/{blind.id}/setPosition/{position}?access_token={config.accessToken}"

                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        logger.info(f"Successfully set {blind.name} to {position}%")
                    else:
                        logger.error(
                            f"Failed to control {blind.name}: HTTP {response.status_code} - {response.text}"
                        )
                except Exception as e:
                    logger.error(f"Error controlling {blind.name}: {e}")

    @staticmethod
    async def get_blind_current_position(config, blind_id: str) -> int:
        """Get current position of a specific blind from Hubitat"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{config.hubitatUrl}/apps/api/{config.makerApiId}/devices/{blind_id}?access_token={config.accessToken}"
                logger.info(f"Getting blind {blind_id} position from: {url}")
                response = await client.get(url)

                if response.status_code == 200:
                    device_data = response.json()
                    # Look for position attribute in the device attributes
                    for attr in device_data.get("attributes", []):
                        if attr.get("name") == "position":
                            return int(attr.get("currentValue", 50))
                    # Fallback to looking for 'level' attribute
                    for attr in device_data.get("attributes", []):
                        if attr.get("name") == "level":
                            return int(attr.get("currentValue", 50))
                    return 50  # Default if no position found
                else:
                    logger.warning(
                        f"Failed to get device {blind_id} status: HTTP {response.status_code}"
                    )
                    return 50
        except Exception as e:
            logger.error(f"Error getting blind {blind_id} position: {e}")
            return 50

    @staticmethod
    async def get_room_current_positions(config, room: str) -> Dict[str, int]:
        """Get current positions of all blinds in a room"""
        if room not in config.rooms:
            return {}

        positions = {}
        for blind in config.rooms[room].blinds:
            positions[blind.name] = await HubitatUtils.get_blind_current_position(
                config, blind.id
            )

        return positions
