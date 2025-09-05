"""
Solar calculation utilities for the Smart Shades Agent
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import pytz
from astral import LocationInfo
from astral.sun import sun, azimuth, elevation, sunrise, sunset

logger = logging.getLogger(__name__)


class SolarUtils:
    """Utility class for solar calculations and window sun exposure"""

    @staticmethod
    def get_solar_info(config) -> Dict[str, Any]:
        """Get current solar position and sun-related information"""
        try:
            if not config.latitude or not config.longitude:
                return {"error": "Location coordinates not configured"}

            # Create location info with timezone from config
            location = LocationInfo(
                "Home",
                "Region",
                config.timezone or "UTC",
                config.latitude,
                config.longitude,
            )

            # Get current time in the configured timezone
            if config.timezone:
                pacific_tz = pytz.timezone(config.timezone)
                now = datetime.now(pacific_tz)
            else:
                now = datetime.now(timezone.utc)

            # Calculate sun position
            sun_azimuth = azimuth(
                location.observer, now
            )  # 0° = North, 90° = East, 180° = South, 270° = West
            sun_elevation_angle = elevation(location.observer, now)  # Above horizon

            # Get sun times for today - use LOCAL date, not UTC date
            if config.timezone:
                pacific_tz = pytz.timezone(config.timezone)
                local_now = datetime.now(pacific_tz)
                today_date = local_now.date()  # Use local date
            else:
                utc_now = datetime.now(pytz.UTC)
                today_date = utc_now.date()

            logger.info(f"Using local date for calculations: {today_date}")

            try:
                # Try calculating for today with explicit UTC date
                sunrise_utc = sunrise(location.observer, date=today_date)
                sunset_utc = sunset(location.observer, date=today_date)

                # Check if sunset is in the morning (wrong!) - if so, try tomorrow's date
                if sunset_utc.time().hour < 12:  # If sunset is in AM, it's wrong
                    logger.warning(
                        f"Sunset appears to be in AM ({sunset_utc.time()}), trying next day..."
                    )
                    tomorrow_date = today_date + timedelta(days=1)
                    sunset_utc = sunset(location.observer, date=tomorrow_date)
                    logger.info(f"  Corrected sunset UTC: {sunset_utc}")

                sun_times = {"sunrise": sunrise_utc, "sunset": sunset_utc}

            except Exception as e:
                logger.error(f"Error with individual sun calculations: {e}")
                # Fallback to original method
                sun_times = sun(location.observer, date=today_date)

            # Convert sunrise/sunset to the same timezone as 'now'
            if config.timezone:
                pacific_tz = pytz.timezone(config.timezone)
                sunrise_local = (
                    sun_times["sunrise"].replace(tzinfo=pytz.UTC).astimezone(pacific_tz)
                )
                sunset_local = (
                    sun_times["sunset"].replace(tzinfo=pytz.UTC).astimezone(pacific_tz)
                )

                logger.info(f"Converted sunrise: {sunrise_local}")
                logger.info(f"Converted sunset: {sunset_local}")
            else:
                sunrise_local = sun_times["sunrise"]
                sunset_local = sun_times["sunset"]

            # Determine if sun is up (comparing times in the same timezone)
            is_sun_up = sunrise_local <= now <= sunset_local

            # Convert azimuth to cardinal direction
            def azimuth_to_direction(azimuth_deg):
                directions = [
                    "north",
                    "northeast",
                    "east",
                    "southeast",
                    "south",
                    "southwest",
                    "west",
                    "northwest",
                ]
                idx = int((azimuth_deg + 22.5) // 45) % 8
                return directions[idx]

            sun_direction = azimuth_to_direction(sun_azimuth)

            return {
                "azimuth": sun_azimuth,
                "elevation": sun_elevation_angle,
                "direction": sun_direction,
                "is_up": is_sun_up,
                "sunrise": sunrise_local.strftime("%H:%M %Z"),
                "sunset": sunset_local.strftime("%H:%M %Z"),
                "current_time": now.strftime("%H:%M %Z"),
                "timezone": config.timezone or "UTC",
                "debug": f"Now: {now}, Sunrise: {sunrise_local}, Sunset: {sunset_local}, Sun up: {is_sun_up}",
            }

        except Exception as e:
            logger.error(f"Error calculating solar info: {e}")
            return {"error": f"Solar calculation failed: {e}"}

    @staticmethod
    def is_window_sunny(
        window_orientation: str, sun_azimuth: float, sun_elevation: float
    ) -> bool:
        """Determine if a window with given orientation is getting direct sunlight"""
        if sun_elevation < 0:  # Sun is below horizon
            return False

        # Map orientations to azimuth ranges
        orientation_ranges = {
            "north": (315, 45),  # 315° to 45° (through 0°)
            "northeast": (0, 90),  # 0° to 90°
            "east": (45, 135),  # 45° to 135°
            "southeast": (90, 180),  # 90° to 180°
            "south": (135, 225),  # 135° to 225°
            "southwest": (180, 270),  # 180° to 270°
            "west": (225, 315),  # 225° to 315°
            "northwest": (270, 360),  # 270° to 360°
        }

        if window_orientation.lower() not in orientation_ranges:
            return False

        start, end = orientation_ranges[window_orientation.lower()]

        # Handle wrapping around 0° for north
        if window_orientation.lower() == "north":
            return sun_azimuth >= start or sun_azimuth <= end
        else:
            return start <= sun_azimuth <= end

    @staticmethod
    def calculate_sun_intensity(
        window_orientation: str, sun_azimuth: float, sun_elevation: float
    ) -> str:
        """Calculate relative sun intensity for a window"""
        if not SolarUtils.is_window_sunny(
            window_orientation, sun_azimuth, sun_elevation
        ):
            return "none"

        # Higher elevation = more intense
        if sun_elevation > 60:
            return "high"
        elif sun_elevation > 30:
            return "medium"
        else:
            return "low"

    @staticmethod
    def get_window_sun_exposure(config, room: str) -> Dict[str, Dict[str, Any]]:
        """Determine which windows are currently exposed to direct sunlight"""
        solar_info = SolarUtils.get_solar_info(config)

        if "error" in solar_info:
            return {"error": solar_info["error"]}

        if not solar_info.get("is_up", False):
            return {"message": "Sun is down - no direct sunlight"}

        window_exposure = {}

        # Check each blind in the room
        if room in config.rooms:
            for blind in config.rooms[room].blinds:
                orientation = getattr(blind, "orientation", "south")

                # Calculate if this window orientation is getting direct sun
                is_sunny = SolarUtils.is_window_sunny(
                    orientation, solar_info["azimuth"], solar_info["elevation"]
                )

                window_exposure[blind.name] = {
                    "orientation": orientation,
                    "is_sunny": is_sunny,
                    "sun_intensity": SolarUtils.calculate_sun_intensity(
                        orientation, solar_info["azimuth"], solar_info["elevation"]
                    ),
                }

        window_exposure["solar_info"] = solar_info
        return window_exposure
