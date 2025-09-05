"""
Solar calculation utilities for the Smart Shades Agent
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import pytz
from astral import LocationInfo
from astral.sun import azimuth, elevation, sunrise, sunset
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

logger = logging.getLogger(__name__)


class SolarUtils:
    """Utility class for solar calculations and window sun exposure"""

    # Cache for geocoded coordinates to avoid repeated API calls
    _coordinate_cache = {}

    @staticmethod
    def _get_timezone_and_now(config):
        """Get timezone and current time consistently"""
        if config.location.timezone:
            tz = pytz.timezone(config.location.timezone)
            return tz, datetime.now(tz)
        else:
            return timezone.utc, datetime.now(timezone.utc)

    @staticmethod
    def _get_coordinates_from_city(city: str) -> Tuple[float, float]:
        """Get coordinates from city name using geocoding service"""

        # Check cache first
        if city in SolarUtils._coordinate_cache:
            logger.debug(f"Using cached coordinates for {city}")
            return SolarUtils._coordinate_cache[city]

        try:
            # Use Nominatim (free OpenStreetMap geocoding service)
            geolocator = Nominatim(user_agent="smart_shades_agent")
            location = geolocator.geocode(city, timeout=10)

            if location:
                coords = (location.latitude, location.longitude)
                # Cache the result
                SolarUtils._coordinate_cache[city] = coords
                logger.info(f"Geocoded '{city}' to coordinates: {coords}")
                return coords
            else:
                logger.warning(
                    f"Could not geocode city '{city}', solar calculations will be unavailable"
                )
                raise ValueError(f"Could not geocode city: {city}")

        except (GeocoderTimedOut, GeocoderUnavailable, Exception) as e:
            logger.warning(
                f"Geocoding failed for '{city}': {e}, solar calculations will be unavailable"
            )
            raise ValueError(f"Geocoding failed: {e}")

    @staticmethod
    def get_solar_info(config) -> Dict[str, Any]:
        """Get current solar position and sun-related information"""
        try:
            # Get coordinates from city
            latitude, longitude = SolarUtils._get_coordinates_from_city(
                config.location.city
            )

            # Create location info
            location = LocationInfo(
                "Home",
                "Region",
                config.location.timezone or "UTC",
                latitude,
                longitude,
            )

            # Get current time in configured timezone
            tz, now = SolarUtils._get_timezone_and_now(config)

            # Calculate sun position
            sun_azimuth = azimuth(location.observer, now)
            sun_elevation_angle = elevation(location.observer, now)

            # Get sun times for today
            today_date = now.date()
            sunrise_utc = sunrise(location.observer, date=today_date)
            sunset_utc = sunset(location.observer, date=today_date)

            # Convert to local timezone
            if config.location.timezone:
                sunrise_local = sunrise_utc.replace(tzinfo=pytz.UTC).astimezone(tz)
                sunset_local = sunset_utc.replace(tzinfo=pytz.UTC).astimezone(tz)
            else:
                sunrise_local = sunrise_utc
                sunset_local = sunset_utc

            # Determine if sun is up
            is_sun_up = sunrise_local <= now <= sunset_local

            # Convert azimuth to cardinal direction
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
            direction_idx = int((sun_azimuth + 22.5) // 45) % 8
            sun_direction = directions[direction_idx]

            return {
                "azimuth": sun_azimuth,
                "elevation": sun_elevation_angle,
                "direction": sun_direction,
                "is_up": is_sun_up,
                "sunrise": sunrise_local.strftime("%H:%M %Z"),
                "sunset": sunset_local.strftime("%H:%M %Z"),
                "current_time": now.strftime("%H:%M %Z"),
                "timezone": config.location.timezone or "UTC",
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

        # Simplified orientation ranges (45° wide each)
        orientation_ranges = {
            "north": (315, 45),
            "northeast": (0, 90),
            "east": (45, 135),
            "southeast": (90, 180),
            "south": (135, 225),
            "southwest": (180, 270),
            "west": (225, 315),
            "northwest": (270, 360),
        }

        range_data = orientation_ranges.get(window_orientation.lower())
        if not range_data:
            return False

        start, end = range_data
        # Handle north wrapping around 0°
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

        # Simple elevation-based intensity
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
