"""
Solar calculation utilities for the Smart Shades Agent using pvlib-python
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import pytz
import numpy as np
from pvlib import location, irradiance
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

logger = logging.getLogger(__name__)


class SolarUtils:
    """Utility class for solar calculations and window sun exposure using pvlib"""

    # Cache for geocoded coordinates to avoid repeated API calls
    _coordinate_cache = {}
    # Cache for pvlib Location objects to avoid recreation
    _location_cache = {}
    # Cache for solar calculations to avoid repeated calculations for same time
    _solar_cache = {}
    _solar_cache_ttl = 300  # 5 minutes TTL for solar cache

    # Orientation constants for better performance
    ORIENTATION_RANGES = {
        "north": (315, 45),
        "northeast": (0, 90),
        "east": (45, 135),
        "southeast": (90, 180),
        "south": (135, 225),
        "southwest": (180, 270),
        "west": (225, 315),
        "northwest": (270, 360),
    }

    ORIENTATION_TO_AZIMUTH = {
        "north": 0,
        "northeast": 45,
        "east": 90,
        "southeast": 135,
        "south": 180,
        "southwest": 225,
        "west": 270,
        "northwest": 315,
    }

    # Compass directions for precise azimuth conversion
    COMPASS_DIRECTIONS = [
        "north",
        "north-northeast",
        "northeast",
        "east-northeast",
        "east",
        "east-southeast",
        "southeast",
        "south-southeast",
        "south",
        "south-southwest",
        "southwest",
        "west-southwest",
        "west",
        "west-northwest",
        "northwest",
        "north-northwest",
    ]

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
        """Get current solar position and sun-related information using pvlib with caching"""
        try:
            # Create cache key based on location and time (rounded to nearest 5 minutes)
            now_rounded = datetime.now().replace(
                minute=(datetime.now().minute // 5) * 5, second=0, microsecond=0
            )
            cache_key = f"{config.location.city}_{now_rounded.isoformat()}"

            # Check solar calculation cache first
            if cache_key in SolarUtils._solar_cache:
                cached_result = SolarUtils._solar_cache[cache_key]
                # Check if cache is still valid (within TTL)
                cache_time = cached_result.get("_cache_time", 0)
                if (
                    datetime.now().timestamp() - cache_time
                ) < SolarUtils._solar_cache_ttl:
                    logger.debug(f"Using cached solar info for {config.location.city}")
                    return {
                        k: v for k, v in cached_result.items() if not k.startswith("_")
                    }

            # Get coordinates from city
            latitude, longitude = SolarUtils._get_coordinates_from_city(
                config.location.city
            )

            # Create cache key for location object
            site_timezone = config.location.timezone or "UTC"
            altitude = getattr(config.location, "altitude", 100)
            location_cache_key = f"{latitude}_{longitude}_{site_timezone}_{altitude}"

            # Check location cache first
            if location_cache_key in SolarUtils._location_cache:
                site = SolarUtils._location_cache[location_cache_key]
            else:
                # Create pvlib Location object with proper altitude
                site = location.Location(
                    latitude=latitude,
                    longitude=longitude,
                    tz=site_timezone,
                    altitude=altitude,
                    name=f"Smart Shades - {config.location.city}",
                )
                # Cache the location object
                SolarUtils._location_cache[location_cache_key] = site

            # Get current time in configured timezone
            tz, now = SolarUtils._get_timezone_and_now(config)

            # Calculate solar position using pvlib with error handling
            try:
                solar_position = site.get_solarposition(now)
                sun_azimuth = float(solar_position["azimuth"].iloc[0])
                sun_elevation_true = float(solar_position["elevation"].iloc[0])
                sun_elevation_apparent = float(
                    solar_position["apparent_elevation"].iloc[0]
                )
            except Exception as solar_error:
                logger.error(f"Solar position calculation failed: {solar_error}")
                # Use fallback values
                sun_azimuth = 180.0  # South
                sun_elevation_true = 30.0
                sun_elevation_apparent = 30.0

            # Calculate DNI with error handling
            try:
                clearsky = site.get_clearsky(now)
                dni = float(clearsky["dni"].iloc[0]) if not clearsky.empty else 0
            except Exception as clearsky_error:
                logger.warning(f"Clearsky calculation failed: {clearsky_error}")
                dni = 0

            # Get sunrise and sunset times - with comprehensive error handling
            try:
                times = site.get_sun_rise_set_transit(now.date())
                sunrise_time = times["sunrise"].iloc[0]
                sunset_time = times["sunset"].iloc[0]

                # Try multiple approaches to extract time strings
                try:
                    if hasattr(sunrise_time, "strftime"):
                        sunrise_str = sunrise_time.strftime("%H:%M")
                        sunset_str = sunset_time.strftime("%H:%M")
                    elif hasattr(sunrise_time, "time"):
                        sunrise_str = sunrise_time.time().strftime("%H:%M")
                        sunset_str = sunset_time.time().strftime("%H:%M")
                    else:
                        # Extract from string representation
                        sunrise_str = (
                            str(sunrise_time).split()[1][:5]
                            if " " in str(sunrise_time)
                            else "06:00"
                        )
                        sunset_str = (
                            str(sunset_time).split()[1][:5]
                            if " " in str(sunset_time)
                            else "18:00"
                        )
                except Exception as format_error:
                    logger.warning(f"Time formatting error: {format_error}")
                    sunrise_str = "06:00"
                    sunset_str = "18:00"

            except Exception as time_error:
                logger.warning(f"Sunrise/sunset calculation failed: {time_error}")
                sunrise_str = "06:00"
                sunset_str = "18:00"

            # Determine if sun is up using apparent elevation (accounts for refraction)
            is_sun_up = sun_elevation_apparent > -0.833  # Civil twilight threshold

            # Convert azimuth to cardinal direction with better precision
            directions = [
                "north",
                "north-northeast",
                "northeast",
                "east-northeast",
                "east",
                "east-southeast",
                "southeast",
                "south-southeast",
                "south",
                "south-southwest",
                "southwest",
                "west-southwest",
                "west",
                "west-northwest",
                "northwest",
                "north-northwest",
            ]
            direction_idx = int((sun_azimuth + 11.25) // 22.5) % 16
            sun_direction = directions[direction_idx]

            # Prepare result
            result = {
                "azimuth": sun_azimuth,
                "elevation": sun_elevation_apparent,  # Use apparent elevation
                "elevation_true": sun_elevation_true,  # Also provide true elevation
                "direction": sun_direction,
                "is_up": is_sun_up,
                "dni": dni,  # Direct Normal Irradiance for advanced calculations
                "sunrise": f"{sunrise_str} {site_timezone}",
                "sunset": f"{sunset_str} {site_timezone}",
                "current_time": now.strftime("%H:%M %Z"),
                "timezone": site_timezone,
                "coordinates": {"lat": latitude, "lon": longitude, "alt": altitude},
            }

            # Cache the result with timestamp
            cached_result = result.copy()
            cached_result["_cache_time"] = datetime.now().timestamp()
            SolarUtils._solar_cache[cache_key] = cached_result

            # Clean old cache entries to prevent memory bloat
            current_time = datetime.now().timestamp()
            SolarUtils._solar_cache = {
                k: v
                for k, v in SolarUtils._solar_cache.items()
                if (current_time - v.get("_cache_time", 0))
                < SolarUtils._solar_cache_ttl * 2
            }

            return result

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

        orientation_lower = window_orientation.lower()
        range_data = SolarUtils.ORIENTATION_RANGES.get(orientation_lower)
        if not range_data:
            return False

        start, end = range_data
        # Handle north wrapping around 0°
        if orientation_lower == "north":
            return sun_azimuth >= start or sun_azimuth <= end
        else:
            return start <= sun_azimuth <= end

    @staticmethod
    def calculate_sun_intensity(
        window_orientation: str,
        sun_azimuth: float,
        sun_elevation: float,
        dni: Optional[float] = None,
    ) -> str:
        """Calculate relative sun intensity using elevation angle and optional DNI data"""
        if not SolarUtils.is_window_sunny(
            window_orientation, sun_azimuth, sun_elevation
        ):
            return "none"

        # Use DNI (Direct Normal Irradiance) if available for more accurate intensity
        if dni is not None and dni > 0:
            # DNI-based intensity (W/m²)
            if dni > 800:
                return "high"
            elif dni > 500:
                return "medium"
            elif dni > 200:
                return "low"
            elif dni > 50:
                return "minimal"
            else:
                return "none"

        # Fallback to elevation-based calculation with improved thresholds
        if sun_elevation > 60:
            return "high"
        elif sun_elevation > 40:
            return "medium"
        elif sun_elevation > 20:
            return "low"
        elif sun_elevation > 5:
            return "minimal"
        else:
            return "none"

    @staticmethod
    def calculate_window_irradiance(
        window_orientation: str, sun_azimuth: float, sun_elevation: float, dni: float
    ) -> float:
        """Calculate direct solar irradiance on a window surface using pvlib algorithms"""
        if not SolarUtils.is_window_sunny(
            window_orientation, sun_azimuth, sun_elevation
        ):
            return 0.0

        # Use pre-defined orientation mapping for better performance
        surface_azimuth = SolarUtils.ORIENTATION_TO_AZIMUTH.get(
            window_orientation.lower(), 180  # Default to south
        )
        surface_tilt = 90  # Vertical window

        # Calculate angle of incidence
        aoi = irradiance.aoi(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=90 - sun_elevation,  # Convert elevation to zenith
            solar_azimuth=sun_azimuth,
        )

        # Calculate direct irradiance on the window surface
        if np.isscalar(aoi):
            cos_aoi = np.cos(np.radians(aoi))
        else:
            cos_aoi = np.cos(np.radians(aoi.iloc[0] if hasattr(aoi, "iloc") else aoi))

        # Direct component (only if sun is facing the window)
        direct_irradiance = dni * max(0, cos_aoi) if cos_aoi > 0 else 0

        return float(direct_irradiance)

    @staticmethod
    def get_window_sun_exposure(config, room: str) -> Dict[str, Dict[str, Any]]:
        """Determine which windows are currently exposed to direct sunlight with enhanced data"""
        solar_info = SolarUtils.get_solar_info(config)

        if "error" in solar_info:
            return {"error": solar_info["error"]}

        if not solar_info.get("is_up", False):
            return {"message": "Sun is down - no direct sunlight"}

        window_exposure = {}
        dni = solar_info.get("dni", 0)

        # Check each blind in the room
        if room in config.rooms:
            for blind in config.rooms[room].blinds:
                orientation = getattr(blind, "orientation", "south")

                # Calculate if this window orientation is getting direct sun
                is_sunny = SolarUtils.is_window_sunny(
                    orientation, solar_info["azimuth"], solar_info["elevation"]
                )

                # Calculate irradiance on window surface
                window_irradiance = SolarUtils.calculate_window_irradiance(
                    orientation, solar_info["azimuth"], solar_info["elevation"], dni
                )

                window_exposure[blind.name] = {
                    "orientation": orientation,
                    "is_sunny": is_sunny,
                    "sun_intensity": SolarUtils.calculate_sun_intensity(
                        orientation, solar_info["azimuth"], solar_info["elevation"], dni
                    ),
                    "irradiance_w_per_m2": round(window_irradiance, 1),
                    "glare_potential": (
                        "high"
                        if window_irradiance > 300
                        else "medium" if window_irradiance > 100 else "low"
                    ),
                }

        window_exposure["solar_info"] = solar_info
        return window_exposure

    @staticmethod
    def predict_solar_conditions(
        config, target_time: datetime, room: str
    ) -> Dict[str, Any]:
        """Predict solar conditions at a future time for scheduling optimization"""
        try:
            # Get coordinates and create location
            latitude, longitude = SolarUtils._get_coordinates_from_city(
                config.location.city
            )
            site_timezone = config.location.timezone or "UTC"
            altitude = getattr(config.location, "altitude", 100)

            cache_key = f"{latitude}_{longitude}_{site_timezone}_{altitude}"
            if cache_key in SolarUtils._location_cache:
                site = SolarUtils._location_cache[cache_key]
            else:
                site = location.Location(
                    latitude=latitude,
                    longitude=longitude,
                    tz=site_timezone,
                    altitude=altitude,
                )
                SolarUtils._location_cache[cache_key] = site

            # Calculate solar position at target time
            solar_position = site.get_solarposition(target_time)
            clearsky = site.get_clearsky(target_time)

            sun_azimuth = float(solar_position["azimuth"].iloc[0])
            sun_elevation = float(solar_position["apparent_elevation"].iloc[0])
            dni = float(clearsky["dni"].iloc[0]) if not clearsky.empty else 0

            # Predict conditions for each window in the room
            predictions = {}
            if room in config.rooms:
                for blind in config.rooms[room].blinds:
                    orientation = getattr(blind, "orientation", "south")

                    will_be_sunny = SolarUtils.is_window_sunny(
                        orientation, sun_azimuth, sun_elevation
                    )
                    predicted_intensity = SolarUtils.calculate_sun_intensity(
                        orientation, sun_azimuth, sun_elevation, dni
                    )
                    predicted_irradiance = SolarUtils.calculate_window_irradiance(
                        orientation, sun_azimuth, sun_elevation, dni
                    )

                    predictions[blind.name] = {
                        "will_be_sunny": will_be_sunny,
                        "predicted_intensity": predicted_intensity,
                        "predicted_irradiance": round(predicted_irradiance, 1),
                        "glare_risk": predicted_irradiance > 300,
                    }

            return {
                "target_time": target_time.strftime("%H:%M %Z"),
                "sun_elevation": sun_elevation,
                "sun_azimuth": sun_azimuth,
                "dni": dni,
                "windows": predictions,
            }

        except Exception as e:
            logger.error(f"Error predicting solar conditions: {e}")
            return {"error": f"Solar prediction failed: {e}"}
