"""
Core solar calculation utilities using pvlib-python
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import pytz
from pvlib import location
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from .cache import SolarCache
from .window_analysis import WindowAnalyzer
from .constants import SolarConstants
import pandas as pd

logger = logging.getLogger(__name__)


class SolarCalculator:
    """Main class for solar calculations and window sun exposure using pvlib"""

    # Class-level cache instance for static methods
    _cache = None

    @classmethod
    def _get_cache(cls):
        """Get or create the shared cache instance"""
        if cls._cache is None:
            cls._cache = SolarCache()
        return cls._cache

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
        cache = SolarCalculator._get_cache()

        # Check cache first
        cached_coords = cache.get_coordinates(city)
        if cached_coords:
            logger.debug(f"Using cached coordinates for {city}")
            return cached_coords

        try:
            # Use Nominatim (free OpenStreetMap geocoding service)
            geolocator = Nominatim(user_agent="smart_shades_agent")
            location_result = geolocator.geocode(city, timeout=10)

            if location_result:
                coords = (location_result.latitude, location_result.longitude)
                # Cache the result
                cache.set_coordinates(city, coords)
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
    def _get_or_create_site(
        latitude: float, longitude: float, site_timezone: str, altitude: float
    ):
        """Get cached pvlib Location object or create a new one"""
        cache = SolarCalculator._get_cache()
        location_cache_key = cache.create_location_cache_key(
            latitude, longitude, site_timezone, altitude
        )

        # Check cache first
        site = cache.get_location(location_cache_key)
        if site:
            return site

        # Create new pvlib Location object
        site = location.Location(
            latitude=latitude,
            longitude=longitude,
            tz=site_timezone,
            altitude=altitude,
            name=f"Smart Shades Location",
        )

        # Cache it
        cache.set_location(location_cache_key, site)
        return site

    @staticmethod
    def _calculate_solar_position(site, now):
        """Calculate solar position with error handling"""
        try:
            # Ensure now is timezone-aware for pandas operations
            if hasattr(now, "tz_localize") or hasattr(now, "tz_convert"):
                # Already pandas timestamp
                times_index = now
            else:
                # Convert datetime to pandas timestamp with timezone
                if now.tzinfo is None:
                    # No timezone, localize to site timezone
                    times_index = pd.to_datetime(now).tz_localize(site.tz)
                else:
                    # Has timezone, convert to pandas timestamp
                    times_index = pd.to_datetime(now)

            solar_position = site.get_solarposition(times_index)
            return {
                "azimuth": float(solar_position["azimuth"].iloc[0]),
                "elevation_true": float(solar_position["elevation"].iloc[0]),
                "elevation_apparent": float(
                    solar_position["apparent_elevation"].iloc[0]
                ),
            }
        except Exception as solar_error:
            logger.error(f"Solar position calculation failed: {solar_error}")
            # Use fallback values
            return {
                "azimuth": 180.0,  # South
                "elevation_true": 30.0,
                "elevation_apparent": 30.0,
            }

    @staticmethod
    def _calculate_dni(site, now):
        """Calculate Direct Normal Irradiance with error handling"""
        try:
            # Ensure now is timezone-aware and convert to pandas timestamp
            if hasattr(now, "tz_localize") or hasattr(now, "tz_convert"):
                # Already pandas timestamp
                times_index = now
            else:
                # Convert datetime to pandas timestamp with timezone
                import pandas as pd

                if now.tzinfo is None:
                    # No timezone, localize to site timezone
                    times_index = pd.to_datetime(now).tz_localize(site.tz)
                else:
                    # Has timezone, convert to pandas timestamp
                    times_index = pd.to_datetime(now)

            clearsky = site.get_clearsky(times_index)
            return float(clearsky["dni"].iloc[0]) if not clearsky.empty else 0
        except Exception as clearsky_error:
            logger.warning(f"Clearsky calculation failed: {clearsky_error}")
            return 0

    @staticmethod
    def _calculate_sunrise_sunset(site, now):
        """Calculate sunrise and sunset times with comprehensive error handling"""
        try:
            # Convert datetime to pandas timestamp with proper timezone handling
            import pandas as pd

            if hasattr(now, "date"):
                target_date = now.date()
            else:
                target_date = now

            # Create timezone-aware pandas timestamp for the date
            if hasattr(target_date, "year"):
                date_str = target_date.strftime("%Y-%m-%d")
            else:
                date_str = str(target_date)

            # Create pandas timestamp and localize to site timezone
            target_datetime = pd.to_datetime(date_str).tz_localize(site.tz)

            times = site.get_sun_rise_set_transit(target_datetime)
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

            return sunrise_str, sunset_str

        except Exception as time_error:
            logger.warning(f"Sunrise/sunset calculation failed: {time_error}")
            return "06:00", "18:00"

    @staticmethod
    def _azimuth_to_direction(sun_azimuth: float) -> str:
        """Convert azimuth to cardinal direction with better precision"""
        direction_idx = int((sun_azimuth + 11.25) // 22.5) % 16
        return SolarConstants.COMPASS_DIRECTIONS[direction_idx]

    @staticmethod
    def get_solar_info(config) -> Dict[str, Any]:
        """Get current solar position and sun-related information using pvlib with caching"""
        try:
            cache = SolarCalculator._get_cache()

            # Create cache key based on location and time (rounded to nearest 5 minutes)
            now_rounded = datetime.now().replace(
                minute=(datetime.now().minute // 5) * 5, second=0, microsecond=0
            )
            cache_key = cache.create_cache_key(config.location.city, now_rounded)

            # Check solar calculation cache first
            cached_result = cache.get_solar_data(cache_key)
            if cached_result:
                return cached_result

            # Get coordinates from city
            latitude, longitude = SolarCalculator._get_coordinates_from_city(
                config.location.city
            )

            # Get or create pvlib Location object
            site_timezone = config.location.timezone or "UTC"
            altitude = getattr(config.location, "altitude", 100)
            site = SolarCalculator._get_or_create_site(
                latitude, longitude, site_timezone, altitude
            )

            # Get current time in configured timezone
            tz, now = SolarCalculator._get_timezone_and_now(config)

            # Calculate solar position
            solar_pos = SolarCalculator._calculate_solar_position(site, now)

            # Calculate DNI
            dni = SolarCalculator._calculate_dni(site, now)

            # Calculate sunrise and sunset times
            sunrise_str, sunset_str = SolarCalculator._calculate_sunrise_sunset(
                site, now
            )

            # Determine if sun is up using apparent elevation (accounts for refraction)
            is_sun_up = (
                solar_pos["elevation_apparent"]
                > SolarConstants.CIVIL_TWILIGHT_THRESHOLD
            )

            # Convert azimuth to cardinal direction
            sun_direction = SolarCalculator._azimuth_to_direction(solar_pos["azimuth"])

            # Prepare result
            result = {
                "azimuth": solar_pos["azimuth"],
                "elevation": solar_pos["elevation_apparent"],  # Use apparent elevation
                "elevation_true": solar_pos[
                    "elevation_true"
                ],  # Also provide true elevation
                "direction": sun_direction,
                "is_up": is_sun_up,
                "dni": dni,  # Direct Normal Irradiance for advanced calculations
                "sunrise": f"{sunrise_str} {site_timezone}",
                "sunset": f"{sunset_str} {site_timezone}",
                "current_time": now.strftime("%H:%M %Z"),
                "timezone": site_timezone,
                "coordinates": {"lat": latitude, "lon": longitude, "alt": altitude},
            }

            # Cache the result
            cache.set_solar_data(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error calculating solar info: {e}")
            return {"error": f"Solar calculation failed: {e}"}

    @staticmethod
    def get_window_sun_exposure(config, room: str) -> Dict[str, Dict[str, Any]]:
        """Determine which windows are currently exposed to direct sunlight with enhanced data"""
        solar_info = SolarCalculator.get_solar_info(config)
        return WindowAnalyzer.analyze_window_exposure(config, room, solar_info)

    @staticmethod
    def predict_solar_conditions(
        config, target_time: datetime, room: str
    ) -> Dict[str, Any]:
        """Predict solar conditions at a future time for scheduling optimization"""
        try:
            # Get coordinates and create location
            latitude, longitude = SolarCalculator._get_coordinates_from_city(
                config.location.city
            )
            site_timezone = config.location.timezone or "UTC"
            altitude = getattr(config.location, "altitude", 100)

            site = SolarCalculator._get_or_create_site(
                latitude, longitude, site_timezone, altitude
            )

            # Calculate solar position at target time
            solar_position = site.get_solarposition(target_time)
            clearsky = site.get_clearsky(target_time)

            sun_azimuth = float(solar_position["azimuth"].iloc[0])
            sun_elevation = float(solar_position["apparent_elevation"].iloc[0])
            dni = float(clearsky["dni"].iloc[0]) if not clearsky.empty else 0

            # Create solar data for window analysis
            solar_data = {
                "sun_azimuth": sun_azimuth,
                "sun_elevation": sun_elevation,
                "dni": dni,
            }

            # Predict conditions for windows
            window_predictions = WindowAnalyzer.predict_window_conditions(
                config, target_time, room, solar_data
            )

            return {
                "target_time": target_time.strftime("%H:%M %Z"),
                "sun_elevation": sun_elevation,
                "sun_azimuth": sun_azimuth,
                "dni": dni,
                "windows": window_predictions,
            }

        except Exception as e:
            logger.error(f"Error predicting solar conditions: {e}")
            return {"error": f"Solar prediction failed: {e}"}

    # Backward compatibility methods - delegate to WindowAnalyzer
    @staticmethod
    def is_window_sunny(
        window_orientation: str, sun_azimuth: float, sun_elevation: float
    ) -> bool:
        """Backward compatibility wrapper"""
        return WindowAnalyzer.is_window_sunny(
            window_orientation, sun_azimuth, sun_elevation
        )

    @staticmethod
    def calculate_sun_intensity(
        window_orientation: str, sun_azimuth: float, sun_elevation: float, dni=None
    ) -> str:
        """Backward compatibility wrapper"""
        return WindowAnalyzer.calculate_sun_intensity(
            window_orientation, sun_azimuth, sun_elevation, dni
        )

    @staticmethod
    def calculate_window_irradiance(
        window_orientation: str, sun_azimuth: float, sun_elevation: float, dni: float
    ) -> float:
        """Backward compatibility wrapper"""
        return WindowAnalyzer.calculate_window_irradiance(
            window_orientation, sun_azimuth, sun_elevation, dni
        )
