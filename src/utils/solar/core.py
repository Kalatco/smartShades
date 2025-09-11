"""
Core solar calculation utilities for sunrise/sunset scheduling
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import pytz
from pvlib import location
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from .cache import SolarCache
import pandas as pd

logger = logging.getLogger(__name__)


class SolarCalculator:
    """Main class for sunrise/sunset calculations using pvlib"""

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
            geolocator = Nominatim(user_agent="smart_shades_agent_v2")
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
    def _calculate_sunrise_sunset(site, now):
        """Calculate actual sunrise and sunset times using pvlib with proper error handling"""
        try:
            # Use the date() method which pvlib expects for sunrise/sunset calculations
            import pandas as pd

            # Create a timezone-aware pandas DatetimeIndex for the target date
            # pvlib requires timezone-aware datetime objects
            if hasattr(now, "date"):
                target_date = now.date()
            elif hasattr(now, "to_pydatetime"):
                target_date = now.to_pydatetime().date()
            else:
                # Fallback: convert to datetime and extract date
                target_date = pd.to_datetime(now).date()

            # Create a timezone-aware pandas DatetimeIndex
            # This is what pvlib actually expects
            times_index = pd.DatetimeIndex([target_date], tz=site.tz)

            # Use the timezone-aware DatetimeIndex - pvlib expects this format
            times = site.get_sun_rise_set_transit(times_index)

            # Check if we have valid data
            if times.empty or len(times) == 0:
                logger.warning("No sunrise/sunset data returned")
                return "06:00", "18:00"

            # Use .iat for safer scalar access instead of .iloc[0]
            try:
                sunrise_time = times["sunrise"].iat[0]
                sunset_time = times["sunset"].iat[0]

                # Check for NaN values
                if pd.isna(sunrise_time) or pd.isna(sunset_time):
                    logger.debug("Sunrise/sunset times are NaN, using defaults")
                    return "06:00", "18:00"

                # Convert to pandas Timestamp if needed for consistent handling
                if not isinstance(sunrise_time, pd.Timestamp):
                    sunrise_time = pd.Timestamp(sunrise_time)
                if not isinstance(sunset_time, pd.Timestamp):
                    sunset_time = pd.Timestamp(sunset_time)

            except (IndexError, ValueError, KeyError) as e:
                logger.warning(f"Error accessing sunrise/sunset values: {e}")
                return "06:00", "18:00"

            # Try multiple approaches to extract time strings
            try:
                if hasattr(sunrise_time, "strftime"):
                    sunrise_str = sunrise_time.strftime("%H:%M")
                    sunset_str = sunset_time.strftime("%H:%M")
                elif hasattr(sunrise_time, "time"):
                    sunrise_str = sunrise_time.time().strftime("%H:%M")
                    sunset_str = sunset_time.time().strftime("%H:%M")
                else:
                    # Extract from string representation with better error handling
                    sunrise_parts = str(sunrise_time).split()
                    sunset_parts = str(sunset_time).split()

                    if len(sunrise_parts) >= 2 and len(sunset_parts) >= 2:
                        sunrise_str = sunrise_parts[1][:5]
                        sunset_str = sunset_parts[1][:5]
                    else:
                        logger.debug(
                            f"Cannot parse time format: sunrise={sunrise_time}, sunset={sunset_time}"
                        )
                        sunrise_str = "06:00"
                        sunset_str = "18:00"
            except Exception as format_error:
                logger.warning(f"Time formatting error: {format_error}")
                sunrise_str = "06:00"
                sunset_str = "18:00"

            logger.debug(f"Calculated sunrise/sunset: {sunrise_str}, {sunset_str}")
            return sunrise_str, sunset_str

        except Exception as time_error:
            logger.warning(f"Sunrise/sunset calculation failed: {time_error}")
            return "06:00", "18:00"

    @staticmethod
    def get_solar_info(config) -> Dict[str, Any]:
        """Get sunrise and sunset information using pvlib with caching"""
        try:
            cache = SolarCalculator._get_cache()

            # Create cache key based on location and time (rounded to nearest hour for sunrise/sunset)
            now_rounded = datetime.now().replace(minute=0, second=0, microsecond=0)
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

            # Calculate sunrise and sunset times
            sunrise_str, sunset_str = SolarCalculator._calculate_sunrise_sunset(
                site, now
            )

            # Prepare result with only sunrise/sunset data
            result = {
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
