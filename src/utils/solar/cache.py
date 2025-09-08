"""
Caching utilities for solar calculations to improve performance
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Cache TTL in seconds (5 minutes)
CACHE_TTL_SECONDS = 300


class SolarCache:
    """Manages multi-level caching for solar calculations"""

    def __init__(self):
        # Cache for geocoded coordinates to avoid repeated API calls
        self._coordinate_cache: Dict[str, tuple] = {}

        # Cache for pvlib Location objects to avoid recreation
        self._location_cache: Dict[str, Any] = {}

        # Cache for solar calculations to avoid repeated calculations
        self._solar_cache: Dict[str, Dict[str, Any]] = {}

    def get_coordinates(self, city: str) -> tuple:
        """Get cached coordinates for a city"""
        return self._coordinate_cache.get(city)

    def set_coordinates(self, city: str, coords: tuple) -> None:
        """Cache coordinates for a city"""
        self._coordinate_cache[city] = coords
        logger.debug(f"Cached coordinates for {city}: {coords}")

    def get_location(self, cache_key: str) -> Any:
        """Get cached pvlib Location object"""
        return self._location_cache.get(cache_key)

    def set_location(self, cache_key: str, location: Any) -> None:
        """Cache a pvlib Location object"""
        self._location_cache[cache_key] = location
        logger.debug(f"Cached location object: {cache_key}")

    def get_solar_data(self, cache_key: str) -> Dict[str, Any]:
        """Get cached solar calculation data"""
        cached_result = self._solar_cache.get(cache_key)
        if not cached_result:
            return None

        # Check if cache is still valid (within TTL)
        cache_time = cached_result.get("_cache_time", 0)
        current_time = datetime.now().timestamp()

        if (current_time - cache_time) < CACHE_TTL_SECONDS:
            logger.debug(f"Using cached solar data: {cache_key}")
            # Return data without internal cache metadata
            return {k: v for k, v in cached_result.items() if not k.startswith("_")}

        # Cache expired, remove it
        del self._solar_cache[cache_key]
        return None

    def set_solar_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache solar calculation data with timestamp"""
        cached_result = data.copy()
        cached_result["_cache_time"] = datetime.now().timestamp()
        self._solar_cache[cache_key] = cached_result

        # Clean old cache entries to prevent memory bloat
        self._cleanup_solar_cache()
        logger.debug(f"Cached solar data: {cache_key}")

    def _cleanup_solar_cache(self) -> None:
        """Remove expired entries from solar cache"""
        current_time = datetime.now().timestamp()
        ttl_threshold = CACHE_TTL_SECONDS * 2  # Keep entries for double TTL

        expired_keys = [
            key
            for key, value in self._solar_cache.items()
            if (current_time - value.get("_cache_time", 0)) > ttl_threshold
        ]

        for key in expired_keys:
            del self._solar_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired solar cache entries")

    def create_cache_key(self, city: str, time_rounded: datetime) -> str:
        """Create a standardized cache key for solar calculations"""
        return f"{city}_{time_rounded.isoformat()}"

    def create_location_cache_key(
        self, lat: float, lon: float, timezone: str, altitude: float
    ) -> str:
        """Create a standardized cache key for location objects"""
        return f"{lat}_{lon}_{timezone}_{altitude}"

    def clear_all(self) -> None:
        """Clear all caches (useful for testing or memory management)"""
        self._coordinate_cache.clear()
        self._location_cache.clear()
        self._solar_cache.clear()
        logger.info("Cleared all solar caches")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage"""
        return {
            "coordinates_cached": len(self._coordinate_cache),
            "locations_cached": len(self._location_cache),
            "solar_data_cached": len(self._solar_cache),
        }
