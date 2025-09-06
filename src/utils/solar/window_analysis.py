"""
Window sun exposure analysis and irradiance calculations
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from pvlib import irradiance
from .constants import SolarConstants

logger = logging.getLogger(__name__)


class WindowAnalyzer:
    """Analyzes window sun exposure and calculates irradiance"""

    @staticmethod
    def is_window_sunny(
        window_orientation: str, sun_azimuth: float, sun_elevation: float
    ) -> bool:
        """Determine if a window with given orientation is getting direct sunlight"""
        if sun_elevation < 0:  # Sun is below horizon
            return False

        orientation_lower = window_orientation.lower()
        range_data = SolarConstants.ORIENTATION_RANGES.get(orientation_lower)
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
        if not WindowAnalyzer.is_window_sunny(
            window_orientation, sun_azimuth, sun_elevation
        ):
            return "none"

        # Use DNI (Direct Normal Irradiance) if available for more accurate intensity
        if dni is not None and dni > 0:
            # DNI-based intensity (W/m²)
            if dni > SolarConstants.DNI_THRESHOLDS["high"]:
                return "high"
            elif dni > SolarConstants.DNI_THRESHOLDS["medium"]:
                return "medium"
            elif dni > SolarConstants.DNI_THRESHOLDS["low"]:
                return "low"
            elif dni > SolarConstants.DNI_THRESHOLDS["minimal"]:
                return "minimal"
            else:
                return "none"

        # Fallback to elevation-based calculation with improved thresholds
        if sun_elevation > SolarConstants.ELEVATION_THRESHOLDS["high"]:
            return "high"
        elif sun_elevation > SolarConstants.ELEVATION_THRESHOLDS["medium"]:
            return "medium"
        elif sun_elevation > SolarConstants.ELEVATION_THRESHOLDS["low"]:
            return "low"
        elif sun_elevation > SolarConstants.ELEVATION_THRESHOLDS["minimal"]:
            return "minimal"
        else:
            return "none"

    @staticmethod
    def calculate_window_irradiance(
        window_orientation: str, sun_azimuth: float, sun_elevation: float, dni: float
    ) -> float:
        """Calculate direct solar irradiance on a window surface using pvlib algorithms"""
        if not WindowAnalyzer.is_window_sunny(
            window_orientation, sun_azimuth, sun_elevation
        ):
            return 0.0

        # Use pre-defined orientation mapping for better performance
        surface_azimuth = SolarConstants.ORIENTATION_TO_AZIMUTH.get(
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
    def calculate_glare_potential(irradiance: float) -> str:
        """Determine glare potential based on window irradiance"""
        if irradiance > SolarConstants.GLARE_THRESHOLDS["high"]:
            return "high"
        elif irradiance > SolarConstants.GLARE_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"

    @staticmethod
    def analyze_window_exposure(
        config, room: str, solar_info: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze sun exposure for all windows in a room"""
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
                is_sunny = WindowAnalyzer.is_window_sunny(
                    orientation, solar_info["azimuth"], solar_info["elevation"]
                )

                # Calculate irradiance on window surface
                window_irradiance = WindowAnalyzer.calculate_window_irradiance(
                    orientation, solar_info["azimuth"], solar_info["elevation"], dni
                )

                window_exposure[blind.name] = {
                    "orientation": orientation,
                    "is_sunny": is_sunny,
                    "sun_intensity": WindowAnalyzer.calculate_sun_intensity(
                        orientation, solar_info["azimuth"], solar_info["elevation"], dni
                    ),
                    "irradiance_w_per_m2": round(window_irradiance, 1),
                    "glare_potential": WindowAnalyzer.calculate_glare_potential(
                        window_irradiance
                    ),
                }

        window_exposure["solar_info"] = solar_info
        return window_exposure

    @staticmethod
    def predict_window_conditions(
        config, target_time, room: str, solar_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict solar conditions for windows at a future time"""
        sun_azimuth = solar_data["sun_azimuth"]
        sun_elevation = solar_data["sun_elevation"]
        dni = solar_data["dni"]

        # Predict conditions for each window in the room
        predictions = {}
        if room in config.rooms:
            for blind in config.rooms[room].blinds:
                orientation = getattr(blind, "orientation", "south")

                will_be_sunny = WindowAnalyzer.is_window_sunny(
                    orientation, sun_azimuth, sun_elevation
                )
                predicted_intensity = WindowAnalyzer.calculate_sun_intensity(
                    orientation, sun_azimuth, sun_elevation, dni
                )
                predicted_irradiance = WindowAnalyzer.calculate_window_irradiance(
                    orientation, sun_azimuth, sun_elevation, dni
                )

                predictions[blind.name] = {
                    "will_be_sunny": will_be_sunny,
                    "predicted_intensity": predicted_intensity,
                    "predicted_irradiance": round(predicted_irradiance, 1),
                    "glare_risk": predicted_irradiance
                    > SolarConstants.GLARE_THRESHOLDS["high"],
                }

        return predictions
