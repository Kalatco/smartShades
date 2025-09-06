"""
Constants for solar calculations and window orientations
"""


class SolarConstants:
    """Constants used throughout the solar calculation package"""

    # Orientation ranges for window sun exposure (azimuth angles in degrees)
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

    # Orientation to azimuth mapping for surface calculations
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

    # Compass directions for precise azimuth conversion (16 directions)
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

    # Solar calculation thresholds
    CIVIL_TWILIGHT_THRESHOLD = -0.833  # degrees below horizon
    CACHE_TTL_SECONDS = 300  # 5 minutes

    # DNI thresholds for intensity classification (W/m²)
    DNI_THRESHOLDS = {
        "high": 800,
        "medium": 500,
        "low": 200,
        "minimal": 50,
    }

    # Elevation thresholds for fallback intensity calculation (degrees)
    ELEVATION_THRESHOLDS = {
        "high": 60,
        "medium": 40,
        "low": 20,
        "minimal": 5,
    }

    # Irradiance thresholds for glare potential (W/m²)
    GLARE_THRESHOLDS = {
        "high": 300,
        "medium": 100,
    }
