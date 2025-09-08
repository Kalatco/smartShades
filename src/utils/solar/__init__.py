"""
Solar calculation package for the Smart Shades Agent

This package provides sunrise/sunset calculations using pvlib-python
for intelligent shade scheduling.
"""

from .core import SolarCalculator
from .cache import SolarCache

# For backward compatibility, expose the main class as SolarUtils
SolarUtils = SolarCalculator

__all__ = [
    "SolarCalculator",
    "SolarCache",
    "SolarUtils",  # Backward compatibility
]
