"""
Solar calculation package for the Smart Shades Agent

This package provides comprehensive solar calculations using pvlib-python
for intelligent shade automation.
"""

from .core import SolarCalculator
from .cache import SolarCache
from .window_analysis import WindowAnalyzer
from .constants import SolarConstants

# For backward compatibility, expose the main class as SolarUtils
SolarUtils = SolarCalculator

__all__ = [
    "SolarCalculator",
    "SolarCache",
    "WindowAnalyzer",
    "SolarConstants",
    "SolarUtils",  # Backward compatibility
]
