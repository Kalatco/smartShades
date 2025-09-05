"""
Package initialization for models
"""

# API models (for FastAPI)
from .api import (
    ShadeControlCommand,
    ShadeStatusResponse,
    BlindInfo,
    RoomInfo,
    RoomsResponse,
    WindowExposure,
    SolarInfo,
    SolarResponse,
)

# Agent models (for LangChain/internal processing)
from .agent import (
    HouseWideDetection,
    BlindOperation,
    ShadeAnalysis,
    ExecutionResult,
)

# Configuration models
from .config import (
    BlindConfig,
    RoomConfig,
    HubitatConfig,
)

__all__ = [
    # API models
    "ShadeControlCommand",
    "ShadeStatusResponse",
    "BlindInfo",
    "RoomInfo",
    "RoomsResponse",
    "WindowExposure",
    "SolarInfo",
    "SolarResponse",
    # Agent models
    "HouseWideDetection",
    "BlindOperation",
    "ShadeAnalysis",
    "ExecutionResult",
    # Config models
    "BlindConfig",
    "RoomConfig",
    "HubitatConfig",
]
