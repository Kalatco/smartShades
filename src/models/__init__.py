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
    ScheduleRequest,
    ScheduleInfo,
    ScheduleResponse,
    ScheduleListResponse,
)

# Agent models (for LangChain/internal processing)
from .agent import (
    HouseWideDetection,
    ExecutionTiming,
    ScheduleOperation,
    DurationInfo,
    BlindOperation,
    ShadeAnalysis,
    ExecutionResult,
    RoomBlindsExecution,
    BlindExecutionRequest,
    BlindExecutionResult,
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
    "ScheduleRequest",
    "ScheduleInfo",
    "ScheduleResponse",
    "ScheduleListResponse",
    # Agent models
    "HouseWideDetection",
    "ExecutionTiming",
    "ScheduleOperation",
    "DurationInfo",
    "BlindOperation",
    "ShadeAnalysis",
    "ExecutionResult",
    "RoomBlindsExecution",
    "BlindExecutionRequest",
    "BlindExecutionResult",
    # Config models
    "BlindConfig",
    "RoomConfig",
    "HubitatConfig",
]
