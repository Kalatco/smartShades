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
    ScheduleRequest,
    ScheduleInfo,
    ScheduleResponse,
    ScheduleListResponse,
)

# Agent models (for LangChain/internal processing)
from .agent import (
    ExecutionTiming,
    ScheduleOperation,
    DurationInfo,
    BlindOperation,
    ShadeAnalysis,
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
    "ScheduleRequest",
    "ScheduleInfo",
    "ScheduleResponse",
    "ScheduleListResponse",
    # Agent models
    "ExecutionTiming",
    "ScheduleOperation",
    "DurationInfo",
    "BlindOperation",
    "ShadeAnalysis",
    "RoomBlindsExecution",
    "BlindExecutionRequest",
    "BlindExecutionResult",
    # Config models
    "BlindConfig",
    "RoomConfig",
    "HubitatConfig",
]
