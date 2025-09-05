"""
Pydantic models for FastAPI requests and responses
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ShadeControlCommand(BaseModel):
    """Request model for simple shade control commands"""

    command: str = Field(..., description="Natural language command for shade control")


class ShadeStatusResponse(BaseModel):
    """Response model for shade status and control operations"""

    success: bool = Field(..., description="Whether the operation was successful")
    position: int = Field(
        ..., ge=0, le=100, description="Current shade position (0-100%)"
    )
    message: str = Field(..., description="Human-readable status or result message")
    room: str = Field(..., description="Room that was controlled")
    affected_blinds: List[str] = Field(
        default_factory=list, description="List of blind names that were affected"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now, description="Timestamp of the status or operation"
    )


class BlindInfo(BaseModel):
    """Information about a specific blind"""

    id: str = Field(..., description="Device ID")
    name: str = Field(..., description="Friendly name")
    orientation: Optional[str] = Field(
        default="south", description="Window orientation: north, south, east, west"
    )


class RoomInfo(BaseModel):
    """Room information response"""

    blind_count: int = Field(..., description="Number of blinds in the room")
    blinds: List[BlindInfo] = Field(..., description="List of blinds in the room")


class RoomsResponse(BaseModel):
    """Response for available rooms endpoint"""

    rooms: Dict[str, RoomInfo] = Field(
        ..., description="Dictionary of room configurations"
    )


class WindowExposure(BaseModel):
    """Sun exposure information for a specific window"""

    orientation: str = Field(..., description="Window orientation")
    is_sunny: bool = Field(
        ..., description="Whether the window is receiving direct sunlight"
    )
    sun_intensity: float = Field(
        ..., ge=0, le=1, description="Sun intensity factor (0-1)"
    )


class SolarInfo(BaseModel):
    """Solar position and timing information"""

    is_up: bool = Field(..., description="Whether the sun is currently up")
    azimuth: Optional[float] = Field(None, description="Sun azimuth angle in degrees")
    elevation: Optional[float] = Field(
        None, description="Sun elevation angle in degrees"
    )
    direction: Optional[str] = Field(None, description="Cardinal direction of the sun")
    sunrise: Optional[str] = Field(None, description="Sunrise time in local timezone")
    sunset: Optional[str] = Field(None, description="Sunset time in local timezone")


class SolarResponse(BaseModel):
    """Response for solar information endpoint"""

    room: str = Field(..., description="Room name")
    solar_data: Dict[str, Any] = Field(
        ...,
        description="Solar exposure data including window analysis and solar position",
    )
