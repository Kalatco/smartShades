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
    """Solar timing information for scheduling"""

    sunrise: Optional[str] = Field(None, description="Sunrise time in local timezone")
    sunset: Optional[str] = Field(None, description="Sunset time in local timezone")
    current_time: Optional[str] = Field(
        None, description="Current time in local timezone"
    )
    timezone: Optional[str] = Field(None, description="Local timezone")


class SolarResponse(BaseModel):
    """Response for solar information endpoint"""

    room: str = Field(..., description="Room name")
    solar_data: Dict[str, Any] = Field(
        ...,
        description="Solar timing data for scheduling (sunrise/sunset only)",
    )


class ScheduleRequest(BaseModel):
    """Request model for creating or modifying a schedule"""

    command: str = Field(..., description="Natural language command for scheduling")


class ScheduleInfo(BaseModel):
    """Information about a scheduled task"""

    id: str = Field(..., description="Unique schedule ID")
    room: Optional[str] = Field(None, description="Room associated with the schedule")
    command: str = Field(..., description="Original command for the schedule")
    description: str = Field(
        ..., description="Human-readable description of the schedule"
    )
    trigger_type: str = Field(..., description="Type of trigger (cron, date, interval)")
    next_run_time: Optional[datetime] = Field(
        None, description="Next scheduled execution time"
    )
    end_date: Optional[datetime] = Field(
        None, description="End date for duration-based schedules (when they expire)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Schedule creation time"
    )
    is_active: bool = Field(
        True, description="Whether the schedule is currently active"
    )


class ScheduleResponse(BaseModel):
    """Response for schedule operations"""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    schedule: Optional[ScheduleInfo] = Field(
        None, description="Schedule information (for create/modify)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class ScheduleListResponse(BaseModel):
    """Response for listing all schedules"""

    schedules: List[ScheduleInfo] = Field(
        ..., description="List of all active schedules"
    )
    total_count: int = Field(..., description="Total number of schedules")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
