"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
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


class BlindConfig(BaseModel):
    """Individual blind configuration"""

    id: str = Field(..., description="Device ID")
    name: str = Field(..., description="Friendly name")
    orientation: Optional[str] = Field(
        default="south", description="Window orientation: north, south, east, west"
    )


class RoomConfig(BaseModel):
    """Room configuration with blinds"""

    blinds: List[BlindConfig] = Field(..., description="List of blinds in the room")


class HubitatConfig(BaseModel):
    """Hubitat configuration"""

    rooms: Dict[str, RoomConfig] = Field(..., description="Room configurations")
    makerApiId: Optional[str] = Field(default=None, description="Maker API ID")
    accessToken: Optional[str] = Field(default=None, description="Access token")
    hubitatUrl: Optional[str] = Field(default=None, description="Hubitat hub URL")
    location: str = Field(..., description="Location description")
    latitude: Optional[float] = Field(
        default=None, description="Latitude for solar calculations"
    )
    longitude: Optional[float] = Field(
        default=None, description="Longitude for solar calculations"
    )
    timezone: Optional[str] = Field(
        default="UTC",
        description="Timezone for solar calculations (e.g., 'America/Los_Angeles')",
    )
    house_orientation: Optional[str] = Field(
        default=None, description="House orientation (e.g., 'east-west', 'north-south')"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes about the house layout"
    )


class AgentState(BaseModel):
    """Internal state model for the LangGraph agent"""

    room: str = Field(default="", description="Current room being controlled")
    target_position: int = Field(default=50, ge=0, le=100)
    messages: list = Field(default_factory=list)
    reasoning: Optional[Dict[str, Any]] = Field(default=None)
    config: Optional[HubitatConfig] = Field(default=None)


class BlindOperation(BaseModel):
    """Individual blind operation with position"""

    blind_filter: List[str] = Field(
        ..., description="Keywords to match blind names (e.g., ['side'], ['front'])"
    )
    position: int = Field(..., ge=0, le=100, description="Target position (0-100%)")
    reasoning: str = Field(..., description="Explanation for this specific blind")


class ShadeAnalysis(BaseModel):
    """Structured response from LLM for shade control analysis"""

    operations: List[BlindOperation] = Field(
        default_factory=list,
        description="List of blind operations with individual positions",
    )
    scope: Literal["specific", "room", "house"] = Field(
        ..., description="Control scope"
    )
    # Legacy support for single operation commands
    position: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Target position for single operation (0-100%)",
    )
    blind_filter: List[str] = Field(
        default_factory=list,
        description="Keywords to match blind names for single operation (empty for all)",
    )
    reasoning: str = Field(..., description="Overall explanation of the decision")


class ExecutionResult(BaseModel):
    """Result of shade control execution"""

    executed_blinds: List[str] = Field(
        default_factory=list, description="Names of blinds that were controlled"
    )
    affected_rooms: List[str] = Field(
        default_factory=list, description="Rooms that were affected"
    )
    total_blinds: int = Field(
        default=0, description="Total number of blinds controlled"
    )
    position: int = Field(..., ge=0, le=100, description="Position that was set")
    scope: str = Field(..., description="Control scope that was executed")
    reasoning: str = Field(..., description="Explanation of what was done")
