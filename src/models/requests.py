"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field


class ShadeControlRequest(BaseModel):
    """Request model for shade control operations"""

    command: str = Field(..., description="Natural language command for shade control")
    room: str = Field(
        ..., description="Room identifier (master_bedroom or guest_bedroom)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context"
    )


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


class BlindConfig(BaseModel):
    """Individual blind configuration"""

    id: str = Field(..., description="Device ID")
    name: str = Field(..., description="Friendly name")


class RoomConfig(BaseModel):
    """Room configuration with blinds"""

    blinds: List[BlindConfig] = Field(..., description="List of blinds in the room")


class HubitatConfig(BaseModel):
    """Hubitat configuration"""

    rooms: Dict[str, RoomConfig] = Field(..., description="Room configurations")
    makerApiId: str = Field(..., description="Maker API ID")
    accessToken: str = Field(..., description="Access token")
    hubitatUrl: str = Field(..., description="Hubitat hub URL")
    location: str = Field(..., description="Location description")


class AgentState(BaseModel):
    """Internal state model for the LangGraph agent"""

    room: str = Field(default="", description="Current room being controlled")
    target_position: int = Field(default=50, ge=0, le=100)
    messages: list = Field(default_factory=list)
    reasoning: Optional[Dict[str, Any]] = Field(default=None)
    config: Optional[HubitatConfig] = Field(default=None)


class ShadeAnalysis(BaseModel):
    """Structured response from LLM for shade control analysis"""

    position: int = Field(..., ge=0, le=100, description="Target position (0-100%)")
    scope: Literal["specific", "room", "house"] = Field(
        ..., description="Control scope"
    )
    blind_filter: List[str] = Field(
        default_factory=list,
        description="Keywords to match blind names (empty for all)",
    )
    reasoning: str = Field(..., description="Explanation of the decision")


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
