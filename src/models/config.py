"""
Pydantic models for configuration data
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field


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


class LocationConfig(BaseModel):
    """Location configuration"""

    city: str = Field(..., description="City for solar calculations")
    timezone: Optional[str] = Field(
        default="UTC",
        description="Timezone for solar calculations (e.g., 'America/Los_Angeles')",
    )


class HouseInformationConfig(BaseModel):
    """House-specific information"""

    orientation: Optional[str] = Field(
        default=None, description="House orientation (e.g., 'east-west', 'north-south')"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes about the house layout"
    )


class HubitatConfig(BaseModel):
    """Hubitat configuration"""

    rooms: Dict[str, RoomConfig] = Field(..., description="Room configurations")
    makerApiId: Optional[str] = Field(default=None, description="Maker API ID")
    accessToken: Optional[str] = Field(default=None, description="Access token")
    hubitatUrl: Optional[str] = Field(default=None, description="Hubitat hub URL")
    location: LocationConfig = Field(..., description="Location information")
    houseInformation: HouseInformationConfig = Field(
        ..., description="House-specific information"
    )
