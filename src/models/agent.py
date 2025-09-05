"""
Pydantic models for LangChain agent and internal processing
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class HouseWideDetection(BaseModel):
    """Result of house-wide command detection"""

    is_house_wide: bool = Field(
        ..., description="Whether the command is meant for the entire house"
    )


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
