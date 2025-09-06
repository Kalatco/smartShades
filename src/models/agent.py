"""
Pydantic models for LangChain agent and internal processing
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class HouseWideDetection(BaseModel):
    """Result of house-wide command detection"""

    is_house_wide: bool = Field(
        ..., description="Whether the command is meant for the entire house"
    )


class ExecutionTiming(BaseModel):
    """Determines if command is for immediate or scheduled execution"""

    execution_type: Literal["current", "scheduled"] = Field(
        ..., description="Whether to execute now or schedule for later"
    )
    reasoning: str = Field(..., description="Explanation of why this timing was chosen")


class ScheduleOperation(BaseModel):
    """Schedule management operation details"""

    action_type: Literal["create", "modify", "delete"] = Field(
        ..., description="Type of schedule operation to perform"
    )
    schedule_time: Optional[str] = Field(
        None, description="Time specification (e.g., '9pm', 'sunset', '08:00')"
    )
    schedule_date: Optional[str] = Field(
        None, description="Date specification (e.g., 'today', '2025-09-06', 'everyday')"
    )
    recurrence: Optional[str] = Field(
        None, description="Recurrence pattern (e.g., 'daily', 'weekdays', 'once')"
    )
    command_to_execute: str = Field(
        ..., description="The shade command to execute when scheduled"
    )
    schedule_description: str = Field(
        ..., description="Human-readable description of the schedule"
    )
    existing_schedule_id: Optional[str] = Field(
        None, description="ID of existing schedule to modify/delete"
    )
    reasoning: str = Field(
        ..., description="Explanation of the schedule operation decision"
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
