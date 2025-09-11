"""
Smart Shades Agent V2 implementation using LangGraph
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Literal

from langgraph.graph import StateGraph, END

from models.agent import (
    ExecutionTiming,
    ScheduleOperation,
    BlindExecutionRequest,
    BlindExecutionResult,
)
from chains.execution_timing import ExecutionTimingChain
from chains.schedule_management import ScheduleManagementChain
from chains.blind_execution_planning_v2 import BlindExecutionPlanningChain
from utils.config_utils import ConfigManager
from utils.smart_scheduler import SmartScheduler
from utils.agent.smart_shades.execution_utils_v2 import ExecutionUtilsV2

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent"""

    command: str
    room: str
    context: Optional[Dict[str, Any]]
    execution_timing: Optional[ExecutionTiming]
    schedule_operation: Optional[ScheduleOperation]
    blind_execution_request: Optional[BlindExecutionRequest]
    blind_execution_result: Optional[BlindExecutionResult]
    final_response: Optional[Dict[str, Any]]
    error: Optional[str]


class SmartShadesAgentV2:
    """LangGraph-based agent for intelligent shade control with v2 execution"""

    def __init__(self):
        self.llm = None
        self.config = None
        self.execution_timing_chain = None
        self.schedule_management_chain = None
        self.blind_execution_planning_chain = None
        self.scheduler = None
        self.graph = None

    async def initialize(self):
        """Initialize the agent and LangGraph components"""
        # Load environment variables
        ConfigManager.load_environment()

        # Validate environment
        if not ConfigManager.validate_environment():
            raise ValueError("Required environment variables are not set")

        # Load configuration
        self.config = await ConfigManager.load_blinds_config()

        # Override config with environment variables for Hubitat
        self.config = ConfigManager.override_hubitat_config(self.config)

        # Initialize LLM
        self.llm = ConfigManager.create_azure_llm()

        # Initialize chains
        self.execution_timing_chain = ExecutionTimingChain(self.llm)
        self.schedule_management_chain = ScheduleManagementChain(self.llm)
        self.blind_execution_planning_chain = BlindExecutionPlanningChain(self.llm)

        # Initialize scheduler
        self.scheduler = SmartScheduler(agent_instance=self)
        self.scheduler.set_config(self.config)
        await self.scheduler.start()

        # Build the LangGraph
        self._build_graph()

        # Log configuration summary
        config_summary = ConfigManager.get_config_summary(self.config)
        logger.info(f"Smart Shades Agent V2 initialized successfully: {config_summary}")

    def _build_graph(self):
        """Build the LangGraph workflow"""
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("execution_timing", self._execution_timing_node)
        workflow.add_node("schedule_management", self._schedule_management_node)
        workflow.add_node(
            "blind_execution_planning", self._blind_execution_planning_node
        )
        workflow.add_node("execute_blinds", self._execute_blinds_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Set entry point
        workflow.set_entry_point("execution_timing")

        # Add conditional edges from execution_timing
        workflow.add_conditional_edges(
            "execution_timing",
            self._route_after_timing,
            {
                "schedule": "schedule_management",
                "execute": "blind_execution_planning",
                "error": "error_handler",
            },
        )

        # Add edges from schedule_management
        workflow.add_conditional_edges(
            "schedule_management",
            self._route_after_schedule,
            {
                "end": END,
                "error": "error_handler",
            },
        )

        # Add edges from blind_execution_planning
        workflow.add_edge("blind_execution_planning", "execute_blinds")

        # Add edges to END
        workflow.add_edge("execute_blinds", END)
        workflow.add_edge("error_handler", END)

        # Compile the graph
        self.graph = workflow.compile()

    async def _execution_timing_node(self, state: AgentState) -> AgentState:
        """Entry node: Determine execution timing"""
        try:
            logger.info(f"Analyzing execution timing for command: {state['command']}")

            timing = await self.execution_timing_chain.ainvoke(
                {"command": state["command"]}
            )

            state["execution_timing"] = timing
            logger.info(f"Execution timing determined: {timing.execution_type}")

        except Exception as e:
            logger.error(f"Error in execution timing: {e}")
            state["error"] = f"Error determining execution timing: {e}"

        return state

    async def _schedule_management_node(self, state: AgentState) -> AgentState:
        """Handle schedule creation/management"""
        try:
            logger.info(f"Processing schedule management for: {state['command']}")

            # Get existing schedules for context
            existing_schedules = (
                self.scheduler.get_schedules(state["room"]) if self.scheduler else []
            )

            # Analyze the schedule request
            schedule_op = await self.schedule_management_chain.ainvoke(
                {
                    "command": state["command"],
                    "room": state["room"],
                    "existing_schedules": existing_schedules,
                }
            )

            state["schedule_operation"] = schedule_op
            logger.info(f"Schedule operation created: {schedule_op.action_type}")

            # Execute the schedule operation
            if schedule_op.action_type == "create":
                result = await self.scheduler.create_schedule(
                    schedule_op, state["room"]
                )
            elif schedule_op.action_type == "modify":
                result = await self.scheduler.modify_schedule(
                    schedule_op, state["room"]
                )
            elif schedule_op.action_type == "delete":
                result = await self.scheduler.delete_schedule(
                    schedule_op.existing_schedule_id
                )
            else:
                raise ValueError(f"Unknown schedule action: {schedule_op.action_type}")

            # Store schedule result for later use
            state["context"] = state.get("context", {})
            state["context"]["schedule_result"] = result

            # Create final response for scheduled commands (no immediate execution)
            state["final_response"] = {
                "message": f"Schedule created: {schedule_op.schedule_description}",
                "room": state["room"],
                "schedule_id": result.get("job_id"),
                "next_run": result.get("next_run"),
                "schedule_description": schedule_op.schedule_description,
                "operation": "schedule_created",
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error in schedule management: {e}")
            state["error"] = f"Error processing schedule: {e}"

        return state

    async def _blind_execution_planning_node(self, state: AgentState) -> AgentState:
        """Plan blind execution using V2 chain"""
        try:
            logger.info(f"Planning blind execution for: {state['command']}")

            # For scheduled operations, we need to execute the actual blind command
            command_to_execute = state["command"]
            if state.get("schedule_operation"):
                # Use the command from the schedule operation
                command_to_execute = state["schedule_operation"].command_to_execute

            # Plan the execution
            execution_request = await self.blind_execution_planning_chain.ainvoke(
                {
                    "command": command_to_execute,
                    "current_room": state["room"],
                    "config": self.config,
                }
            )

            state["blind_execution_request"] = execution_request
            logger.info(
                f"Blind execution planned for {len(execution_request.rooms)} room(s)"
            )

        except Exception as e:
            logger.error(f"Error in blind execution planning: {e}")
            state["error"] = f"Error planning blind execution: {e}"

        return state

    async def _execute_blinds_node(self, state: AgentState) -> AgentState:
        """Execute blind operations using V2 utils"""
        try:
            logger.info("Executing blind operations")

            if not state.get("blind_execution_request"):
                raise ValueError("No blind execution request available")

            # Execute the blinds using V2 utils
            execution_result = await ExecutionUtilsV2.execute_blinds(
                self.config, state["blind_execution_request"]
            )

            state["blind_execution_result"] = execution_result

            # Build final response
            if (
                state.get("execution_timing")
                and state["execution_timing"].execution_type == "scheduled"
            ):
                # Scheduled execution response
                schedule_result = state.get("context", {}).get("schedule_result", {})
                state["final_response"] = {
                    "message": f"Schedule created and executed: {state['schedule_operation'].schedule_description}",
                    "room": state["room"],
                    "schedule_id": schedule_result.get("job_id"),
                    "next_run": schedule_result.get("next_run"),
                    "execution_result": {
                        "successful_blinds": execution_result.successful_blinds,
                        "total_successful": execution_result.total_successful,
                        "execution_summary": execution_result.execution_summary,
                    },
                    "operation": "scheduled_execution",
                    "timestamp": datetime.now(),
                }
            else:
                # Current execution response
                state["final_response"] = {
                    "message": execution_result.execution_summary,
                    "room": state["room"],
                    "successful_blinds": execution_result.successful_blinds,
                    "failed_blinds": execution_result.failed_blinds,
                    "total_attempted": execution_result.total_attempted,
                    "total_successful": execution_result.total_successful,
                    "operation": "current_execution",
                    "timestamp": datetime.now(),
                }

            logger.info(
                f"Blind execution completed: {execution_result.total_successful}/{execution_result.total_attempted} successful"
            )

        except Exception as e:
            logger.error(f"Error executing blinds: {e}")
            state["error"] = f"Error executing blinds: {e}"

        return state

    async def _error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors and create error response"""
        error_message = state.get("error", "Unknown error occurred")
        logger.error(f"Handling error: {error_message}")

        state["final_response"] = {
            "error": error_message,
            "room": state.get("room", "unknown"),
            "operation": "error",
            "timestamp": datetime.now(),
        }

        return state

    def _route_after_timing(
        self, state: AgentState
    ) -> Literal["schedule", "execute", "error"]:
        """Route after execution timing determination"""
        if state.get("error"):
            return "error"

        timing = state.get("execution_timing")
        if not timing:
            return "error"

        if timing.execution_type == "scheduled":
            return "schedule"
        else:
            return "execute"

    def _route_after_schedule(self, state: AgentState) -> Literal["end", "error"]:
        """Route after schedule management - should end for scheduled commands"""
        if state.get("error"):
            return "error"
        else:
            return "end"

    async def process_request(
        self, command: str, room: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user request through the LangGraph pipeline"""
        try:
            # Validate room
            if not self._validate_room(room):
                return self._create_error_response(
                    f"Invalid room: {room}. Available rooms: {list(self.config.rooms.keys())}",
                    room,
                )

            # Create initial state
            initial_state = AgentState(
                command=command,
                room=room,
                context=context or {},
                execution_timing=None,
                schedule_operation=None,
                blind_execution_request=None,
                blind_execution_result=None,
                final_response=None,
                error=None,
            )

            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state)

            # Return the final response
            return final_state.get(
                "final_response",
                self._create_error_response("No response generated", room),
            )

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return self._create_error_response(f"Error processing command: {e}", room)

    def _validate_room(self, room: str) -> bool:
        """Validate that a room exists in the configuration"""
        return bool(room) and room in self.config.rooms

    def _create_error_response(self, error_message: str, room: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "error": error_message,
            "room": room,
            "operation": "error",
            "timestamp": datetime.now(),
        }

    async def shutdown(self):
        """Shutdown the agent and cleanup resources"""
        if self.scheduler:
            await self.scheduler.shutdown()
        logger.info("Smart Shades Agent V2 shutdown completed")

    def get_schedules(self, room: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all schedules, optionally filtered by room"""
        if self.scheduler:
            return self.scheduler.get_schedules(room)
        return []

    async def get_current_status(self, room: str) -> Dict[str, Any]:
        """Get current shade status for a room"""
        try:
            if not self._validate_room(room):
                return self._create_error_response(f"Invalid room: {room}", room)

            # Use ExecutionUtilsV2 to get current status
            current_positions = await ExecutionUtilsV2.get_room_current_positions(
                self.config, room
            )

            return {
                "room": room,
                "current_positions": current_positions,
                "timestamp": datetime.now(),
                "operation": "status_check",
            }

        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return self._create_error_response(f"Error getting status: {e}", room)
