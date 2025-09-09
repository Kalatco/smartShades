"""
Smart scheduler using APScheduler for automated shade control
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, Any, List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.job import Job
import asyncio
import re
from utils.solar import SolarUtils
from models.agent import ScheduleOperation

logger = logging.getLogger(__name__)

# Global registry for job execution functions (needed for pickle serialization)
_job_registry = {}


def register_job_function(name: str, func):
    """Register a function for job execution"""
    _job_registry[name] = func


def get_job_function(name: str):
    """Get a registered job function"""
    return _job_registry.get(name)


async def execute_scheduled_shade_command(agent_instance, room: str, command: str):
    """Execute a scheduled shade command through the agent"""
    try:
        logger.info(f"Executing scheduled command for {room}: {command}")

        # Process the command through the normal agent flow
        result = await agent_instance.process_request(command, room)

        logger.info(f"Scheduled command completed: {result.get('message', 'Success')}")
        return result

    except Exception as e:
        logger.error(f"Error executing scheduled command: {e}")
        return {"error": f"Scheduled execution failed: {e}"}


class SmartScheduler:
    """Advanced scheduler for automated shade control with APScheduler"""

    def __init__(self, agent_instance=None):
        self.agent = agent_instance

        # Configure APScheduler
        self.scheduler = AsyncIOScheduler(
            jobstores={"default": MemoryJobStore()},
            executors={"default": AsyncIOExecutor()},
            job_defaults={
                "coalesce": False,
                "max_instances": 3,
                "misfire_grace_time": 30,
            },
        )

        # Register the execution function
        register_job_function(
            "execute_scheduled_shade_command", execute_scheduled_shade_command
        )

        self.config = None

    async def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("SmartScheduler started")

    async def shutdown(self):
        """Shutdown the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("SmartScheduler shutdown")

    def set_config(self, config):
        """Set the configuration for solar calculations"""
        self.config = config

    async def create_schedule(
        self, schedule_op: ScheduleOperation, room: str
    ) -> Dict[str, Any]:
        """Create a new scheduled job"""
        try:
            # Parse the schedule timing
            trigger = await self._parse_schedule_trigger(
                schedule_op.schedule_time,
                schedule_op.schedule_date,
                schedule_op.recurrence,
            )

            if not trigger:
                return {"error": "Could not parse schedule timing"}

            # Create job ID
            job_id = self._generate_job_id(schedule_op, room)

            # Add the job
            job = self.scheduler.add_job(
                func=execute_scheduled_shade_command,
                trigger=trigger,
                args=[self.agent, room, schedule_op.command_to_execute],
                id=job_id,
                name=schedule_op.schedule_description,
                replace_existing=False,
            )

            logger.info(
                f"Created schedule: {job_id} - {schedule_op.schedule_description}"
            )

            return {
                "success": True,
                "job_id": job_id,
                "description": schedule_op.schedule_description,
                "next_run": (
                    job.next_run_time.isoformat() if job.next_run_time else None
                ),
            }

        except Exception as e:
            logger.error(f"Error creating schedule: {e}")
            return {"error": f"Failed to create schedule: {e}"}

    async def modify_schedule(
        self, schedule_op: ScheduleOperation, room: str
    ) -> Dict[str, Any]:
        """Modify an existing scheduled job"""
        try:
            # Use only the explicit schedule ID from the operation
            job_id = schedule_op.existing_schedule_id

            if not job_id:
                # No existing job ID provided, create new schedule instead
                return await self.create_schedule(schedule_op, room)

            # Parse new trigger
            trigger = await self._parse_schedule_trigger(
                schedule_op.schedule_time,
                schedule_op.schedule_date,
                schedule_op.recurrence,
            )

            if not trigger:
                return {"error": "Could not parse new schedule timing"}

            # Modify the job
            job = self.scheduler.modify_job(
                job_id=job_id,
                trigger=trigger,
                args=[self.agent, room, schedule_op.command_to_execute],
                name=schedule_op.schedule_description,
            )

            logger.info(
                f"Modified schedule: {job_id} - {schedule_op.schedule_description}"
            )

            return {
                "success": True,
                "job_id": job_id,
                "description": schedule_op.schedule_description,
                "next_run": (
                    job.next_run_time.isoformat() if job.next_run_time else None
                ),
                "action": "modified",
            }

        except Exception as e:
            logger.error(f"Error modifying schedule: {e}")
            return {"error": f"Failed to modify schedule: {e}"}

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule by its ID directly"""
        try:
            # Check if the job exists
            job = self.scheduler.get_job(schedule_id)
            if not job:
                logger.warning(f"Schedule {schedule_id} not found")
                return False

            # Remove the job
            self.scheduler.remove_job(schedule_id)
            logger.info(f"Deleted schedule by ID: {schedule_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting schedule {schedule_id}: {e}")
            return False

    def get_schedules(self, room: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all scheduled jobs, optionally filtered by room"""
        schedules = []

        for job in self.scheduler.get_jobs():
            # Extract room from job args if available
            job_room = None
            if len(job.args) >= 2:
                job_room = job.args[1]

            # Filter by room if specified
            if room and job_room != room:
                continue

            schedule_info = {
                "id": job.id,
                "name": job.name,
                "room": job_room,
                "next_run_time": (
                    job.next_run_time.isoformat() if job.next_run_time else None
                ),
                "trigger": str(job.trigger),
                "command": job.args[2] if len(job.args) >= 3 else "Unknown command",
            }
            schedules.append(schedule_info)

        return schedules

    def get_all_schedules(self) -> Dict[str, Dict[str, Any]]:
        """Get all scheduled jobs as a dictionary with job IDs as keys"""
        schedules = {}

        for job in self.scheduler.get_jobs():
            # Extract room from job args if available
            job_room = None
            job_command = "Unknown command"
            if len(job.args) >= 2:
                job_room = job.args[1]
            if len(job.args) >= 3:
                job_command = job.args[2]

            schedule_info = {
                "room": job_room,
                "command": job_command,
                "description": job.name or f"Schedule for {job_room}",
                "trigger_type": str(type(job.trigger).__name__)
                .lower()
                .replace("trigger", ""),
                "next_run_time": job.next_run_time,
                "created_at": datetime.now(),  # APScheduler doesn't track creation time
                "is_active": True,
            }
            schedules[job.id] = schedule_info

        return schedules

    async def _parse_schedule_trigger(
        self, schedule_time: str, schedule_date: str, recurrence: str
    ):
        """Parse schedule parameters into APScheduler trigger"""
        try:
            current_time = datetime.now()

            # Handle recurrence patterns
            if recurrence and recurrence.lower() in ["daily", "everyday"]:
                # Daily recurring schedule
                hour, minute = await self._parse_time(schedule_time, current_time)
                return CronTrigger(hour=hour, minute=minute)

            elif recurrence and recurrence.lower() == "weekdays":
                # Weekday recurring schedule
                hour, minute = await self._parse_time(schedule_time, current_time)
                return CronTrigger(hour=hour, minute=minute, day_of_week="mon-fri")

            elif recurrence and recurrence.lower() == "weekends":
                # Weekend recurring schedule
                hour, minute = await self._parse_time(schedule_time, current_time)
                return CronTrigger(hour=hour, minute=minute, day_of_week="sat-sun")

            elif recurrence and recurrence.lower() == "weekly":
                # Weekly recurring schedule
                hour, minute = await self._parse_time(schedule_time, current_time)
                weekday = current_time.weekday()  # Use current day of week
                return CronTrigger(hour=hour, minute=minute, day_of_week=weekday)

            else:
                # One-time schedule
                target_datetime = await self._parse_datetime(
                    schedule_time, schedule_date, current_time
                )
                return DateTrigger(run_date=target_datetime)

        except Exception as e:
            logger.error(f"Error parsing schedule trigger: {e}")
            return None

    async def _parse_time(self, time_str: str, reference_time: datetime) -> tuple:
        """Parse time string into hour and minute"""
        if not time_str:
            return reference_time.hour, reference_time.minute

        # Handle solar times
        if time_str.lower() in ["sunrise", "sunset"]:
            return await self._get_solar_time(time_str.lower(), reference_time)

        # Handle relative times
        if "+" in time_str:
            base_time, offset = time_str.split("+", 1)
            base_hour, base_minute = await self._parse_time(base_time, reference_time)
            offset_minutes = self._parse_time_offset(offset)

            total_minutes = base_hour * 60 + base_minute + offset_minutes
            return (total_minutes // 60) % 24, total_minutes % 60

        # Handle standard time formats
        time_patterns = [
            r"^(\d{1,2}):(\d{2})$",  # 14:30, 9:00
            r"^(\d{1,2})([ap]m)$",  # 9pm, 2am
            r"^(\d{1,2})\s*([ap]m)$",  # 9 pm, 2 am
        ]

        for pattern in time_patterns:
            match = re.match(pattern, time_str.lower().strip())
            if match:
                hour = int(match.group(1))

                if len(match.groups()) == 2 and match.group(2) in ["am", "pm"]:
                    # 12-hour format
                    if match.group(2) == "pm" and hour != 12:
                        hour += 12
                    elif match.group(2) == "am" and hour == 12:
                        hour = 0
                    minute = 0
                else:
                    # 24-hour format with minutes
                    minute = int(match.group(2))

                return hour, minute

        # Default fallback
        return reference_time.hour, reference_time.minute

    async def _get_solar_time(
        self, solar_event: str, reference_date: datetime
    ) -> tuple:
        """Get solar time (sunrise/sunset) for a given date"""
        if not self.config:
            # Fallback times if no config
            if solar_event == "sunrise":
                return 6, 0  # 6:00 AM
            else:  # sunset
                return 18, 0  # 6:00 PM

        try:
            # Get solar info for the reference date
            solar_info = SolarUtils.get_solar_info(self.config)

            if solar_event == "sunrise":
                time_str = solar_info.get("sunrise", "06:00 UTC")
            else:  # sunset
                time_str = solar_info.get("sunset", "18:00 UTC")

            # Parse the time string which includes timezone (e.g., "19:45 America/Los_Angeles")
            parts = time_str.split()
            if len(parts) >= 1:
                time_part = parts[0]  # "19:45"
                hour, minute = map(int, time_part.split(":"))

                # Log the solar time for debugging
                logger.info(
                    f"Solar {solar_event} time: {time_str} -> {hour:02d}:{minute:02d}"
                )
                return hour, minute
            else:
                # Fallback if parsing fails
                logger.warning(f"Could not parse solar time: {time_str}")
                if solar_event == "sunrise":
                    return 6, 0
                else:
                    return 18, 0

        except Exception as e:
            logger.warning(f"Could not get solar time: {e}")
            # Fallback times
            if solar_event == "sunrise":
                return 6, 0
            else:
                return 18, 0

    async def _parse_datetime(
        self, time_str: str, date_str: str, reference_time: datetime
    ) -> datetime:
        """Parse date and time strings into datetime object"""
        # Parse time
        hour, minute = await self._parse_time(time_str, reference_time)

        # Parse date
        if not date_str or date_str.lower() == "today":
            target_date = reference_time.date()
        elif date_str.lower() == "tomorrow":
            target_date = (reference_time + timedelta(days=1)).date()
        else:
            # Try to parse date string (basic implementation)
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except:
                target_date = reference_time.date()

        # Create target datetime in the same timezone as reference_time
        if reference_time.tzinfo:
            # Create naive datetime first, then localize to the same timezone
            naive_target = datetime.combine(target_date, time(hour, minute))
            target_datetime = reference_time.tzinfo.localize(naive_target)
        else:
            target_datetime = datetime.combine(target_date, time(hour, minute))

        # Log for debugging
        logger.info(
            f"Parsed datetime: time_str='{time_str}', date_str='{date_str}' -> {target_datetime}"
        )
        logger.info(f"Reference time: {reference_time}")

        # If the target time has already passed today, schedule for tomorrow
        # But only for non-solar times or if we're scheduling for "today"
        if target_datetime <= reference_time and (date_str or "").lower() in [
            "",
            "today",
        ]:
            logger.info(f"Target time {target_datetime} has passed, moving to tomorrow")
            target_datetime += timedelta(days=1)

        return target_datetime

    def _parse_time_offset(self, offset_str: str) -> int:
        """Parse time offset string into minutes"""
        offset_str = offset_str.lower().strip()

        if "h" in offset_str:
            hours = int(re.findall(r"(\d+)h", offset_str)[0])
            return hours * 60
        elif "m" in offset_str:
            minutes = int(re.findall(r"(\d+)m", offset_str)[0])
            return minutes
        else:
            # Default to 0 if can't parse
            return 0

    def _generate_job_id(self, schedule_op: ScheduleOperation, room: str) -> str:
        """Generate a unique job ID"""
        import hashlib

        content = f"{room}_{schedule_op.command_to_execute}_{schedule_op.schedule_time}_{schedule_op.recurrence}"
        return f"shade_{hashlib.md5(content.encode()).hexdigest()[:8]}"
