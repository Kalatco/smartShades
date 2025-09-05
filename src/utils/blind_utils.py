"""
Blind filtering and targeting utilities for the Smart Shades Agent
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class BlindUtils:
    """Utility class for blind filtering and targeting"""

    @staticmethod
    def filter_blinds(blinds, filter_keywords):
        """Filter blinds based on keywords matching blind names"""
        if not filter_keywords:
            return blinds

        filtered = []
        for blind in blinds:
            blind_name_lower = blind.name.lower()
            if any(keyword.lower() in blind_name_lower for keyword in filter_keywords):
                filtered.append(blind)
        return filtered

    @staticmethod
    def get_target_blinds_for_operation(
        config, scope: str, blind_filter: List[str], room: str
    ) -> Tuple[List, List[str]]:
        """Get target blinds based on scope and filters for a specific operation"""
        target_blinds = []
        affected_rooms = []

        if scope == "house":
            # All rooms
            for room_name, room_config in config.rooms.items():
                if blind_filter:
                    filtered_blinds = BlindUtils.filter_blinds(
                        room_config.blinds, blind_filter
                    )
                    if filtered_blinds:
                        target_blinds.extend(filtered_blinds)
                        affected_rooms.append(room_name)
                else:
                    target_blinds.extend(room_config.blinds)
                    affected_rooms.append(room_name)
        else:
            # Current room only (both "room" and "specific" scope)
            room_blinds = config.rooms[room].blinds
            if blind_filter:
                filtered_blinds = BlindUtils.filter_blinds(room_blinds, blind_filter)
                if filtered_blinds:
                    target_blinds = filtered_blinds
                    affected_rooms = [room]
            else:
                target_blinds = room_blinds
                affected_rooms = [room]

        return target_blinds, affected_rooms
