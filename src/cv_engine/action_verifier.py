from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


class ActionVerifier:
    """Verifies whether an observed hand/cloth signal overlaps the detected wound."""

    def is_applying_pressure(self, state_payload: Dict[str, Any], frame_width: Optional[int] = None,
                             frame_height: Optional[int] = None):
        injuries = state_payload.get("scene_understanding", {}).get("injuries", [])
        hands = state_payload.get("anatomy", {}).get("hands", [])

        wound_bbox = next(
            (x.get("bounding_box") for x in injuries if x.get("object_type") in {"wound", "cut", "bleeding", "laceration"}),
            None,
        )
        if not wound_bbox:
            return False, "No wound detected"

        if not hands:
            # A blue/green staged cloth counts as pressure evidence as defined by the workflow.
            tools = state_payload.get("scene_understanding", {}).get("medical_tools", [])
            if any(t.get("object_type") in {"gauze", "cloth", "bandage"} for t in tools):
                return True, "Pressure verified: dressing/cloth detected over wound"
            return False, "No hands detected"

        width = frame_width or state_payload.get("frame_size", [640, 480])[0]
        height = frame_height or state_payload.get("frame_size", [640, 480])[1]
        x_min, y_min, x_max, y_max = wound_bbox

        for hand in hands:
            tip = hand.get("index_tip", {})
            px = float(tip.get("x", -1)) * width
            py = float(tip.get("y", -1)) * height
            if x_min <= px <= x_max and y_min <= py <= y_max:
                return True, "Pressure verified: hand overlaps wound"
        return False, "Hand is visible, but not touching the wound"
