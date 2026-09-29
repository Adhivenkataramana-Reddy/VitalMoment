from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


class FirstAidProtocolEngine:
    """Priority-ordered adaptive workflow engine matching VitalMoment workflow v6."""

    RULE_TABLE_VERSION = "rule_table_v4"

    def __init__(self):
        self.previous_rule = None
        self.firing_log: List[Dict[str, Any]] = []
        self.rules: List[Tuple[str, int, Any, str]] = [
            ("R-100", 100, lambda f: not f["wound_visible"], "No wound detected — position camera on the injury."),
            ("R-99", 99, lambda f: f["confidence"] < 0.4, "Unable to clearly assess the situation — please check manually or call for help."),
            ("R-97", 97, lambda f: (not f["pressure_applied"] and f["pressure_recently_removed"] and f["bleeding"] == "severe"), "Pressure was released — reapply pressure immediately."),
            ("R-96", 96, lambda f: (not f["pressure_applied"] and f["bleeding"] == "reducing"), "Pressure was released — reapply pressure immediately."),
            ("R-95", 95, lambda f: (f["bleeding"] == "severe" and not f["pressure_applied"] and not f["pressure_recently_removed"]), "Apply firm, direct pressure to the wound now."),
            ("R-90", 90, lambda f: (f["bleeding"] == "severe" and f["pressure_applied"] and f["time_since_last_change_s"] > 15), "Pressure isn't reducing bleeding — seek emergency help immediately."),
            ("R-85", 85, lambda f: (f["pressure_applied"] and f["bleeding"] == "severe"), "Keep applying firm pressure. Do not remove yet."),
            ("R-80", 80, lambda f: (f["pressure_applied"] and f["bleeding"] == "reducing"), "Bleeding is reducing — continue holding pressure."),
            ("R-75", 75, lambda f: (f["bleeding"] == "controlled" and not f["gauze_present"]), "Bleeding is controlled. Apply a clean dressing/gauze now."),
            ("R-70", 70, lambda f: (f["bleeding"] == "controlled" and f["gauze_present"]), "Dressing applied. Monitor the wound and seek further care if needed."),
            ("R-10", 10, lambda f: (f["wound_visible"] and f["bleeding"] in {"none", "moderate"}), "No significant active bleeding detected. Clean the area and monitor; apply a dressing if needed."),
        ]

    def evaluate(self, facts: Dict[str, Any], timestamp: float = 0.0) -> Dict[str, Any]:
        selected = next(((rid, priority, text) for rid, priority, condition, text in self.rules if condition(facts)), None)
        if selected is None:
            # Defensive fallback; the documented table should be exhaustive for valid Facts.
            rid, priority, guidance = "R-99", 99, "Unable to clearly assess the situation — please check manually or call for help."
        else:
            rid, priority, guidance = selected
        if rid != self.previous_rule:
            entry = {
                "t": timestamp,
                "rule_fired": rid,
                "rule_table_version": self.RULE_TABLE_VERSION,
                "facts_snapshot": {k: facts.get(k) for k in ("bleeding", "pressure_applied", "pressure_recently_removed", "gauze_present", "confidence", "wound_visible")},
                "prev_rule_fired": self.previous_rule,
            }
            self.firing_log.append(entry)
            self.previous_rule = rid
        return {"rule_fired": rid, "priority": priority, "guidance": guidance, "rule_table_version": self.RULE_TABLE_VERSION}

    def process_state(self, state_payload: Dict[str, Any]) -> Dict[str, Any]:
        facts = state_payload.get("facts")
        if facts is None:
            # Backwards-compatible extraction for callers that only provide scene data.
            injuries = state_payload.get("scene_understanding", {}).get("injuries", [])
            facts = {
                "wound_visible": bool(injuries), "confidence": 1.0,
                "bleeding": "severe" if injuries else "none", "pressure_applied": False,
                "pressure_recently_removed": False, "gauze_present": False,
                "time_since_last_change_s": 0.0,
            }
        result = self.evaluate(facts, float(state_payload.get("timestamp", 0.0)))
        return {"current_step": result["rule_fired"], "system_instruction": result["guidance"], **result}

    def save_firing_log(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.firing_log, f, indent=2)
