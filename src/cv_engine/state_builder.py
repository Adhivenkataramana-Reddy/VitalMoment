from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from .anatomy_tracker import AnatomyTracker
from .detector import HSVBleedingDetector


class ObservationValidator:
    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold

    def validate(self, obs: Dict[str, Any]):
        if obs.get("detection_confidence", 0.0) < self.confidence_threshold:
            return False, "confidence"
        if not obs.get("blood_detected", False) and obs.get("bleeding_severity") == "severe":
            return False, "consistency"
        if obs.get("cloth_on_wound", False) and not obs.get("wound_visible", False) and not obs.get("blood_detected", False):
            return False, "sanity"
        return True, None


class StateBuilder:
    """Builds time-windowed Facts from validated observations."""

    def __init__(self, detector: Optional[HSVBleedingDetector] = None, tracker: Optional[Any] = None,
                 buffer_seconds: float = 5.0, expected_hz: float = 10.0):
        self.detector = detector or HSVBleedingDetector()
        self.tracker = tracker if tracker is not None else AnatomyTracker()
        self.buffer_seconds = buffer_seconds
        self.expected_hz = expected_hz
        self.observations: Deque[Dict[str, Any]] = deque()
        self.discards: List[Dict[str, Any]] = []
        self.last_facts: Optional[Dict[str, Any]] = None
        self.last_pressure = False
        self.pressure_true_since: Optional[float] = None
        self.last_change_time = 0.0

    def _prune(self, timestamp: float) -> None:
        while self.observations and timestamp - self.observations[0]["timestamp"] > self.buffer_seconds:
            self.observations.popleft()

    def observation_from_raw(self, raw: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        pct = float(raw["blood_pixels_pct"])
        severe = pct > 6.0
        obs = {
            "timestamp": timestamp,
            "wound_visible": bool(raw.get("wound_contour_present")),
            "blood_detected": pct > 1.5,
            "bleeding_severity": "severe" if severe else ("moderate" if pct > 1.5 else "none"),
            "blood_pixels_pct": pct,
            "hand_near_wound": float(raw.get("hand_distance_to_wound_cm_est", 999.0)) < 5.0,
            "cloth_on_wound": float(raw.get("cloth_overlap_pct", 0.0)) > 40.0,
            "detection_confidence": float(raw.get("detection_confidence", 0.0)),
            "detector_id": raw.get("detector_id", "unknown"),
            "threshold_set": raw.get("threshold_set", "unknown"),
        }
        return obs

    def add_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        valid, reason = ObservationValidator().validate(obs)
        if not valid:
            self.discards.append({"timestamp": obs.get("timestamp"), "reason": reason})
        else:
            self.observations.append(obs)
            self._prune(float(obs["timestamp"]))
        return self.build_facts(float(obs.get("timestamp", 0.0)))

    def _duration_true(self, field: str, now: float, window: float) -> float:
        start = max(0.0, now - window)
        vals = [o for o in self.observations if o["timestamp"] >= start]
        if not vals:
            return 0.0
        # Approximate continuous duration by intervals between samples where the field remains true.
        duration = 0.0
        for a, b in zip(vals, vals[1:]):
            if a.get(field) and b["timestamp"] - a["timestamp"] <= 0.75:
                duration += b["timestamp"] - a["timestamp"]
        return duration

    def build_facts(self, now: float) -> Dict[str, Any]:
        self._prune(now)
        if not self.observations:
            return self.last_facts or {
                "bleeding": "none", "wound_visible": False, "pressure_applied": False,
                "pressure_recently_removed": False, "gauze_present": False,
                "time_since_last_change_s": 0.0, "confidence": 0.0,
                "facts_schema_version": "1.1",
            }

        obs = list(self.observations)
        window_start = obs[0]["timestamp"]
        expected = max(1.0, (min(self.buffer_seconds, now - window_start + 0.01)) * self.expected_hz)
        confidence = min(1.0, len(obs) / expected)
        recent = obs[-1]
        wound_visible = sum(o["wound_visible"] for o in obs) >= max(1, len(obs) * 0.3)

        pressure_input = bool(recent.get("hand_near_wound") or recent.get("cloth_on_wound"))
        if pressure_input:
            if self.pressure_true_since is None:
                self.pressure_true_since = now
        else:
            if self.pressure_true_since is not None:
                # Keep the start long enough for reversal detection, then clear it after 4 seconds.
                if now - self.pressure_true_since > 4.0:
                    self.pressure_true_since = None

        pressure_duration = self._duration_true("hand_near_wound", now, 3.0)
        cloth_duration = self._duration_true("cloth_on_wound", now, 3.0)
        pressure_applied = pressure_input and max(pressure_duration, cloth_duration) >= 2.0
        prior_pressure_duration = max(
            self._duration_true("hand_near_wound", now, 4.0),
            self._duration_true("cloth_on_wound", now, 4.0),
        )
        pressure_recently_removed = (not pressure_input) and prior_pressure_duration >= 2.0 and self.last_pressure

        avg = sum(o["blood_pixels_pct"] for o in obs) / len(obs)
        first_avg = sum(o["blood_pixels_pct"] for o in obs[:max(1, len(obs)//3)]) / max(1, len(obs)//3)
        last_avg = sum(o["blood_pixels_pct"] for o in obs[-max(1, len(obs)//3):]) / max(1, len(obs)//3)
        if avg > 6.0:
            bleeding = "severe"
        elif last_avg < first_avg * 0.75 and first_avg > 1.5:
            bleeding = "reducing"
        elif avg > 1.5:
            bleeding = "moderate"
        elif wound_visible:
            bleeding = "controlled"
        else:
            bleeding = "none"

        cloth_present = cloth_duration >= 2.0
        facts = {
            "bleeding": bleeding,
            "bleeding_basis": f"avg blood_pixels_pct={avg:.2f} over {now-window_start:.1f}s, trend={'decreasing' if last_avg < first_avg else 'flat'}",
            "wound_visible": wound_visible,
            "pressure_applied": pressure_applied,
            "pressure_basis": f"hand/cloth pressure duration={max(pressure_duration, cloth_duration):.1f}s over 3.0s",
            "pressure_recently_removed": pressure_recently_removed,
            "pressure_recently_removed_basis": f"prior pressure={prior_pressure_duration:.1f}s; current={pressure_input}",
            "gauze_present": cloth_present,
            "gauze_present_basis": f"cloth_on_wound duration={cloth_duration:.1f}s over 3.0s",
            "time_since_last_change_s": now - self.last_change_time,
            "confidence": confidence,
            "facts_schema_version": "1.1",
        }
        if self.last_facts is None or any(facts[k] != self.last_facts.get(k) for k in ("bleeding", "pressure_applied", "wound_visible", "gauze_present")):
            self.last_change_time = now
            facts["time_since_last_change_s"] = 0.0
        self.last_pressure = pressure_applied or pressure_input
        self.last_facts = facts
        return facts

    def generate_state_payload(self, frame, timestamp: float) -> Dict[str, Any]:
        raw = self.detector.detect(frame)
        # Hand-to-wound distance is computed from the wound center and index fingertips.
        hands = self.tracker.get_hand_landmarks(frame)
        raw["hand_distance_to_wound_cm_est"] = 999.0
        if raw.get("blood_bbox") and hands:
            x, y, w, h = raw["blood_bbox"]
            cx, cy = x + w / 2.0, y + h / 2.0
            fw, fh = frame.shape[1], frame.shape[0]
            raw["hand_distance_to_wound_cm_est"] = min(
                (((hand["index_tip"]["x"] * fw - cx) ** 2 + (hand["index_tip"]["y"] * fh - cy) ** 2) ** 0.5) / 10.0
                for hand in hands
            )
        obs = self.observation_from_raw(raw, timestamp)
        facts = self.add_observation(obs)
        injuries = []
        if raw.get("blood_bbox") and raw.get("wound_contour_present"):
            x, y, w, h = raw["blood_bbox"]
            injuries.append({"object_type": "wound", "confidence": round(raw["detection_confidence"], 2), "bounding_box": [x, y, x+w, y+h]})
        tools = []
        if raw.get("cloth_overlap_pct", 0.0) > 40.0 and raw.get("blood_bbox"):
            x, y, w, h = raw["blood_bbox"]
            tools.append({"object_type": "gauze", "confidence": 0.8, "bounding_box": [x, y, x+w, y+h]})
        return {
            "timestamp": timestamp,
            "frame_size": [frame.shape[1], frame.shape[0]],
            "scene_understanding": {"injuries": injuries, "medical_tools": tools, "other_objects": []},
            "anatomy": {"hands": hands},
            "observation": obs,
            "facts": facts,
            "discard_log": list(self.discards[-10:]),
        }

    def close(self):
        close = getattr(self.tracker, "close", None)
        if close:
            close()
