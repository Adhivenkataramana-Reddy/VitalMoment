from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # Optional: OpenCV fallback remains available.
    mp = None


class AnatomyTracker:
    """Hand landmark adapter.

    Uses the classic MediaPipe Solutions API when available. Newer MediaPipe releases
    removed ``mp.solutions``; instead of crashing at startup, VitalMoment falls back
    to a lightweight OpenCV skin-region tracker. The fallback is deliberately simple
    and is intended for prototype pressure/contact estimation, not clinical use.
    """

    def __init__(self, max_num_hands: int = 2):
        self.max_num_hands = max_num_hands
        self.hands = None
        self.mode = "opencv_fallback"

        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.mode = "mediapipe"

    def _opencv_fallback(self, frame) -> List[Dict[str, Any]]:
        """Estimate hand/fingertip locations from skin-colored connected regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad skin range; morphology removes isolated pixels/noise.
        mask = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 220, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 25, 45]), np.array([180, 220, 255]))
        mask = mask | mask2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        output: List[Dict[str, Any]] = []
        h, w = frame.shape[:2]

        for contour in contours[: self.max_num_hands]:
            area = cv2.contourArea(contour)
            if area < max(900.0, 0.003 * w * h):
                continue
            # The topmost point is a useful fingertip/contact proxy for a hand-sized blob.
            pts = contour.reshape(-1, 2)
            tip = pts[np.argmin(pts[:, 1])]
            output.append({
                "index_tip": {
                    "x": float(tip[0] / max(1, w)),
                    "y": float(tip[1] / max(1, h)),
                    "z": 0.0,
                },
                "tracker_mode": self.mode,
            })
        return output

    def get_hand_landmarks(self, frame) -> List[Dict[str, Any]]:
        if self.mode == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            output: List[Dict[str, Any]] = []
            if not results.multi_hand_landmarks:
                return output
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark[8]
                output.append({
                    "index_tip": {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)},
                    "tracker_mode": self.mode,
                })
            return output
        return self._opencv_fallback(frame)

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
