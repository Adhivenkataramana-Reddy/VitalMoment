"""Offline-first visual detector used by the VitalMoment prototype.

The detector deliberately contains no workflow decisions. It only extracts raw visual
measurements which are converted to observations by StateBuilder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    detector_id: str = "hsv_skin_guard_v3"
    threshold_set: str = "v3_skin_guard"
    min_red_pixels: int = 250
    wound_presence_pct: float = 0.75
    severe_blood_pct: float = 6.0
    blood_detected_pct: float = 1.5
    blue_hue_low: int = 90
    blue_hue_high: int = 140


class HSVBleedingDetector:
    """Detect staged blood/red regions and a blue dressing prop with OpenCV only."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

    @staticmethod
    def _largest_component(mask: np.ndarray) -> Tuple[np.ndarray, float, Optional[Tuple[int, int, int, int]]]:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cleaned, 0.0, None
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 1:
            return cleaned, 0.0, None
        return cleaned, area, cv2.boundingRect(contour)

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a BGR image with shape (H, W, 3)")

        h, w = frame.shape[:2]
        total_pixels = float(h * w)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        b, g, r = cv2.split(frame)
        rf, gf, bf = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)

        # A plain HSV red threshold is too permissive for skin. In particular, normal
        # warm skin can fall inside H=0..12 and produce enormous false wound regions.
        # Keep HSV as the first gate, then require red dominance and local contrast.
        red_hue = (hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168)
        red_excess = rf - ((gf + bf) * 0.5)
        red_minus_green = rf - gf
        red_minus_blue = rf - bf
        local_red_excess = red_excess - cv2.GaussianBlur(red_excess, (0, 0), 7)

        broad_red = (
            red_hue
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 40)
            & (red_minus_green >= 35)
            & (red_minus_blue >= 45)
            & (red_excess >= 55)
        )

        # Approximate skin mask. It is NOT used to blanket-reject blood, because fresh
        # blood on skin can share some skin chroma. It is used only when a red component
        # is diffuse and skin-like rather than locally distinct.
        skin = (
            (ycrcb[:, :, 1] >= 135) & (ycrcb[:, :, 1] <= 180)
            & (ycrcb[:, :, 2] >= 85) & (ycrcb[:, :, 2] <= 135)
        )

        candidate = broad_red & (local_red_excess >= 5)
        candidate = candidate.astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Select a coherent wound-like component. Large diffuse skin regions are rejected;
        # compact, strongly red/local-contrast regions remain eligible.
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        accepted_mask = np.zeros_like(candidate)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 1.0:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            comp = np.zeros_like(candidate)
            cv2.drawContours(comp, [contour], -1, 255, -1)
            pix = comp > 0
            skin_fraction = float(np.count_nonzero(skin & pix)) / max(1, int(np.count_nonzero(pix)))
            mean_excess = float(np.mean(red_excess[pix]))
            mean_local = float(np.mean(local_red_excess[pix]))
            mean_sat = float(np.mean(hsv[:, :, 1][pix]))
            area_pct = area / total_pixels * 100.0

            # Normal skin tends to be broad, low-local-contrast and skin-dominant.
            # A compact red region with strong red excess/local contrast is retained.
            strong_blood = mean_excess >= 75.0 and mean_local >= 7.0
            diffuse_skin = skin_fraction >= 0.75 and mean_local < 12.0 and mean_excess < 85.0
            too_large = area_pct > 45.0
            if diffuse_skin or too_large:
                continue

            # Score favors red dominance and local contrast, with a mild compactness term.
            perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
            compactness = min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter))
            score = (
                min(2.0, mean_excess / 55.0)
                + min(2.0, mean_local / 12.0)
                + min(1.0, mean_sat / 140.0)
                + compactness
            )
            if strong_blood:
                score += 1.0
            if skin_fraction > 0.65:
                score -= 0.5

            if score > best_score:
                best_score = score
                best = (x, y, bw, bh, area, mean_excess, mean_local, skin_fraction, strong_blood)

        if best is not None:
            x, y, bw, bh, area, mean_excess, mean_local, skin_fraction, strong_blood = best
            # Only expose the winning component as the blood mask. This prevents unrelated
            # red facial/background pixels from inflating blood_pixels_pct.
            contour_mask = np.zeros_like(candidate)
            contours2, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours2:
                if cv2.boundingRect(c) == (x, y, bw, bh) and abs(cv2.contourArea(c) - area) < max(2.0, area * 0.02):
                    cv2.drawContours(contour_mask, [c], -1, 255, -1)
                    break
            blood_mask = contour_mask
            blood_bbox = (x, y, bw, bh)
            blood_area = area
            evidence = max(0.0, min(1.0, 0.45 * min(1.0, mean_excess / 100.0) + 0.35 * min(1.0, mean_local / 20.0) + 0.20 * min(1.0, mean_sat / 180.0)))
            confidence = 0.35 + 0.60 * evidence
            if skin_fraction > 0.70 and not strong_blood:
                confidence *= 0.65
        else:
            blood_mask = accepted_mask
            blood_bbox = None
            blood_area = 0.0
            confidence = 0.20

        blood_pct = (float(cv2.countNonZero(blood_mask)) / total_pixels) * 100.0
        wound_present = bool(
            blood_bbox
            and blood_area >= self.config.min_red_pixels
            and blood_pct >= self.config.wound_presence_pct
            and confidence >= 0.45
        )

        # Blue/green staged cloth. We measure overlap against the selected wound box.
        blue_mask = cv2.inRange(
            hsv,
            np.array([self.config.blue_hue_low, 80, 40]),
            np.array([self.config.blue_hue_high, 255, 255]),
        )
        cloth_overlap_pct = 0.0
        if blood_bbox:
            x, y, bw, bh = blood_bbox
            roi = blue_mask[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            if roi.size:
                cloth_overlap_pct = float(cv2.countNonZero(roi)) / float(roi.size) * 100.0

        return {
            "detector_id": self.config.detector_id,
            "threshold_set": self.config.threshold_set,
            "blood_pixels_pct": blood_pct,
            "blood_bbox": list(blood_bbox) if blood_bbox else None,
            "wound_contour_present": wound_present,
            "cloth_overlap_pct": cloth_overlap_pct,
            "detection_confidence": confidence,
        }
