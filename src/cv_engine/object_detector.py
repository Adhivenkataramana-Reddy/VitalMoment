"""Optional YOLO adapter.

The live prototype does not require YOLO. This adapter is retained so a trained detector
can be introduced later without changing the downstream Observation/State contracts.
"""
from __future__ import annotations

from pathlib import Path


class ObjectDetector:
    def __init__(self, model_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Optional YOLO support requires: python -m pip install ultralytics") from exc
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")
        self.model = YOLO(str(model_path))

    def get_observations(self, frame):
        result = self.model(frame, verbose=False)[0]
        observations = []
        for box in result.boxes:
            observations.append({
                "class": self.model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(float(c), 1) for c in box.xyxy[0].tolist()],
            })
        return observations
