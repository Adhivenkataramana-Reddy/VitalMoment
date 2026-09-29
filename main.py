"""VitalMoment offline-first live prototype.

Run:
    python main.py
Press Q to quit.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2

from src.cv_engine.state_builder import StateBuilder
from src.protocol_engine.engine import FirstAidProtocolEngine


def draw_overlay(frame, payload, decision):
    out = frame.copy()
    for injury in payload["scene_understanding"]["injuries"]:
        x1, y1, x2, y2 = map(int, injury["bounding_box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"WOUND {injury['confidence']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    facts = payload["facts"]
    lines = [
        f"BLEEDING: {facts['bleeding'].upper()}   CONF: {facts['confidence']:.2f}",
        f"PRESSURE: {'YES' if facts['pressure_applied'] else 'NO'}   GAUZE: {'YES' if facts['gauze_present'] else 'NO'}",
        f"RULE: {decision['rule_fired']}",
        decision["guidance"],
        "Q = quit",
    ]
    y = 28
    for i, line in enumerate(lines):
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .58 if i != 3 else .5, (0, 255, 255) if i < 3 else (0, 255, 0), 2)
        y += 28
    return out


def run(camera: int = 0, width: int = 960, height: int = 540):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera}. Check camera permissions or use --camera <index>.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    builder = StateBuilder()
    engine = FirstAidProtocolEngine()
    start = time.monotonic()
    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            timestamp = time.monotonic() - start
            payload = builder.generate_state_payload(frame, timestamp)
            decision = engine.process_state(payload)
            display = draw_overlay(frame, payload, decision)
            cv2.imshow("VitalMoment - Adaptive First Aid Assistant", display)
            if timestamp - last_print >= 1.0:
                print(json.dumps({"t": round(timestamp, 2), "rule": decision["rule_fired"], "facts": payload["facts"]}, indent=2))
                last_print = timestamp
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        builder.close()
        cv2.destroyAllWindows()
        engine.save_firing_log("firing_log.json")
        print("Firing log saved to firing_log.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VitalMoment live prototype")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(camera=args.camera)
