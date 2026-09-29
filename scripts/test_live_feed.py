import cv2
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_engine.state_builder import StateBuilder
from src.protocol_engine.engine import FirstAidProtocolEngine

def draw_visuals_from_json(frame, payload):
    """Draws bounding boxes and hand points on the frame using the JSON data."""
    annotated_frame = frame.copy()
    height, width, _ = annotated_frame.shape
    
    scene = payload.get("scene_understanding", {})
    
    # 1. Draw Injuries (Red)
    for obs in scene.get("injuries", []):
        x1, y1, x2, y2 = [int(c) for c in obs["bounding_box"]]
        label = f"INJURY: {obs['object_type']} ({obs['confidence']})"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 2. Draw Medical Tools (Blue)
    for obs in scene.get("medical_tools", []):
        x1, y1, x2, y2 = [int(c) for c in obs["bounding_box"]]
        label = f"TOOL: {obs['object_type']} ({obs['confidence']})"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    # 3. Draw Anatomy (Green)
    for hand in payload.get("anatomy", {}).get("hands", []):
        index_x = int(hand["index_tip"]["x"] * width)
        index_y = int(hand["index_tip"]["y"] * height)
        cv2.circle(annotated_frame, (index_x, index_y), 8, (0, 255, 0), -1)
        cv2.putText(annotated_frame, "Index", (index_x + 10, index_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame

def main():
    print("Initializing AI models... This may take a few seconds.")
    builder = StateBuilder()
    engine = FirstAidProtocolEngine()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Starting AI First-Aid Assistant... Press 'q' to quit.")
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            continue

        current_timestamp = round(time.time() - start_time, 2)

        # 1. Perception Layer: Get data
        payload = builder.generate_state_payload(frame, current_timestamp)
        
        # 2. Print pure JSON to terminal
        print(json.dumps(payload, indent=2))

        # 3. Reasoning Layer: Process logic
        protocol_status = engine.process_state(payload)

        # 4. Visual Layer: Draw boxes and instructions
        display_frame = draw_visuals_from_json(frame, payload)
        
        cv2.putText(display_frame, f"Status: {protocol_status['current_step']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, protocol_status['system_instruction'], 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('VitalMoment - End-to-End System', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()