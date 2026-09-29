import json
from .object_detector import ObjectDetector
from .anatomy_tracker import AnatomyTracker

class StateBuilder:
    def __init__(self):
        self.detector = ObjectDetector(model_path="runs/detect/models/wound_detector/weights/best.pt")
        self.tracker = AnatomyTracker()

        # Define our naming conventions (The Medical Ontology)
        self.injury_classes = ["wound", "cut", "bleeding", "laceration", "burn"]
        self.tool_classes = ["bandage", "gauze", "tourniquet", "tape", "glove"]

    def generate_state_payload(self, frame, timestamp):
        # 1. Get raw YOLO object observations
        raw_yolo_obs = self.detector.get_observations(frame)
        hand_obs = self.tracker.get_hand_landmarks(frame)
        
        # 2. Apply Naming Convention & Categorization
        injuries_detected = []
        medical_tools_detected = []
        other_objects = []

        for obs in raw_yolo_obs:
            cls_name = obs["class"].lower()
            
            # Format the object with explicit naming
            formatted_object = {
                "object_type": cls_name,
                "confidence": obs["confidence"],
                "bounding_box": obs["bbox"]
            }

            # Sort it into the correct category
            if cls_name in self.injury_classes:
                injuries_detected.append(formatted_object)
            elif cls_name in self.tool_classes:
                medical_tools_detected.append(formatted_object)
            else:
                other_objects.append(formatted_object)
        
        # 3. Construct the deeply structured JSON payload
        state_payload = {
            "timestamp": timestamp,
            "scene_understanding": {
                "injuries": injuries_detected,
                "medical_tools": medical_tools_detected,
                "other_objects": other_objects
            },
            "anatomy": {
                "hands": hand_obs
            }
        }
        
        return state_payload