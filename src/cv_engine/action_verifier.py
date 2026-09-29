class ActionVerifier:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def is_applying_pressure(self, state_payload):
        hands = state_payload.get("anatomy", {}).get("hands", [])
        if not hands:
            return False, "No hands detected"

        # FIX: Read from the new nested Ontology structure
        injuries = state_payload.get("scene_understanding", {}).get("injuries", [])
        
        wound_bbox = None
        for obs in injuries:
            if obs["object_type"] in ["wound", "cut", "bleeding"]:  
                wound_bbox = obs["bounding_box"]
                break
        
        if not wound_bbox:
            return False, "No wound detected"

        x_min, y_min, x_max, y_max = wound_bbox
        
        for hand in hands:
            index_tip = hand["index_tip"]
            hand_pixel_x = index_tip["x"] * self.frame_width
            hand_pixel_y = index_tip["y"] * self.frame_height

            is_x_inside = x_min <= hand_pixel_x <= x_max
            is_y_inside = y_min <= hand_pixel_y <= y_max

            if is_x_inside and is_y_inside:
                return True, "Pressure verified: Hand is touching the wound"
        
        return False, "Hand is visible, but not touching the wound"