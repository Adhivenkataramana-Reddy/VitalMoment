import cv2
import mediapipe as mp

class AnatomyTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )

    def get_hand_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract the index finger tip (landmark 8) for action verification
                index_tip = hand_landmarks.landmark[8]
                hands_data.append({
                    "index_tip": {
                        "x": round(index_tip.x, 3),
                        "y": round(index_tip.y, 3),
                        "z": round(index_tip.z, 3)
                    }
                })
        return hands_data