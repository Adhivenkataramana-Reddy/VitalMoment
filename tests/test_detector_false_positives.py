import unittest
import cv2
import numpy as np

from src.cv_engine.detector import HSVBleedingDetector


class TestDetectorFalsePositives(unittest.TestCase):
    def setUp(self):
        self.detector = HSVBleedingDetector()

    def test_common_skin_tones_are_not_blood(self):
        # These warm skin-like BGR colors were deliberately chosen because the old
        # HSV-only detector classified some of them as 100% blood.
        for bgr in [
            (80, 120, 170), (70, 100, 150), (60, 90, 130),
            (40, 60, 90), (30, 50, 80), (20, 40, 70),
        ]:
            frame = np.full((480, 640, 3), bgr, dtype=np.uint8)
            raw = self.detector.detect(frame)
            self.assertFalse(raw["wound_contour_present"], bgr)
            self.assertEqual(raw["blood_pixels_pct"], 0.0)

    def test_clean_face_like_scene_is_not_blood(self):
        frame = np.full((480, 640, 3), (70, 100, 150), dtype=np.uint8)
        # Skin-tone face area with darker hair/eyes and a small lip region.
        cv2.ellipse(frame, (320, 240), (145, 185), 0, 0, 360, (70, 100, 150), -1)
        cv2.ellipse(frame, (270, 215), (14, 7), 0, 0, 360, (35, 45, 55), -1)
        cv2.ellipse(frame, (370, 215), (14, 7), 0, 0, 360, (35, 45, 55), -1)
        cv2.ellipse(frame, (320, 305), (28, 10), 0, 0, 360, (50, 45, 95), -1)
        raw = self.detector.detect(frame)
        self.assertFalse(raw["wound_contour_present"])
        self.assertLess(raw["blood_pixels_pct"], 1.5)

    def test_localized_red_wound_on_skin_is_detected(self):
        frame = np.full((480, 640, 3), (70, 100, 150), dtype=np.uint8)
        cv2.ellipse(frame, (320, 240), (145, 185), 0, 0, 360, (70, 100, 150), -1)
        cv2.ellipse(frame, (320, 250), (70, 30), 0, 0, 360, (20, 20, 180), -1)
        raw = self.detector.detect(frame)
        self.assertTrue(raw["wound_contour_present"])
        self.assertGreater(raw["blood_pixels_pct"], 1.5)
        self.assertIsNotNone(raw["blood_bbox"])
        self.assertGreater(raw["detection_confidence"], 0.45)

    def test_lighting_gradient_does_not_create_whole_frame_blood(self):
        h, w = 480, 640
        base = np.zeros((h, w, 3), dtype=np.float32)
        for x in range(w):
            tone = 0.65 + 0.35 * x / (w - 1)
            base[:, x] = np.array([70, 100, 150], dtype=np.float32) * tone
        frame = np.clip(base, 0, 255).astype(np.uint8)
        raw = self.detector.detect(frame)
        self.assertLess(raw["blood_pixels_pct"], 1.5)
        self.assertFalse(raw["wound_contour_present"])


if __name__ == "__main__":
    unittest.main()
