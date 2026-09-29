import unittest
import cv2
import numpy as np

from src.cv_engine.detector import HSVBleedingDetector


class TestDetector(unittest.TestCase):
    def test_red_region_is_detected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (250, 170), (390, 310), (0, 0, 255), -1)
        raw = HSVBleedingDetector().detect(frame)
        self.assertTrue(raw["wound_contour_present"])
        self.assertGreater(raw["blood_pixels_pct"], 1.5)
        self.assertIsNotNone(raw["blood_bbox"])

    def test_blue_cloth_over_wound_is_detected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (250, 170), (390, 310), (0, 0, 255), -1)
        cv2.rectangle(frame, (260, 180), (380, 300), (255, 0, 0), -1)
        raw = HSVBleedingDetector().detect(frame)
        self.assertGreater(raw["cloth_overlap_pct"], 40.0)


if __name__ == "__main__":
    unittest.main()
