from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    print("VitalMoment checkpoint")
    print(f"Project: {root}")
    print("Architecture: Webcam -> HSV + MediaPipe -> Observation -> Validator -> State Builder -> Adaptive Rule Engine")
    print("Runtime dependency: OpenCV + NumPy + MediaPipe")
    print("Optional experiment: Ultralytics YOLO checkpoint")
    print(f"Tests: {root / 'tests'}")


if __name__ == "__main__":
    main()
