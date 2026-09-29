from pathlib import Path
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]
    data_yaml = root / "data" / "yolo_formatted" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml), epochs=25, imgsz=640, batch=8,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
        project=str(root / "models"), name="wound_detector", exist_ok=True,
    )
    print(root / "models" / "wound_detector" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
