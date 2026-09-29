import os
from ultralytics import YOLO

def main():
    # Dynamically find the absolute path to your data.yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(project_root, "data", "yolo_formatted", "data.yaml")

    if not os.path.exists(yaml_path):
        print(f"Error: Could not find dataset config at {yaml_path}")
        return

    # Load the base pretrained Nano model
    model = YOLO('yolov8n.pt')
    
    print(f"Starting training on medical dataset: {yaml_path}")

    # Train the model
    model.train(
        data=yaml_path,
        epochs=25,               # Number of times it reviews the whole dataset
        imgsz=640,               # Standard image resolution for YOLO
        batch=8,                 # Small batch size to prevent memory crashes
        device='cpu',            # Set to '0' if you have an NVIDIA GPU
        project='models',        # Where to save the output
        name='wound_detector'    # Name of the output folder
    )
    
    print("Training complete! Your custom weights are in: models/wound_detector/weights/best.pt")

if __name__ == "__main__":
    main()