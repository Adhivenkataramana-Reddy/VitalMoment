from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def get_observations(self, frame):
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        observations = []
        for box in results.boxes:
            # Extract coordinates and class info
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf)
            cls_name = self.model.names[int(box.cls)]
            
            observations.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "bbox": [round(c, 1) for c in coords]
            })
            
        return observations