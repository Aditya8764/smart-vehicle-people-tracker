from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize the YOLOv8 object detector."""
        self.model = YOLO(model_path)

    def detect(self, frame, classes=None, conf=0.35):
        """
        Run detection on a single frame.
        Args:
            frame: OpenCV image array.
            classes: List of class IDs to filter by (e.g., [0] for person).
            conf: Confidence threshold.
        Returns:
            Ultralytics result object for the frame.
        """
        # Run inference
        results = self.model(frame, classes=classes, conf=conf, verbose=False)
        return results[0]
