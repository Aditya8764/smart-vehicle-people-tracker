import supervision as sv

class Tracker:
    def __init__(self):
        """Initialize the ByteTrack object tracker."""
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update the tracker with new frame detections.
        Args:
            detections: supervision.Detections object from YOLO inference.
        Returns:
            supervision.Detections object enriched with tracker IDs.
        """
        return self.tracker.update_with_detections(detections=detections)
