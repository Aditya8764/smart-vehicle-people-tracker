import supervision as sv
import cv2

class Annotator:
    def __init__(self):
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(thickness=2)
        self.line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        
    def annotate_frame(self, frame, detections, line_zone=None, classes_names=None):
        """Annotate the frame with bounding boxes, labels, and the line."""
        annotated_frame = frame.copy()
        
        if detections is not None and len(detections) > 0:
            # Generate labels
            labels = []
            for i in range(len(detections)):
                tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else ""
                class_id = detections.class_id[i]
                conf = detections.confidence[i]
                class_name = classes_names.get(class_id, str(class_id)) if classes_names else str(class_id)
                labels.append(f"#{tracker_id} {class_name} {conf:.2f}")
                
            # Annotate traces, boxes, and labels
            annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
        if line_zone is not None:
            # Annotate the counting line
            annotated_frame = self.line_zone_annotator.annotate(frame=annotated_frame, line_counter=line_zone)
            
        return annotated_frame
