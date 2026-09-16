import supervision as sv

class Counter:
    def __init__(self, start_point, end_point):
        """
        Initialize the line crossing counter.
        Args:
            start_point: Tuple (x, y) for start of line.
            end_point: Tuple (x, y) for end of line.
        """
        self.set_line(start_point, end_point)
        
    def set_line(self, start_point, end_point):
        self.start_point = sv.Point(x=start_point[0], y=start_point[1])
        self.end_point = sv.Point(x=end_point[0], y=end_point[1])
        self.line_zone = sv.LineZone(start=self.start_point, end=self.end_point)

    def trigger(self, detections: sv.Detections):
        """
        Check if any detections crossed the line.
        Args:
            detections: Tracked detections.
        Returns:
            Tuple (in_count, out_count)
        """
        self.line_zone.trigger(detections=detections)
        return self.line_zone.in_count, self.line_zone.out_count
