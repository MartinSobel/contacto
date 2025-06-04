# -*- coding: utf-8 -*-
import cv2  # type: ignore

# Parche para protobuf: si GetMessageClass no existe, lo definimos usando symbol_database
try:
    import google.protobuf.message_factory as _message_factory
    from google.protobuf import symbol_database as _symbol_database
    # If MediaPipe expects GetMessageClass but it's not defined, assign it via symbol_database.GetPrototype
    if not hasattr(_message_factory, 'GetMessageClass'):
        _message_factory.GetMessageClass = lambda descriptor: _symbol_database.Default().GetPrototype(descriptor)
except Exception as e:
    # If the patch fails, emit a warning and let MediaPipe fall back (pero idealmente no debería fallar de nuevo)
    print(f"Warning: failed to patch protobuf: {e}", flush=True)

import mediapipe as mp  # type: ignore

class Webcam:
    """Responsible for capturing frames from a USB webcam with configurable resolution"""
    def __init__(self, src=0, width=1280, height=720):
        # Try to open the camera using AVFoundation (recommended on macOS/M1)
        try:
            self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        except Exception:
            # Fallback: without specifying backend
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        # Set desired resolution for better detection of distant faces
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """Return a single frame from the webcam"""
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Failed to read frame from webcam")
        return frame

    def release(self):
        """Release the webcam resource"""
        self.cap.release()

class FaceDetector:
    """Uses MediaPipe Face Detection to find face bounding boxes at long range and then applies NMS"""
    def __init__(self, min_confidence=0.3, model_selection=1, nms_threshold=0.3):
        # model_selection=1 for full-range (up to ~5 m)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )
        # Threshold for Non-Maximum Suppression (IoU)
        self.nms_threshold = nms_threshold

    def detect(self, frame):
        """
        Detect faces and return list of bounding boxes as (x, y, width, height).
        Applies Non-Maximum Suppression to filter overlapping boxes.
        Coordinates are in pixel values.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        boxes = []
        scores = []
        h, w, _ = frame.shape

        if results.detections:
            for detection in results.detections:
                # Extract relative bounding box and score
                rel_bbox = detection.location_data.relative_bounding_box
                x = int(rel_bbox.xmin * w)
                y = int(rel_bbox.ymin * h)
                box_w = int(rel_bbox.width * w)
                box_h = int(rel_bbox.height * h)
                boxes.append((x, y, x + box_w, y + box_h))
                # detection.score es una lista, el primer elemento es la confianza
                scores.append(detection.score[0])

            # Apply Non-Maximum Suppression before devolver los cuadros finales
            kept_indices = self._non_max_suppression(boxes, scores, self.nms_threshold)
            final_boxes = []
            for idx in kept_indices:
                x1, y1, x2, y2 = boxes[idx]
                final_boxes.append((x1, y1, x2 - x1, y2 - y1))
            return final_boxes

        return []

    @staticmethod
    def _non_max_suppression(boxes, scores, iou_thresh):
        """
        Apply Non-Maximum Suppression (NMS) to bounding boxes.
        boxes: list of (x1, y1, x2, y2)
        scores: list of confidence scores
        iou_thresh: IoU threshold for suppression
        Returns list of indices of boxes to keep.
        """
        x1 = [b[0] for b in boxes]
        y1 = [b[1] for b in boxes]
        x2 = [b[2] for b in boxes]
        y2 = [b[3] for b in boxes]

        # Compute areas
        areas = [(x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1) for i in range(len(boxes))]
        # Sort by score descending
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            suppress_list = []
            for j in order:
                # Compute intersection
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                inter = w * h
                union = areas[i] + areas[j] - inter
                iou = inter / union if union > 0 else 0
                if iou > iou_thresh:
                    suppress_list.append(j)
            # Remove suprimidos de la lista de orden
            order = [o for o in order if o not in suppress_list]

        return keep

class BoxDrawer:
    """Draws rectangles on a frame given bounding boxes"""
    def __init__(self, thickness=2):
        self.thickness = thickness
        self.color = (0, 255, 0)  # Green in BGR

    def draw(self, frame, boxes):
        """Draw one rectangle per bounding box"""
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, self.thickness)

class FaceBoxApp:
    """Orchestrates webcam capture, face detection and drawing of boxes"""
    def __init__(self, webcam, detector, drawer):
        self.webcam = webcam
        self.detector = detector
        self.drawer = drawer

    def run(self):
        """Main loop: read frame, detect faces, draw boxes and display"""
        try:
            while True:
                frame = self.webcam.read()
                boxes = self.detector.detect(frame)
                # Debug: print number of faces detected
                print(f"Detected {len(boxes)} face(s) in current frame")
                self.drawer.draw(frame, boxes)

                cv2.imshow("Face Boxes", frame)
                key = cv2.waitKey(1) & 0xFF
                # Exit on 'q' or Esc
                if key == ord('q') or key == 27:
                    break
        finally:
            self.webcam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Instantiate components with higher resolution and long-range model
        cam = Webcam(src=0, width=1280, height=720)
        # Aquí cambiamos nms_threshold a 0.3 (puedes ajustar si aún detecta dos cuadros muy cercanos)
        detector = FaceDetector(min_confidence=0.3, model_selection=1, nms_threshold=0.3)
        drawer = BoxDrawer(thickness=3)
        app = FaceBoxApp(cam, detector, drawer)
        app.run()
    except Exception as e:
        # Print any error to console
        print(f"Error: {e}", flush=True)
