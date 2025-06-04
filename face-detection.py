# -*- coding: utf-8 -*-
import os
import cv2  # type: ignore
from datetime import datetime

# Parche para protobuf: si GetMessageClass no existe, lo definimos usando symbol_database
try:
    import google.protobuf.message_factory as _message_factory
    from google.protobuf import symbol_database as _symbol_database
    # If MediaPipe expects GetMessageClass but it's not defined, assign it via symbol_database.GetPrototype
    if not hasattr(_message_factory, 'GetMessageClass'):
        _message_factory.GetMessageClass = lambda descriptor: _symbol_database.Default().GetPrototype(descriptor)
except Exception as e:
    # If the patch fails, emit a warning and let MediaPipe fall back
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
                # Convert to (x1, y1, x2, y2) for NMS
                boxes.append((x, y, x + box_w, y + box_h))
                # detection.score is a list; the first element is the confidence
                scores.append(detection.score[0])

            # Apply Non-Maximum Suppression before returning final boxes
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
                w_int = max(0, xx2 - xx1 + 1)
                h_int = max(0, yy2 - yy1 + 1)
                inter = w_int * h_int
                union = areas[i] + areas[j] - inter
                iou = inter / union if union > 0 else 0
                if iou > iou_thresh:
                    suppress_list.append(j)
            # Remove suppressed indices from order
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

class HeadOrientationChecker:
    """
    Uses Haar Cascades to check if a face is oriented frontal by detecting eyes inside the face region.
    """
    def __init__(self, eye_cascade_path=None, min_eye_count=2):
        # Load the Haar Cascade for eye detection
        # If no path provided, use the default haarcascade_eye.xml from OpenCV data
        if eye_cascade_path is None:
            eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        # Minimum number of eyes to consider the face as frontal
        self.min_eye_count = min_eye_count

    def is_facing_camera(self, frame, box):
        """
        Given the full frame and a bounding box (x, y, w, h), return True
        if the face in that box is likely looking forward (frontal).
        We crop the box, convert to grayscale, and detect eyes inside it.
        If we detect at least `min_eye_count` eyes, asumimos que es frontal.
        """
        frame_h, frame_w, _ = frame.shape
        x, y, w, h = box

        # Clamp coordinates to image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, frame_w)
        y2 = min(y + h, frame_h)
        crop_w = x2 - x1
        crop_h = y2 - y1

        # If region is invalid, no podemos afirmar que mire frontal
        if crop_w <= 0 or crop_h <= 0:
            return False

        # Crop the face region
        face_img = frame[y1:y2, x1:x2]
        if face_img is None or face_img.size == 0:
            return False

        # Convert to grayscale para aplicar Haar Cascade
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Ajustar parámetros de detectMultiScale según condiciones de iluminación/tamaño
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Si detecta al menos `min_eye_count` ojos, asumimos que la cara está frontal
        return len(eyes) >= self.min_eye_count

class FaceSaver:
    """
    Saves each detected face region as an image file into a 'faces' directory,
    pero sólo si el rostro está orientado frontal (por detección de ojos) y es nuevo.
    Usa comparación por histogramas de escala de grises para detectar unicidad.
    """
    def __init__(self, output_dir="faces", hist_threshold=0.9):
        self.output_dir = output_dir
        self.hist_threshold = hist_threshold  # Threshold for histogram correlation
        # Create the directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        # List to store known face histograms
        self.known_histograms = []

    def save(self, frame, boxes, orientation_checker):
        """
        Given the full frame, a list of bounding boxes (x, y, w, h) and a HeadOrientationChecker,
        sólo guarda las caras que estén orientadas frontalmente y sean nuevas.
        """
        frame_h, frame_w, _ = frame.shape
        for (x, y, w, h) in boxes:
            # Sólo seguimos si la cabeza está orientada frontal (detectamos ojos dentro del rostro)
            if not orientation_checker.is_facing_camera(frame, (x, y, w, h)):
                continue

            # Clamp coordinates to image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, frame_w)
            y2 = min(y + h, frame_h)
            crop_w = x2 - x1
            crop_h = y2 - y1

            # Skip invalid or empty regions
            if crop_w <= 0 or crop_h <= 0:
                continue

            # Crop the face region from the frame
            face_img = frame[y1:y2, x1:x2]
            if face_img is None or face_img.size == 0:
                continue

            # Convert to grayscale
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Compute normalized histogram (256 bins)
            hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)

            # Compare with known histograms
            is_new = True
            for known_hist in self.known_histograms:
                correlation = cv2.compareHist(known_hist, hist, cv2.HISTCMP_CORREL)
                if correlation >= self.hist_threshold:
                    # Too similar to an existing face
                    is_new = False
                    break

            if not is_new:
                continue

            # Si es una cara nueva y orientada frontalmente: guardamos
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"face_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, face_img)
            print(f"Saved new face (frontal) to {filepath}", flush=True)

            # Store normalized histogram for future comparisons
            self.known_histograms.append(hist)

class FaceBoxApp:
    """Orchestrates webcam capture, face detection, drawing of boxes, y guardado multicondicional"""
    def __init__(self, webcam, detector, drawer, orientation_checker, saver):
        self.webcam = webcam
        self.detector = detector
        self.drawer = drawer
        self.orientation_checker = orientation_checker
        self.saver = saver

    def run(self):
        """Main loop: read frame, detect faces, guardar sólo si frontal y nuevo, draw boxes y display"""
        try:
            while True:
                frame = self.webcam.read()
                boxes = self.detector.detect(frame)

                # Guardar sólo si está orientado frontal y es nuevo
                if boxes:
                    self.saver.save(frame, boxes, self.orientation_checker)

                # Debug: print number of faces detected
                print(f"Detected {len(boxes)} face(s) in current frame", flush=True)
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
        detector = FaceDetector(min_confidence=0.3, model_selection=1, nms_threshold=0.3)
        drawer = BoxDrawer(thickness=3)
        # HeadOrientationChecker usa Haar Cascades para detectar ojos dentro del rostro
        orientation_checker = HeadOrientationChecker(min_eye_count=2)
        saver = FaceSaver(output_dir="faces", hist_threshold=0.9)
        app = FaceBoxApp(cam, detector, drawer, orientation_checker, saver)
        app.run()
    except Exception as e:
        # Print any error to console
        print(f"Error: {e}", flush=True)
