# -*- coding: utf-8 -*-
import os
import time
import cv2  # type: ignore
import serial # type: ignore
from datetime import datetime

# Protobuf patch for MediaPipe compatibility
try:
    import google.protobuf.message_factory as _message_factory # type: ignore
    from google.protobuf import symbol_database as _symbol_database # type: ignore
    if not hasattr(_message_factory, 'GetMessageClass'):
        _message_factory.GetMessageClass = lambda descriptor: _symbol_database.Default().GetPrototype(descriptor)
except Exception as e:
    print(f"Warning: protobuf patch failed: {e}", flush=True)

import mediapipe as mp  # type: ignore

class Webcam:
    """Capture frames from USB webcam with configurable resolution."""
    def __init__(self, src=0, width=1280, height=720):
        # Try AVFoundation (macOS), else default
        try:
            self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        except Exception:
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Failed to read frame from webcam")
        return frame

    def release(self):
        self.cap.release()

class FaceDetector:
    """Detect faces and apply Non-Maximum Suppression."""
    def __init__(self, min_confidence=0.3, model_selection=1, nms_threshold=0.3):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )
        self.nms_threshold = nms_threshold

    def detect(self, frame):
        """Return list of bounding boxes (x, y, w, h) for detected faces."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        boxes, scores = [], []
        h, w, _ = frame.shape

        if results.detections:
            for det in results.detections:
                r = det.location_data.relative_bounding_box
                x1 = int(r.xmin * w)
                y1 = int(r.ymin * h)
                bw = int(r.width * w)
                bh = int(r.height * h)
                boxes.append((x1, y1, x1 + bw, y1 + bh))
                scores.append(det.score[0])

            keep = self._non_max_suppression(boxes, scores, self.nms_threshold)
            final = []
            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                final.append((x1, y1, x2 - x1, y2 - y1))
            return final

        return []

    @staticmethod
    def _non_max_suppression(boxes, scores, iou_thresh):
        x1 = [b[0] for b in boxes]; y1 = [b[1] for b in boxes]
        x2 = [b[2] for b in boxes]; y2 = [b[3] for b in boxes]
        areas = [(x2[i]-x1[i]+1)*(y2[i]-y1[i]+1) for i in range(len(boxes))]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            to_remove = []
            for j in order:
                xx1 = max(x1[i], x1[j]); yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j]); yy2 = min(y2[i], y2[j])
                w_int = max(0, xx2-xx1+1); h_int = max(0, yy2-yy1+1)
                inter = w_int * h_int
                union = areas[i] + areas[j] - inter
                if union > 0 and (inter / union) > iou_thresh:
                    to_remove.append(j)
            order = [o for o in order if o not in to_remove]
        return keep

class BoxDrawer:
    """Draw bounding boxes on a frame."""
    def __init__(self, thickness=2):
        self.thickness = thickness
        self.color = (0, 255, 0)

    def draw(self, frame, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, self.thickness)

class HeadOrientationChecker:
    """Check if a face is frontal by detecting eyes inside the face region."""
    def __init__(self, eye_cascade_path=None, min_eye_count=2):
        if eye_cascade_path is None:
            eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        self.min_eye_count = min_eye_count

    def is_facing_camera(self, frame, box):
        x, y, w, h = box
        h_img, w_img, _ = frame.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + w, w_img), min(y + h, h_img)
        if x2 <= x1 or y2 <= y1:
            return False
        face_img = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return len(eyes) >= self.min_eye_count

class FaceSaver:
    """Save new frontal faces by histogram comparison."""
    def __init__(self, output_dir="faces", hist_threshold=0.9):
        self.output_dir = output_dir
        self.hist_threshold = hist_threshold
        os.makedirs(self.output_dir, exist_ok=True)
        self.known_histograms = []

    def save(self, frame, boxes, orientation_checker):
        h_img, w_img, _ = frame.shape
        for (x, y, w, h) in boxes:
            if not orientation_checker.is_facing_camera(frame, (x,y,w,h)):
                continue
            x1, y1 = max(0,x), max(0,y)
            x2, y2 = min(x+w, w_img), min(y+h, h_img)
            if x2 <= x1 or y2 <= y1:
                continue
            face_img = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            cv2.normalize(hist, hist)
            is_new = True
            for known in self.known_histograms:
                if cv2.compareHist(known, hist, cv2.HISTCMP_CORREL) >= self.hist_threshold:
                    is_new = False
                    break
            if not is_new:
                continue
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(self.output_dir, f"face_{timestamp}.png")
            cv2.imwrite(path, face_img)
            print(f"Saved new face to {path}", flush=True)
            self.known_histograms.append(hist)

class SerialCommunicator:
    """Handle serial connection to Arduino for detection signals."""
    def __init__(self, port, baud_rate=115200, timeout=1):
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=timeout)
            time.sleep(2)  # allow Arduino to reset
            print(f"Serial opened on {port}@{baud_rate}", flush=True)
        except serial.SerialException as e:
            print(f"Failed to open serial port: {e}", flush=True)
            self.ser = None

    def send(self, detected):
        """Always send '1' if face detected, '0' otherwise."""
        if not self.ser:
            return
        msg = b'1' if detected else b'0'
        try:
            # Send every frame, sin chequear last_state
            self.ser.write(msg)
            self.ser.flush()
            print(f"Sent to Arduino: {msg}", flush=True)
        except Exception as e:
            print(f"Serial write error: {e}", flush=True)


class FaceBoxApp:
    """Main loop: capture, detect, draw, save, and notify Arduino."""
    def __init__(self, webcam, detector, drawer, orientation_checker, saver, communicator):
        self.webcam = webcam
        self.detector = detector
        self.drawer = drawer
        self.orientation_checker = orientation_checker
        self.saver = saver
        self.communicator = communicator

    def run(self):
        try:
            while True:
                frame = self.webcam.read()
                boxes = self.detector.detect(frame)
                detected = len(boxes) > 0
                # Notify Arduino
                self.communicator.send(detected)
                if detected:
                    self.saver.save(frame, boxes, self.orientation_checker)
                # print(f"Detected {len(boxes)} face(s)", flush=True)
                self.drawer.draw(frame, boxes)
                cv2.imshow("Face Boxes", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        finally:
            self.webcam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    SERIAL_PORT = "/dev/cu.usbserial-140"  # adjust to your port
    try:
        cam = Webcam(src=0, width=1280, height=720)
        detector = FaceDetector(min_confidence=0.3, model_selection=1, nms_threshold=0.3)
        drawer = BoxDrawer(thickness=3)
        orientation_checker = HeadOrientationChecker(min_eye_count=2)
        saver = FaceSaver(output_dir="faces", hist_threshold=0.9)
        communicator = SerialCommunicator(port=SERIAL_PORT, baud_rate=115200)
        app = FaceBoxApp(cam, detector, drawer, orientation_checker, saver, communicator)
        app.run()
    except Exception as e:
        print(f"Error: {e}", flush=True)
