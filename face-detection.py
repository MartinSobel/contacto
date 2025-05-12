import cv2
import mediapipe as mp 

class Webcam:
    """Responsible for capturing frames from a USB webcam with configurable resolution"""
    def __init__(self, src=0, width=1280, height=720):
        # Initialize the video capture; src puede ajustarse si tu cámara no es 0
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
    """Uses MediaPipe Face Detection to find face bounding boxes at long range"""
    def __init__(self, min_confidence=0.3, model_selection=1):
        # model_selection=1 for full-range (hasta 5 m)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )

    def detect(self, frame):
        """
        Detect faces and return list of bounding boxes as (x, y, width, height)
        Coordinates are in pixel values
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        boxes = []
        h, w, _ = frame.shape

        if results.detections:
            for detection in results.detections:
                rel_bbox = detection.location_data.relative_bounding_box
                x = int(rel_bbox.xmin * w)
                y = int(rel_bbox.ymin * h)
                box_w = int(rel_bbox.width * w)
                box_h = int(rel_bbox.height * h)
                boxes.append((x, y, box_w, box_h))
        return boxes

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
        detector = FaceDetector(min_confidence=0.3, model_selection=1)
        drawer = BoxDrawer(thickness=3)
        app = FaceBoxApp(cam, detector, drawer)
        app.run()
    except Exception as e:
        # Print any error to console
        print(f"Error: {e}", flush=True)