import cv2
from recognizer import GestureRecognizer
from tracker import HandTracker

class VideoCamera:
    def __init__(self, mode="cnn"):
        self.cap = cv2.VideoCapture(0)
        self.mode = mode
        self.recognizer = GestureRecognizer()
        self.tracker = HandTracker()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        feedback = ""

        if self.mode == "cnn":
            x1, y1, x2, y2 = 300, 50, 600, 350
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
            gesture = self.recognizer.predict(thresh)
            feedback = f"Gesture: {gesture}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        else:
            feedback = self.tracker.handle(frame)

        cv2.putText(frame, feedback, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def release(self):
        self.cap.release()
