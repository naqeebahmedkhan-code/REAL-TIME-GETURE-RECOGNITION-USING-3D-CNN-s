import mediapipe as mp
import pyautogui
import time

class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.alt_pressed = False
        self.last_switch = 0
        self.switch_delay = 1.0

    def count_fingers(self, lm):
        tips = [8, 12, 16, 20]
        return sum(lm.landmark[t].y < lm.landmark[t - 2].y for t in tips)

    def handle(self, frame):
        import cv2
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        center = w // 2
        now = time.time()

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                x = int(lm.landmark[mp.solutions.hands.HandLandmark.WRIST].x * w)
                if self.count_fingers(lm) > 0:
                    if x > center + 100 and now - self.last_switch > self.switch_delay:
                        self.press_alt()
                        pyautogui.press('tab')
                        self.last_switch = now
                        return "➡ Next Tab"
                    elif x < center - 100 and now - self.last_switch > self.switch_delay:
                        self.press_alt()
                        pyautogui.hotkey('shift', 'tab')
                        self.last_switch = now
                        return "⬅ Prev Tab"
                else:
                    self.release_alt()
                    return "✊ Fist – Stop"
        else:
            self.release_alt()
            return "No Hand Detected"

    def press_alt(self):
        if not self.alt_pressed:
            pyautogui.keyDown("alt")
            self.alt_pressed = True

    def release_alt(self):
        if self.alt_pressed:
            pyautogui.keyUp("alt")
            self.alt_pressed = False
