import cv2
import os

def create_directories(mode):
    gestures = [
        'move-right', 'move-left', 'fist', 'no-gesture'
    ]
    base_path = os.path.join('data', mode)
    os.makedirs(base_path, exist_ok=True)
    for gesture in gestures:
        os.makedirs(os.path.join(base_path, gesture), exist_ok=True)

def capture_images():
    while True:
        mode = input("Enter mode (train/test): ").strip().lower()
        if mode in ['train', 'test']:
            break
        print("Invalid mode! Please enter 'train' or 'test'")

    create_directories(mode)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    x1, y1, x2, y2 = 300, 50, 600, 350

    gesture_map = {
        ord('0'): 'move-right',
        ord('1'): 'move-left',
        ord('2'): 'fist',
        ord('3'): 'no-gesture'
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0, 255, 0), 2)

            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

            counts = {}
            for gesture in gesture_map.values():
                path = os.path.join('data', mode, gesture)
                counts[gesture] = len(os.listdir(path))

            cv2.putText(frame, f"Mode: {mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset = 60
            for gesture, count in counts.items():
                cv2.putText(frame, f"{gesture}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 30

            cv2.imshow("Frame", frame)
            cv2.imshow("ROI", threshold)

            key = cv2.waitKey(10)
            if key == 27:  # ESC
                break
            elif key in gesture_map:
                gesture_dir = os.path.join('data', mode, gesture_map[key])
                count = len(os.listdir(gesture_dir))
                cv2.imwrite(f'{gesture_dir}/{count}.jpg', threshold)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
