# REAL-TIME-GETURE-RECOGNITION-USING-3D-CNN-s
✋ Hand Gesture Recognition & Control System
A computer vision-based real-time hand gesture recognition system that allows users to control applications (like switching browser tabs) using hand gestures. Built using OpenCV, MediaPipe, TensorFlow/Keras, and Flask.

📌 Features
🎥 Real-time hand gesture detection using webcam

🧠 CNN-based gesture recognition (trained on custom data)

✋ Hand tracking using MediaPipe

📊 Custom training pipeline for gesture classification

🌐 Flask web app to visualize detection in the browser

🔁 Gesture-to-action mapping (e.g., switch tabs with swipe)

🧾 Table of Contents
Project Structure
Installation
Dataset Collection
Model Training
Running the Application
Usage
Model Architecture
License

📂 Project Structure

.
├── app.py                  # Flask app to serve live video feed
├── camera.py               # Handles frame capture and routing to recognizer/tracker
├── data_collector.py       # Script for collecting labeled hand gesture images
├── recognizer.py           # Gesture recognizer using CNN
├── tracker.py              # Hand gesture tracker using MediaPipe
├── train.py                # Training script for CNN model
├── templates/
│   └── index.html          # Frontend HTML for Flask app
├── models/
│   ├── gesture-model.json      # Saved CNN model architecture
│   ├── gesture-model.weights.h5 # Trained weights
│   └── training_history.png    # Training loss and accuracy graph
└── data/
    ├── train/
    └── test/               # Organized gesture images by label


    
🛠️ Installation
1. Clone the Repository
https://github.com/naqeebahmedkhan-code/REAL-TIME-GETURE-RECOGNITION-USING-3D-CNN-s.git
cd gesture-control
2. Create a Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
You may need to install opencv-python, tensorflow, mediapipe, flask, pyautogui, matplotlib, etc.

🎯 Dataset Collection
Run the following to collect training data for gestures:
python data_collector.py
Prompts for mode: train or test
Press:

0 → move-right

1 → move-left

2 → fist

3 → no-gesture

ESC to exit

Captured grayscale thresholded ROI images are saved in data/train/ or data/test/.

🧠 Model Training

Train the CNN model using:
python train.py
Saves trained model in models/
Saves model weights and architecture
Outputs training accuracy/loss graph as training_history.png

🚀 Running the Application
Start the web server and open a browser to visualize the live gesture detection:
python app.py
Then go to: http://127.0.0.1:5000/

Use the UI to see live feedback:

Default mode: CNN-based recognizer

Switch mode via /set_mode/tracker or /set_mode/cnn URL

🎮 Usage
Recognizer Mode (cnn):

Detects gestures inside a defined ROI using the trained model.

Tracker Mode (tracker):

Uses MediaPipe to identify gestures (e.g., number of fingers, swipe direction).

Automatically switches tabs using pyautogui.

🏗️ Model Architecture
CNN Model Summary:

Conv2D (32) → MaxPool

Conv2D (64) → MaxPool

Conv2D (128) → MaxPool

Flatten → Dense (256) → Dropout → Dense (4 - Softmax)

📸 Sample Gestures
Gesture	Keybind	Description
move-right	0	Swipe gesture to the right
move-left	1	Swipe gesture to the left
fist	2	Closed fist gesture
no-gesture	3	Neutral/no gesture shown

✅ Requirements
Python 3.7+
TensorFlow 2.x
OpenCV
Flask
MediaPipe
PyAutoGUI
Matplotlib

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙌 Acknowledgements
TensorFlow
MediaPipe
OpenCV
Flask
