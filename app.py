from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)
camera = VideoCamera(mode="cnn")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    camera.mode = mode
    return f"Mode switched to {mode}"

if __name__ == '__main__':
    app.run(debug=True)
