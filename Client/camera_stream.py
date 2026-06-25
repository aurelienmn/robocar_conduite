from flask import Flask, Response
import cv2
import depthai as dai

app = Flask(__name__)

# Initialisation DepthAI...

@app.route("/video")
def video():
    while True:
        # récupérer une image
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

app.run(host="0.0.0.0", port=5000)