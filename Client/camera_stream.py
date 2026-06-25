# ============================================================
# camera_stream.py
# Lance un serveur HTTP sur le port 5000.
# Ouvre http://<IP_DE_LA_PI>:5000 dans ton navigateur
# pour voir le flux camera avec le mask (rouge) et les rayons.
#
# Lancement : python3 camera_stream.py
# ============================================================

from flask import Flask, Response
import cv2
import depthai as dai

from live_driver import LiveDriver
from live_perception import draw_debug
from live_settings import load_settings

app = Flask(__name__)

settings = load_settings()
driver = LiveDriver(settings)

pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setPreviewSize(426, 240)
cam.setInterleaved(False)
cam.setFps(20)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

device = dai.Device(pipeline)
queue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

@app.route("/")
def index():
    return '<h1>Robocar Camera</h1><img src="/video">'

@app.route("/video")
def video():
    def generate():
        while True:
            packet = queue.get()
            frame = packet.getCvFrame()

            result = driver.predict_bgr(frame)
            debug = draw_debug(
                result.perception,
                result.command.throttle,
                result.command.steering,
                result.command.reason,
            )

            ok, jpeg = cv2.imencode(".jpg", debug)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)