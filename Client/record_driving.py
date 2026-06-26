"""
Enregistrement de circuits en conduite manuelle.

Usage:
    python3 record_driving.py --output data_circuit.csv
    python3 record_driving.py --output data_circuit.csv --dry-run   # sans VESC
    python3 record_driving.py --output data_circuit.csv --stream    # avec camera live

Conduis normalement avec la manette. Appuie sur START pour arreter.
Les donnees (raycasts + ton braquage) sont sauvegardees dans le CSV.
Ensuite utilise train_corner_model.py pour entrainer le modele.
"""
import argparse
import csv
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "Gamepad"))

import cv2
import Gamepad
import numpy as np

from live_perception import WhiteTapePerception, draw_debug, PerceptionResult
from live_settings import DEFAULT_CONFIG, load_settings
from oak_camera import OakCamera
from vesc_control import VescController


_stream_lock = threading.Lock()
_latest_jpeg = b""


def _stream_server(port):
    try:
        from flask import Flask, Response
        app = Flask(__name__)

        @app.route("/")
        def index():
            return '<h1>Robocar Record</h1><img src="/video">'

        @app.route("/video")
        def video():
            def generate():
                while True:
                    with _stream_lock:
                        frame = _latest_jpeg
                    if frame:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                        )
                    time.sleep(0.05)
            return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

        app.run(host="0.0.0.0", port=port, threaded=True)
    except ImportError:
        print("[stream] Flask non installe")


def normalize_trigger(value):
    if value < 0:
        return (value + 1.0) / 2.0
    return value


def apply_deadzone(value, deadzone=0.05):
    if abs(value) < deadzone:
        return 0.0
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data_circuit.csv"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-port", type=int, default=5000)
    parser.add_argument("--max-fps", type=float, default=20.0)
    return parser.parse_args()


def main():
    global _latest_jpeg
    args = parse_args()
    settings = load_settings(args.config)
    perception = WhiteTapePerception(settings.perception)

    gamepad = Gamepad.Xbox360()
    gamepad.startBackgroundUpdates()
    print("Manette connectee. START = arreter l'enregistrement.")

    vesc = None if args.dry_run else VescController(settings.vesc)
    min_frame_time = 1.0 / max(args.max_fps, 1.0)

    if args.stream:
        t = threading.Thread(target=_stream_server, args=(args.stream_port,), daemon=True)
        t.start()
        print(f"Stream: http://<IP_PI>:{args.stream_port}")

    n_rays = settings.perception.n_rays
    fieldnames = [f"ray{i}" for i in range(n_rays)] + ["steering", "throttle"]

    row_count = 0
    output_path = args.output

    print(f"Enregistrement vers {output_path}")
    print("Conduis ! Appuie sur START pour arreter.")

    try:
        with OakCamera(settings.camera) as camera, \
             open(output_path, "w", newline="") as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            last = time.monotonic()

            while gamepad.isConnected():
                if gamepad.beenPressed("START"):
                    print(f"Arret. {row_count} frames enregistrees.")
                    break

                frame_bgr = camera.read()
                result = perception.process(frame_bgr)

                steering = apply_deadzone(gamepad.axis("LX"))
                rt = normalize_trigger(gamepad.axis("RT"))
                lt = normalize_trigger(gamepad.axis("LT"))
                throttle = rt - lt

                if vesc is not None:
                    vesc.set_steering(steering)
                    throttle_cmd = throttle * settings.vesc.max_duty
                    vesc.set_motor(max(0.0, throttle_cmd))

                if result.mask_fraction > settings.controller.lost_mask_fraction:
                    row = {f"ray{i}": float(result.raycast[i]) for i in range(n_rays)}
                    row["steering"] = steering
                    row["throttle"] = throttle
                    writer.writerow(row)
                    row_count += 1

                if args.stream:
                    dbg = _draw_record_debug(result, steering, throttle, row_count)
                    ok, jpeg = cv2.imencode(".jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        with _stream_lock:
                            _latest_jpeg = jpeg.tobytes()

                elapsed = time.monotonic() - last
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
                last = time.monotonic()

    except KeyboardInterrupt:
        print(f"Interrompu. {row_count} frames enregistrees.")
    finally:
        gamepad.disconnect()
        if vesc is not None:
            vesc.stop()
            vesc.close()

    print(f"Sauvegarde: {output_path} ({row_count} lignes)")


def _draw_record_debug(result, steering, throttle, count):
    import math
    frame = result.frame_bgr.copy()
    frame[result.mask_rejected] = (255, 80, 0)
    frame[result.mask] = (0, 0, 255)

    h, w = result.mask.shape
    ox, oy = w / 2.0, h - 1.0
    n_rays = len(result.raycast)
    angles = np.linspace(0.0, 180.0, n_rays) if n_rays > 1 else np.array([90.0])

    for idx, distance in enumerate(result.raycast):
        rad = math.radians(float(angles[idx]))
        x = int(ox + float(distance) * math.cos(rad))
        y = int(oy - float(distance) * math.sin(rad))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        cv2.line(frame, (int(ox), int(oy)), (x, y), (255, 180, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    cv2.putText(frame, f"REC  steer={steering:.2f} thr={throttle:.2f}  n={count}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    main()
