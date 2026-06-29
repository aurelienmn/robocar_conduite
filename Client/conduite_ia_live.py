import argparse
import json
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import cv2

from live_driver import LiveDriver
from live_perception import draw_debug
from live_settings import DEFAULT_CONFIG, load_settings
from oak_camera import OakCamera
from vesc_control import VescController

_stream_lock = threading.Lock()
_latest_jpeg = b""
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = ROOT / "logs" / "ia_runs"


def _stream_server(port):
    try:
        from flask import Flask, Response
        app = Flask(__name__)

        @app.route("/")
        def index():
            return '<h1>Robocar Live</h1><img src="/video">'

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
        print("[stream] Flask non installe, stream desactive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the real car with the live IA line follower.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without opening the VESC.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames. 0 means infinite.")
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--stream", action="store_true", help="Activer le stream camera sur le port 5000.")
    parser.add_argument("--stream-port", type=int, default=5000)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--no-log", action="store_true", help="Desactive les fichiers de log automatiques.")
    parser.add_argument(
        "--camera-source",
        choices=("rgb_preview", "rgb_video", "rgb_isp", "mono_left", "mono_right"),
        default=None,
    )
    return parser.parse_args()


def _make_run_dir(base_dir: Path, dry_run: bool) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dry" if dry_run else "live"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / "{}_{}".format(stamp, suffix)
    idx = 2
    while run_dir.exists():
        run_dir = base_dir / "{}_{}_{}".format(stamp, suffix, idx)
        idx += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frame_payload(frame_idx, result, settings, throttle, steering, start_wall_s, start_mono_s):
    now_wall_s = time.time()
    return {
        "event": "frame",
        "frame_idx": frame_idx,
        "timestamp_s": now_wall_s,
        "elapsed_s": time.monotonic() - start_mono_s,
        "iso_time": datetime.fromtimestamp(now_wall_s).isoformat(timespec="milliseconds"),
        "throttle": throttle,
        "steering": steering,
        "confidence": result.command.confidence,
        "reason": result.command.reason,
        "camera_source": settings.camera.source,
        "fps": None if result.dt_s is None else 1.0 / max(result.dt_s, 1e-6),
        "dt_s": result.dt_s,
        "mask_fraction": result.perception.mask_fraction,
        "track_center": result.perception.track_center_offset,
        "track_heading": result.perception.track_heading,
        "track_confidence": result.perception.track_confidence,
        "track_width_px": result.perception.track_width_px,
        "track_points": [list(point) for point in result.perception.track_points],
        "ray_fov": result.perception.ray_fov,
        "raycast": [int(v) for v in result.perception.raycast],
        "diagnostics": result.command.diagnostics,
    }


def main() -> None:
    global _latest_jpeg

    args = parse_args()
    settings = load_settings(args.config)
    if args.camera_source is not None:
        settings = replace(settings, camera=replace(settings.camera, source=args.camera_source))
    driver = LiveDriver(settings)

    log_file = None
    run_dir = None
    if not args.no_log:
        log_root = args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir
        run_dir = _make_run_dir(log_root, args.dry_run)
        _write_json(run_dir / "config.json", asdict(settings))
        log_file = (run_dir / "events.jsonl").open("w", encoding="utf-8")

    vesc = None
    min_frame_time = 1.0 / max(args.max_fps, 1.0)
    last = time.monotonic()
    frame_idx = 0
    start_mono_s = time.monotonic()
    start_wall_s = time.time()
    stop_reason = "completed"
    error_payload = None
    reason_counts = Counter()
    throttle_values = []
    steering_values = []
    fps_values = []
    mask_values = []

    if args.stream:
        t = threading.Thread(target=_stream_server, args=(args.stream_port,), daemon=True)
        t.start()
        print("Stream camera: http://<IP_PI>:{}".format(args.stream_port))

    print("IA live demarree")
    print("dry_run={}".format(args.dry_run))
    print("camera_source={}".format(settings.camera.source))
    print("n_rays={} fov={}".format(settings.perception.n_rays, settings.perception.fov))
    print("max_throttle={}".format(settings.controller.max_throttle))
    if run_dir is not None:
        print("Logs IA: {}".format(run_dir))
        log_file.write(json.dumps({
            "event": "start",
            "timestamp_s": start_wall_s,
            "iso_time": datetime.fromtimestamp(start_wall_s).isoformat(timespec="milliseconds"),
            "dry_run": args.dry_run,
            "stream": args.stream,
            "max_fps": args.max_fps,
            "frames_limit": args.frames,
            "camera_source": settings.camera.source,
            "config_path": str(args.config),
            "run_dir": str(run_dir),
        }) + "\n")
        log_file.flush()

    try:
        if not args.dry_run:
            vesc = VescController(settings.vesc)
        with OakCamera(settings.camera) as camera:
            while True:
                frame_bgr = camera.read()
                result = driver.predict_bgr(frame_bgr)

                throttle = result.command.throttle
                steering = result.command.steering

                if vesc is not None:
                    vesc.set_steering(steering)
                    vesc.set_motor(throttle)

                if args.stream:
                    debug = draw_debug(result.perception, throttle, steering, result.command.reason)
                    ok, jpeg = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        with _stream_lock:
                            _latest_jpeg = jpeg.tobytes()

                payload = _frame_payload(
                    frame_idx,
                    result,
                    settings,
                    throttle,
                    steering,
                    start_wall_s,
                    start_mono_s,
                )
                if log_file is not None:
                    log_file.write(json.dumps(payload, sort_keys=True) + "\n")
                    log_file.flush()

                reason_counts[result.command.reason] += 1
                throttle_values.append(float(throttle))
                steering_values.append(float(steering))
                mask_values.append(float(result.perception.mask_fraction))
                if payload["fps"] is not None:
                    fps_values.append(float(payload["fps"]))

                compact_payload = {
                    key: payload[key]
                    for key in (
                        "throttle",
                        "steering",
                        "confidence",
                        "reason",
                        "camera_source",
                        "fps",
                        "mask_fraction",
                        "track_center",
                        "track_heading",
                        "track_confidence",
                        "ray_fov",
                        "raycast",
                    )
                }
                print(json.dumps(compact_payload), flush=True)

                frame_idx += 1
                if args.frames > 0 and frame_idx >= args.frames:
                    stop_reason = "frame_limit"
                    break

                elapsed = time.monotonic() - last
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
                last = time.monotonic()

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("IA live interrompue")
    except Exception as exc:
        stop_reason = "exception"
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise

    finally:
        if vesc is not None:
            print("Arret securite VESC")
            vesc.stop()
            vesc.close()
        if log_file is not None:
            end_wall_s = time.time()

            def stats(values):
                if not values:
                    return {"min": None, "max": None, "avg": None}
                return {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                }

            summary = {
                "event": "summary",
                "timestamp_s": end_wall_s,
                "iso_time": datetime.fromtimestamp(end_wall_s).isoformat(timespec="milliseconds"),
                "stop_reason": stop_reason,
                "frames": frame_idx,
                "duration_s": time.monotonic() - start_mono_s,
                "dry_run": args.dry_run,
                "camera_source": settings.camera.source,
                "n_rays": settings.perception.n_rays,
                "fov": settings.perception.fov,
                "reason_counts": dict(reason_counts),
                "throttle": stats(throttle_values),
                "steering": stats(steering_values),
                "abs_steering_avg": (
                    None
                    if not steering_values
                    else sum(abs(v) for v in steering_values) / len(steering_values)
                ),
                "fps": stats(fps_values),
                "mask_fraction": stats(mask_values),
                "error": error_payload,
                "run_dir": str(run_dir),
            }
            stop_event = {**summary, "event": "stop"}
            log_file.write(json.dumps(stop_event, sort_keys=True) + "\n")
            log_file.flush()
            log_file.close()
            _write_json(run_dir / "summary.json", summary)
            print("Summary IA: {}".format(run_dir / "summary.json"))


if __name__ == "__main__":
    main()
