from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLIENT_DIR = Path(__file__).resolve().parent
ROOT = CLIENT_DIR.parent
for import_path in (CLIENT_DIR, ROOT / "Gamepad"):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.append(import_path_str)

import Gamepad  # noqa: E402

from live_perception import PerceptionResult, WhiteTapePerception, draw_debug  # noqa: E402
from live_controller import RaycastLineFollower  # noqa: E402
from live_settings import DEFAULT_CONFIG, LiveSettings, load_settings  # noqa: E402
from oak_camera import OakCamera  # noqa: E402
from vesc_control import VescController  # noqa: E402


MAX_RAYS = 15
MAX_FOV = 180.0
FEATURE_VERSION = 1

DEFAULT_DATASET_DIR = ROOT / "data" / "behavioral_pilot"
DEFAULT_MODEL_PATH = ROOT / "models" / "behavioral_pilot.npz"
DEFAULT_LOG_DIR = ROOT / "logs" / "behavioral_pilot"

_stream_lock = threading.Lock()
_latest_jpeg = b""


class LogitechGamepad(Gamepad.Gamepad):
    """Logitech F310/F710 mapping in XInput mode."""

    fullName = "Logitech F310/F710 controller"

    def __init__(self, joystickNumber: int = 0):
        Gamepad.Gamepad.__init__(self, joystickNumber)
        self.axisNames = {
            0: "LX",
            1: "LY",
            2: "LT",
            3: "RX",
            4: "RY",
            5: "RT",
            6: "DX",
            7: "DY",
        }
        self.buttonNames = {
            0: "A",
            1: "B",
            2: "X",
            3: "Y",
            4: "LB",
            5: "RB",
            6: "BACK",
            7: "START",
            8: "LOGITECH",
            9: "LA",
            10: "RA",
        }
        self._setupReverseMaps()


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_trigger(value: float) -> float:
    if value < 0.0:
        return (value + 1.0) / 2.0
    return value


def apply_deadzone(value: float, deadzone: float = 0.05) -> float:
    if abs(value) < deadzone:
        return 0.0
    return clamp(value, -1.0, 1.0)


def build_gamepad(kind: str):
    mapping = {
        "xbox360": Gamepad.Xbox360,
        "logitech": LogitechGamepad,
        "ps4": Gamepad.PS4,
        "ps3": Gamepad.PS3,
        "generic": Gamepad.Gamepad,
    }
    if kind == "auto":
        kind = "xbox360"
    cls = mapping[kind]
    gamepad = cls()
    gamepad.startBackgroundUpdates()
    return gamepad


def safe_axis(gamepad, names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        try:
            return float(gamepad.axis(name))
        except Exception:
            continue
    return default


def safe_been_pressed(gamepad, names: Sequence[str]) -> bool:
    for name in names:
        try:
            if gamepad.beenPressed(name):
                return True
        except Exception:
            continue
    return False


def configured_settings(
    settings: LiveSettings,
    n_rays: int,
    fov: float,
    camera_source: Optional[str] = None,
) -> LiveSettings:
    safe_n_rays = int(clamp(float(n_rays), 1.0, float(MAX_RAYS)))
    safe_fov = clamp(float(fov), 1.0, MAX_FOV)
    settings = replace(
        settings,
        perception=replace(settings.perception, n_rays=safe_n_rays, fov=safe_fov),
    )
    if camera_source is not None:
        settings = replace(settings, camera=replace(settings.camera, source=camera_source))
    return settings


def _stream_server(port: int) -> None:
    try:
        from flask import Flask, Response

        app = Flask(__name__)

        @app.route("/")
        def index():
            return '<h1>Robocar Behavioral Pilot</h1><img src="/video">'

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
        print("[stream] Flask is not installed, stream disabled")


def start_stream_if_needed(enabled: bool, port: int) -> None:
    if not enabled:
        return
    thread = threading.Thread(target=_stream_server, args=(port,), daemon=True)
    thread.start()
    print("Stream camera: http://<IP_PI>:{}".format(port))


def encode_stream_frame(frame_bgr: np.ndarray) -> None:
    global _latest_jpeg
    ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if ok:
        with _stream_lock:
            _latest_jpeg = jpeg.tobytes()


def ray_angles(n_rays: int, fov: float) -> np.ndarray:
    if n_rays <= 1:
        return np.array([90.0], dtype=np.float32)
    safe_fov = clamp(float(fov), 1.0, MAX_FOV)
    return (
        np.arange(n_rays, dtype=np.float32) * (safe_fov / (n_rays - 1))
        + (180.0 - safe_fov) / 2.0
    )


def ray_safety_stats(rays: np.ndarray) -> Dict[str, float]:
    distances = np.asarray(rays, dtype=np.float32).reshape(-1)
    if distances.size == 0:
        return {
            "front_min": 0.0,
            "front_mean": 0.0,
            "left_min": 0.0,
            "right_min": 0.0,
            "left_mean": 0.0,
            "right_mean": 0.0,
        }
    middle = distances.size // 2
    front = distances[max(0, middle - 1): min(distances.size, middle + 2)]
    right = distances[:middle]
    left = distances[middle + 1:]
    max_distance = float(max(distances.max(), 1.0))
    return {
        "front_min": float(front.min()) if front.size else max_distance,
        "front_mean": float(front.mean()) if front.size else max_distance,
        "left_min": float(left.min()) if left.size else max_distance,
        "right_min": float(right.min()) if right.size else max_distance,
        "left_mean": float(left.mean()) if left.size else max_distance,
        "right_mean": float(right.mean()) if right.size else max_distance,
    }


def build_feature_vector(
    rays: np.ndarray,
    mask_fraction: float,
    track_center: float,
    track_heading: float,
    track_confidence: float,
    track_width_px: float,
    prev_steering: float,
    throttle: float,
    ray_scale_px: float,
) -> np.ndarray:
    distances = np.asarray(rays, dtype=np.float32).reshape(-1)
    scale = max(float(ray_scale_px), 1.0)
    norm = np.clip(distances / scale, 0.0, 1.6)

    n_rays = norm.size
    middle = n_rays // 2
    right = norm[:middle]
    left = norm[middle + 1:]
    pairs = min(right.size, left.size)
    if pairs > 0:
        pair_balance = right[:pairs] - left[::-1][:pairs]
    else:
        pair_balance = np.zeros(0, dtype=np.float32)

    stats = ray_safety_stats(distances)
    left_mean = stats["left_mean"] / scale
    right_mean = stats["right_mean"] / scale
    left_min = stats["left_min"] / scale
    right_min = stats["right_min"] / scale
    front_min = stats["front_min"] / scale
    front_mean = stats["front_mean"] / scale
    balance = (left_mean - right_mean) / max(left_mean + right_mean, 1e-6)

    target_idx = int(np.argmax(distances)) if distances.size else 0
    target_idx_norm = 0.0 if n_rays <= 1 else (target_idx / (n_rays - 1)) * 2.0 - 1.0
    target_distance = float(distances[target_idx] / scale) if distances.size else 0.0

    summary = np.array(
        [
            clamp(left_mean, 0.0, 1.6),
            clamp(right_mean, 0.0, 1.6),
            clamp(left_min, 0.0, 1.6),
            clamp(right_min, 0.0, 1.6),
            clamp(front_min, 0.0, 1.6),
            clamp(front_mean, 0.0, 1.6),
            clamp(balance, -1.6, 1.6),
            clamp(target_idx_norm, -1.0, 1.0),
            clamp(target_distance, 0.0, 1.6),
            clamp(float(mask_fraction) / 0.02, 0.0, 2.5),
            clamp(float(track_center), -1.0, 1.0),
            clamp(float(track_heading), -1.0, 1.0),
            clamp(float(track_confidence), 0.0, 1.0),
            clamp(float(track_width_px) / scale, 0.0, 2.0),
            clamp(float(prev_steering), -1.0, 1.0),
            clamp(float(throttle), -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([norm.astype(np.float32), pair_balance.astype(np.float32), summary])


def feature_dim_for_rays(n_rays: int) -> int:
    dummy = np.ones(int(n_rays), dtype=np.float32)
    return int(
        build_feature_vector(
            dummy,
            mask_fraction=0.0,
            track_center=0.0,
            track_heading=0.0,
            track_confidence=0.0,
            track_width_px=0.0,
            prev_steering=0.0,
            throttle=0.0,
            ray_scale_px=1.0,
        ).size
    )


def light_augmented_frames(
    frame_bgr: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> Iterable[Tuple[str, np.ndarray]]:
    if count <= 0:
        return
    frame = frame_bgr.astype(np.float32)
    for idx in range(count):
        alpha = float(rng.uniform(0.72, 1.32))
        beta = float(rng.uniform(-34.0, 34.0))
        gamma = float(rng.uniform(0.72, 1.34))
        augmented = np.clip(frame * alpha + beta, 0.0, 255.0)
        augmented = np.power(augmented / 255.0, gamma) * 255.0
        if rng.random() < 0.35:
            noise = rng.normal(0.0, 3.0, size=augmented.shape)
            augmented = augmented + noise
        augmented = np.clip(augmented, 0.0, 255.0).astype(np.uint8)
        name = "light{}_a{:.2f}_b{:.0f}_g{:.2f}".format(idx + 1, alpha, beta, gamma)
        yield name, augmented


def dataset_fieldnames(n_rays: int) -> List[str]:
    fields = [
        "session_id",
        "direction",
        "sample_idx",
        "timestamp_s",
        "iso_time",
        "elapsed_s",
        "dt_s",
        "synthetic",
        "augment",
        "steering",
        "prev_steering",
        "throttle",
        "throttle_duty",
        "mask_fraction",
        "track_center",
        "track_heading",
        "track_confidence",
        "track_width_px",
        "front_min",
        "front_mean",
        "left_min",
        "right_min",
        "left_mean",
        "right_mean",
        "camera_source",
        "n_rays",
        "fov",
        "frame_w",
        "frame_h",
    ]
    fields.extend(["ray{}".format(idx) for idx in range(n_rays)])
    return fields


def make_dataset_row(
    result: PerceptionResult,
    session_id: str,
    direction: str,
    sample_idx: int,
    start_wall_s: float,
    start_mono_s: float,
    dt_s: Optional[float],
    synthetic: int,
    augment: str,
    steering: float,
    prev_steering: float,
    throttle: float,
    throttle_duty: float,
    camera_source: str,
) -> Dict[str, Any]:
    now_wall_s = time.time()
    stats = ray_safety_stats(result.raycast)
    row: Dict[str, Any] = {
        "session_id": session_id,
        "direction": direction,
        "sample_idx": sample_idx,
        "timestamp_s": "{:.6f}".format(now_wall_s),
        "iso_time": datetime.fromtimestamp(now_wall_s).isoformat(timespec="milliseconds"),
        "elapsed_s": "{:.6f}".format(time.monotonic() - start_mono_s),
        "dt_s": "" if dt_s is None else "{:.6f}".format(dt_s),
        "synthetic": int(synthetic),
        "augment": augment,
        "steering": "{:.7f}".format(clamp(steering, -1.0, 1.0)),
        "prev_steering": "{:.7f}".format(clamp(prev_steering, -1.0, 1.0)),
        "throttle": "{:.7f}".format(clamp(throttle, -1.0, 1.0)),
        "throttle_duty": "{:.7f}".format(throttle_duty),
        "mask_fraction": "{:.8f}".format(result.mask_fraction),
        "track_center": "{:.7f}".format(result.track_center_offset),
        "track_heading": "{:.7f}".format(result.track_heading),
        "track_confidence": "{:.7f}".format(result.track_confidence),
        "track_width_px": "{:.3f}".format(result.track_width_px),
        "front_min": "{:.3f}".format(stats["front_min"]),
        "front_mean": "{:.3f}".format(stats["front_mean"]),
        "left_min": "{:.3f}".format(stats["left_min"]),
        "right_min": "{:.3f}".format(stats["right_min"]),
        "left_mean": "{:.3f}".format(stats["left_mean"]),
        "right_mean": "{:.3f}".format(stats["right_mean"]),
        "camera_source": camera_source,
        "n_rays": int(len(result.raycast)),
        "fov": "{:.3f}".format(result.ray_fov),
        "frame_w": int(result.frame_bgr.shape[1]),
        "frame_h": int(result.frame_bgr.shape[0]),
    }
    for idx, distance in enumerate(result.raycast):
        row["ray{}".format(idx)] = "{:.3f}".format(float(distance))
    return row


def draw_behavior_debug(
    result: PerceptionResult,
    throttle: float,
    steering: float,
    reason: str,
    mode: str,
    recording: bool = False,
    samples: int = 0,
) -> np.ndarray:
    frame = draw_debug(result, throttle, steering, reason)
    status = "{} {}".format(mode, "REC" if recording else "IDLE")
    cv2.putText(
        frame,
        "{} samples={}".format(status, samples),
        (12, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255) if recording else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def maybe_show_window(window_name: str, frame: np.ndarray, enabled: bool) -> Optional[int]:
    if not enabled:
        return None
    try:
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        print("OpenCV window disabled: {}".format(exc))
        return None


def run_record(args: argparse.Namespace) -> None:
    global _latest_jpeg

    settings = configured_settings(
        load_settings(args.config),
        n_rays=args.n_rays,
        fov=args.fov,
        camera_source=args.camera_source,
    )
    perception = WhiteTapePerception(settings.perception)

    direction = args.direction
    dataset_dir = resolve_path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    session_id = "{}_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), direction)
    output_path = resolve_path(args.output) if args.output else dataset_dir / "{}.csv".format(session_id)

    rng = np.random.default_rng(args.seed)
    record_buttons = parse_csv_list(args.record_button)
    quit_buttons = parse_csv_list(args.quit_button)
    steering_axes = parse_csv_list(args.steering_axis)
    rt_axes = parse_csv_list(args.throttle_axis)
    lt_axes = parse_csv_list(args.brake_axis)

    start_stream_if_needed(args.stream, args.stream_port)

    gamepad = build_gamepad(args.gamepad)
    vesc = None if args.dry_run else VescController(settings.vesc)
    min_frame_time = 1.0 / max(args.max_fps, 1.0)

    print("Record mode")
    print("session_id={}".format(session_id))
    print("direction={}".format(direction))
    print("output={}".format(output_path))
    print("n_rays={} fov={}".format(settings.perception.n_rays, settings.perception.fov))
    print("record button(s)={}".format(",".join(record_buttons)))
    print("quit button(s)={}".format(",".join(quit_buttons)))
    print("dry_run={}".format(args.dry_run))

    recording = False
    row_count = 0
    real_count = 0
    synthetic_count = 0
    previous_steering = 0.0
    last_frame_time: Optional[float] = None
    last_print = time.monotonic()
    start_mono_s = time.monotonic()
    start_wall_s = time.time()

    try:
        with OakCamera(settings.camera) as camera, output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dataset_fieldnames(settings.perception.n_rays))
            writer.writeheader()

            while gamepad.isConnected():
                loop_start = time.monotonic()
                if safe_been_pressed(gamepad, quit_buttons):
                    print("Quit button pressed")
                    break
                if safe_been_pressed(gamepad, record_buttons):
                    recording = not recording
                    print("{} recording at sample {}".format("START" if recording else "STOP", row_count))

                frame_bgr = camera.read()
                now = time.monotonic()
                dt_s = None if last_frame_time is None else now - last_frame_time
                last_frame_time = now
                result = perception.process(frame_bgr)

                steering = apply_deadzone(safe_axis(gamepad, steering_axes))
                rt = normalize_trigger(safe_axis(gamepad, rt_axes))
                lt = normalize_trigger(safe_axis(gamepad, lt_axes))
                throttle = clamp(rt - lt, -1.0, 1.0)
                if throttle >= 0.0:
                    throttle_duty = throttle * settings.vesc.max_duty
                else:
                    throttle_duty = throttle * settings.vesc.max_duty * args.brake_scale

                if vesc is not None:
                    vesc.set_steering(steering)
                    vesc.set_motor(throttle_duty)

                can_save = result.mask_fraction > settings.controller.lost_mask_fraction
                if recording and can_save:
                    row = make_dataset_row(
                        result,
                        session_id=session_id,
                        direction=direction,
                        sample_idx=row_count,
                        start_wall_s=start_wall_s,
                        start_mono_s=start_mono_s,
                        dt_s=dt_s,
                        synthetic=0,
                        augment="real",
                        steering=steering,
                        prev_steering=previous_steering,
                        throttle=throttle,
                        throttle_duty=throttle_duty,
                        camera_source=settings.camera.source,
                    )
                    writer.writerow(row)
                    row_count += 1
                    real_count += 1

                    for aug_name, aug_frame in light_augmented_frames(frame_bgr, rng, args.light_augments):
                        aug_result = perception.process(aug_frame)
                        if aug_result.mask_fraction <= settings.controller.lost_mask_fraction:
                            continue
                        aug_row = make_dataset_row(
                            aug_result,
                            session_id=session_id,
                            direction=direction,
                            sample_idx=row_count,
                            start_wall_s=start_wall_s,
                            start_mono_s=start_mono_s,
                            dt_s=dt_s,
                            synthetic=1,
                            augment=aug_name,
                            steering=steering,
                            prev_steering=previous_steering,
                            throttle=throttle,
                            throttle_duty=throttle_duty,
                            camera_source=settings.camera.source,
                        )
                        writer.writerow(aug_row)
                        row_count += 1
                        synthetic_count += 1

                    if row_count % max(1, args.flush_every) == 0:
                        csvfile.flush()

                debug = None
                if args.stream or args.window:
                    reason = "recording" if recording else "manual"
                    debug = draw_behavior_debug(
                        result,
                        throttle_duty,
                        steering,
                        reason,
                        mode="RECORD",
                        recording=recording,
                        samples=row_count,
                    )
                if args.stream and debug is not None:
                    encode_stream_frame(debug)
                key = maybe_show_window("behavioral record", debug, args.window) if debug is not None else None
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    recording = not recording
                    print("{} recording at sample {}".format("START" if recording else "STOP", row_count))

                previous_steering = steering
                if time.monotonic() - last_print >= 1.0:
                    print(
                        "recording={} rows={} real={} synthetic={} steer={:.2f} thr={:.2f} mask={:.4f}".format(
                            recording,
                            row_count,
                            real_count,
                            synthetic_count,
                            steering,
                            throttle,
                            result.mask_fraction,
                        ),
                        flush=True,
                    )
                    last_print = time.monotonic()

                elapsed = time.monotonic() - loop_start
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)

    except KeyboardInterrupt:
        print("Record interrupted")
    finally:
        try:
            gamepad.disconnect()
        except Exception:
            pass
        if vesc is not None:
            print("VESC safety stop")
            vesc.stop()
            vesc.close()
        if args.window:
            cv2.destroyAllWindows()

    print("Saved {} rows to {}".format(row_count, output_path))


def row_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_dataset_files(inputs: Sequence[Path], dataset_dir: Path) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = resolve_path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.exists():
            files.append(path)
    if not files:
        root = resolve_path(dataset_dir)
        files.extend(sorted(root.glob("*.csv")))
    return files


def read_training_rows(files: Sequence[Path], n_rays: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for line_idx, row in enumerate(reader):
                rays = []
                missing = False
                for idx in range(n_rays):
                    key = "ray{}".format(idx)
                    if key not in row or row[key] in ("", None):
                        missing = True
                        break
                    rays.append(row_float(row, key))
                if missing:
                    continue
                steering = row_float(row, "steering", default=float("nan"))
                if not math.isfinite(steering):
                    continue
                rows.append(
                    {
                        "path": str(path),
                        "line_idx": line_idx + 2,
                        "session_id": row.get("session_id") or path.stem,
                        "direction": row.get("direction") or "unknown",
                        "synthetic": int(row_float(row, "synthetic", 0.0)),
                        "rays": np.asarray(rays, dtype=np.float32),
                        "steering": clamp(steering, -1.0, 1.0),
                        "prev_steering": row_float(row, "prev_steering", 0.0),
                        "throttle": row_float(row, "throttle", 0.0),
                        "mask_fraction": row_float(row, "mask_fraction", 0.0),
                        "track_center": row_float(row, "track_center", 0.0),
                        "track_heading": row_float(row, "track_heading", 0.0),
                        "track_confidence": row_float(row, "track_confidence", 0.0),
                        "track_width_px": row_float(row, "track_width_px", 0.0),
                    }
                )
    return rows


def build_training_arrays(
    rows: Sequence[Dict[str, Any]],
    n_rays: int,
    ray_scale_px: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], float]:
    all_rays = np.asarray([row["rays"] for row in rows], dtype=np.float32)
    if ray_scale_px is None or ray_scale_px <= 0.0:
        ray_scale_px = float(max(50.0, np.percentile(all_rays, 95)))
    features = []
    targets = []
    weights = []
    sessions = []
    for row in rows:
        rays = row["rays"]
        feature = build_feature_vector(
            rays,
            mask_fraction=row["mask_fraction"],
            track_center=row["track_center"],
            track_heading=row["track_heading"],
            track_confidence=row["track_confidence"],
            track_width_px=row["track_width_px"],
            prev_steering=row["prev_steering"],
            throttle=row["throttle"],
            ray_scale_px=ray_scale_px,
        )
        if feature.size != feature_dim_for_rays(n_rays):
            continue

        stats = ray_safety_stats(rays)
        closest_side = min(stats["left_min"], stats["right_min"])
        boundary_weight = 0.0
        if closest_side < ray_scale_px * 0.38:
            boundary_weight = (ray_scale_px * 0.38 - closest_side) / max(ray_scale_px * 0.38, 1.0)
        steer_weight = abs(float(row["steering"]))
        weight = 1.0 + 1.3 * steer_weight + 0.9 * boundary_weight
        if int(row["synthetic"]):
            weight *= 0.78

        features.append(feature)
        targets.append(float(row["steering"]))
        weights.append(float(weight))
        sessions.append(str(row["session_id"]))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        all_rays,
        sessions,
        float(ray_scale_px),
    )


def split_indices_by_session(
    sessions: Sequence[str],
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    unique_sessions = np.asarray(sorted(set(sessions)), dtype=object)
    indices = np.arange(len(sessions))
    if unique_sessions.size >= 2:
        rng.shuffle(unique_sessions)
        val_count = max(1, int(round(unique_sessions.size * val_fraction)))
        val_sessions = set(str(item) for item in unique_sessions[:val_count])
        val_mask = np.asarray([session in val_sessions for session in sessions], dtype=bool)
        train_idx = indices[~val_mask]
        val_idx = indices[val_mask]
        if train_idx.size and val_idx.size:
            return train_idx, val_idx, "session"

    shuffled = indices.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count], "row"


def weighted_huber(pred: np.ndarray, target: np.ndarray, weights: np.ndarray, delta: float) -> float:
    err = pred - target
    abs_err = np.abs(err)
    loss = np.where(abs_err <= delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
    return float(np.sum(loss * weights) / max(np.sum(weights), 1e-8))


def metrics(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    err = pred - target
    abs_err = np.abs(err)
    return {
        "mae": float(np.mean(abs_err)),
        "weighted_mae": float(np.sum(abs_err * weights) / max(np.sum(weights), 1e-8)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs_error": float(abs_err.max()) if abs_err.size else 0.0,
    }


def init_mlp(input_dim: int, hidden1: int, hidden2: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    def weight(rows: int, cols: int) -> np.ndarray:
        scale = math.sqrt(2.0 / max(rows + cols, 1))
        return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)

    return {
        "W1": weight(input_dim, hidden1),
        "b1": np.zeros(hidden1, dtype=np.float32),
        "W2": weight(hidden1, hidden2),
        "b2": np.zeros(hidden2, dtype=np.float32),
        "W3": weight(hidden2, 1),
        "b3": np.zeros(1, dtype=np.float32),
    }


def mlp_forward(params: Dict[str, np.ndarray], x: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    z1 = x @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = np.tanh(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    pred = np.tanh(z3[:, 0])
    return pred.astype(np.float32), (x, a1, a2, pred)


def copy_params(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: value.copy() for key, value in params.items()}


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    params = init_mlp(x_train.shape[1], args.hidden1, args.hidden2, rng)
    moments_m = {key: np.zeros_like(value) for key, value in params.items()}
    moments_v = {key: np.zeros_like(value) for key, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    best_params = copy_params(params)
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience

    train_size = x_train.shape[0]
    batch_size = max(8, min(args.batch_size, train_size))

    for epoch in range(1, args.epochs + 1):
        order = np.arange(train_size)
        rng.shuffle(order)
        for start in range(0, train_size, batch_size):
            batch_idx = order[start: start + batch_size]
            xb = x_train[batch_idx]
            if args.feature_noise > 0.0:
                xb = xb + rng.normal(0.0, args.feature_noise, size=xb.shape).astype(np.float32)
            yb = y_train[batch_idx]
            wb = w_train[batch_idx]
            wb = wb / max(float(np.mean(wb)), 1e-8)

            pred, cache = mlp_forward(params, xb)
            x0, a1, a2, pred_cache = cache
            err = pred - yb
            abs_err = np.abs(err)
            grad = np.where(abs_err <= args.huber_delta, err, args.huber_delta * np.sign(err))
            grad = (grad * wb / max(batch_idx.size, 1)).astype(np.float32)
            dz3 = grad * (1.0 - pred_cache * pred_cache)
            dz3 = dz3[:, None]

            grads: Dict[str, np.ndarray] = {}
            grads["W3"] = a2.T @ dz3 + args.weight_decay * params["W3"]
            grads["b3"] = dz3.sum(axis=0)
            da2 = dz3 @ params["W3"].T
            dz2 = da2 * (1.0 - a2 * a2)
            grads["W2"] = a1.T @ dz2 + args.weight_decay * params["W2"]
            grads["b2"] = dz2.sum(axis=0)
            da1 = dz2 @ params["W2"].T
            dz1 = da1 * (1.0 - a1 * a1)
            grads["W1"] = x0.T @ dz1 + args.weight_decay * params["W1"]
            grads["b1"] = dz1.sum(axis=0)

            step += 1
            for key in params:
                moments_m[key] = beta1 * moments_m[key] + (1.0 - beta1) * grads[key]
                moments_v[key] = beta2 * moments_v[key] + (1.0 - beta2) * (grads[key] * grads[key])
                m_hat = moments_m[key] / (1.0 - beta1 ** step)
                v_hat = moments_v[key] / (1.0 - beta2 ** step)
                params[key] = params[key] - args.lr * m_hat / (np.sqrt(v_hat) + eps)

        train_pred, _ = mlp_forward(params, x_train)
        val_pred, _ = mlp_forward(params, x_val)
        train_loss = weighted_huber(train_pred, y_train, w_train, args.huber_delta)
        val_loss = weighted_huber(val_pred, y_val, w_val, args.huber_delta)

        if epoch == 1 or epoch % args.print_every == 0:
            val_mae = metrics(val_pred, y_val, w_val)["weighted_mae"]
            print(
                "epoch={} train_huber={:.5f} val_huber={:.5f} val_wmae={:.5f}".format(
                    epoch,
                    train_loss,
                    val_loss,
                    val_mae,
                ),
                flush=True,
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_params = copy_params(params)
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stop at epoch {}, best epoch {}".format(epoch, best_epoch))
                break

    return {
        "params": best_params,
        "best_epoch": best_epoch,
        "best_val_huber": best_val,
    }


def train_ridge(x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    x_bias = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    sw = np.sqrt(weights).reshape(-1, 1)
    xw = x_bias * sw
    yw = y_train.reshape(-1, 1) * sw
    reg = np.eye(x_bias.shape[1], dtype=np.float32) * float(alpha)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    return beta[:, 0].astype(np.float32)


def ridge_predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_bias = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    return np.clip(x_bias @ beta, -1.0, 1.0).astype(np.float32)


def predict_arrays(
    mlp_params: Dict[str, np.ndarray],
    ridge_beta: np.ndarray,
    blend_mlp: float,
    x: np.ndarray,
) -> np.ndarray:
    mlp_pred, _ = mlp_forward(mlp_params, x)
    ridge_pred = ridge_predict(ridge_beta, x)
    blend = clamp(float(blend_mlp), 0.0, 1.0)
    return np.clip(blend * mlp_pred + (1.0 - blend) * ridge_pred, -1.0, 1.0).astype(np.float32)


def run_train(args: argparse.Namespace) -> None:
    n_rays = int(clamp(float(args.n_rays), 1.0, float(MAX_RAYS)))
    fov = clamp(float(args.fov), 1.0, MAX_FOV)
    files = collect_dataset_files(args.input, args.dataset_dir)
    if not files:
        raise FileNotFoundError("No CSV dataset found")

    print("Training files:")
    for path in files:
        print("  {}".format(path))

    rows = read_training_rows(files, n_rays)
    if len(rows) < args.min_rows:
        raise RuntimeError(
            "Not enough rows: {} found, {} required. Record more laps first.".format(
                len(rows),
                args.min_rows,
            )
        )

    x, y, weights, all_rays, sessions, ray_scale_px = build_training_arrays(
        rows,
        n_rays=n_rays,
        ray_scale_px=args.ray_scale_px,
    )
    if x.shape[0] < args.min_rows:
        raise RuntimeError("Not enough valid feature rows after filtering")

    train_idx, val_idx, split_kind = split_indices_by_session(sessions, args.val_fraction, args.seed)
    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError("Unable to create train/validation split")

    x_train = x[train_idx]
    y_train = y[train_idx]
    w_train = weights[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    w_val = weights[val_idx]

    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    feature_mean = feature_mean.astype(np.float32)

    x_train_z = ((x_train - feature_mean) / feature_std).astype(np.float32)
    x_val_z = ((x_val - feature_mean) / feature_std).astype(np.float32)

    print(
        "rows={} train={} val={} split={} sessions={} feature_dim={} ray_scale_px={:.2f}".format(
            x.shape[0],
            train_idx.size,
            val_idx.size,
            split_kind,
            len(set(sessions)),
            x.shape[1],
            ray_scale_px,
        )
    )
    print(
        "steering min={:.3f} max={:.3f} mean={:.3f}".format(
            float(y.min()),
            float(y.max()),
            float(y.mean()),
        )
    )

    ridge_beta = train_ridge(x_train_z, y_train, w_train, args.ridge_alpha)
    ridge_train = ridge_predict(ridge_beta, x_train_z)
    ridge_val = ridge_predict(ridge_beta, x_val_z)
    print("ridge train={} val={}".format(metrics(ridge_train, y_train, w_train), metrics(ridge_val, y_val, w_val)))

    mlp_result = train_mlp(x_train_z, y_train, w_train, x_val_z, y_val, w_val, args)
    mlp_params = mlp_result["params"]

    blend_mlp = args.blend_mlp
    if blend_mlp < 0.0:
        mlp_val, _ = mlp_forward(mlp_params, x_val_z)
        best_blend = 0.0
        best_score = float("inf")
        for candidate in np.linspace(0.0, 1.0, 21):
            candidate_val = np.clip(
                candidate * mlp_val + (1.0 - candidate) * ridge_val,
                -1.0,
                1.0,
            )
            score = metrics(candidate_val, y_val, w_val)["weighted_mae"]
            if score < best_score:
                best_score = score
                best_blend = float(candidate)
        blend_mlp = best_blend
    train_pred = predict_arrays(mlp_params, ridge_beta, blend_mlp, x_train_z)
    val_pred = predict_arrays(mlp_params, ridge_beta, blend_mlp, x_val_z)
    train_metrics = metrics(train_pred, y_train, w_train)
    val_metrics = metrics(val_pred, y_val, w_val)
    print("blend_mlp={:.2f}".format(blend_mlp))
    print("final train={}".format(train_metrics))
    print("final val={}".format(val_metrics))

    model_path = resolve_path(args.output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "robocar_behavioral_pilot",
        "version": 1,
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_rays": n_rays,
        "fov": fov,
        "max_rays_rule": MAX_RAYS,
        "max_fov_rule": MAX_FOV,
        "ray_scale_px": ray_scale_px,
        "feature_dim": int(x.shape[1]),
        "hidden1": int(args.hidden1),
        "hidden2": int(args.hidden2),
        "blend_mlp": float(blend_mlp),
        "files": [str(path) for path in files],
        "rows": int(x.shape[0]),
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "sessions": int(len(set(sessions))),
        "split": split_kind,
        "best_epoch": int(mlp_result["best_epoch"]),
        "best_val_huber": float(mlp_result["best_val_huber"]),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "input_stats": {
            "ray_min": float(all_rays.min()),
            "ray_max": float(all_rays.max()),
            "ray_p95": float(np.percentile(all_rays, 95)),
        },
    }

    np.savez(
        model_path,
        W1=mlp_params["W1"],
        b1=mlp_params["b1"],
        W2=mlp_params["W2"],
        b2=mlp_params["b2"],
        W3=mlp_params["W3"],
        b3=mlp_params["b3"],
        ridge_beta=ridge_beta,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print("Saved model: {}".format(model_path))


def load_behavior_model(path: Path) -> Dict[str, Any]:
    model_path = resolve_path(path)
    data = np.load(model_path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    params = {
        "W1": data["W1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "W2": data["W2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
        "W3": data["W3"].astype(np.float32),
        "b3": data["b3"].astype(np.float32),
    }
    return {
        "path": model_path,
        "metadata": metadata,
        "params": params,
        "ridge_beta": data["ridge_beta"].astype(np.float32),
        "feature_mean": data["feature_mean"].astype(np.float32),
        "feature_std": data["feature_std"].astype(np.float32),
    }


def model_predict_one(
    model: Dict[str, Any],
    result: PerceptionResult,
    prev_steering: float,
    throttle: float,
) -> float:
    metadata = model["metadata"]
    feature = build_feature_vector(
        result.raycast,
        mask_fraction=result.mask_fraction,
        track_center=result.track_center_offset,
        track_heading=result.track_heading,
        track_confidence=result.track_confidence,
        track_width_px=result.track_width_px,
        prev_steering=prev_steering,
        throttle=throttle,
        ray_scale_px=float(metadata["ray_scale_px"]),
    )
    expected_dim = int(metadata["feature_dim"])
    if feature.size != expected_dim:
        raise RuntimeError("Feature dim mismatch: got {}, expected {}".format(feature.size, expected_dim))
    x = ((feature[None, :] - model["feature_mean"]) / model["feature_std"]).astype(np.float32)
    pred = predict_arrays(
        model["params"],
        model["ridge_beta"],
        float(metadata["blend_mlp"]),
        x,
    )
    return float(pred[0])


def apply_safety_supervisor(
    model_steering: float,
    result: PerceptionResult,
    settings: LiveSettings,
    args: argparse.Namespace,
    dt_s: Optional[float],
) -> Tuple[float, float, str, bool]:
    if dt_s is not None and dt_s > args.stale_timeout:
        return 0.0, 0.0, "stale_frame", True
    if result.mask_fraction < settings.controller.lost_mask_fraction:
        return 0.0, 0.0, "line_lost", True

    stats = ray_safety_stats(result.raycast)
    steering = clamp(float(model_steering), -1.0, 1.0)
    throttle = min(float(args.target_throttle), float(args.hard_max_throttle))
    reason = "model"
    fast_response = False

    closest_side = min(stats["left_min"], stats["right_min"])
    guard_distance = max(float(args.guard_distance_px), 1.0)
    if closest_side < guard_distance:
        proximity = clamp((guard_distance - closest_side) / guard_distance, 0.0, 1.0)
        forced = args.guard_steering * (0.55 + 0.45 * proximity)
        if stats["right_min"] < stats["left_min"] * 0.92:
            steering = min(steering, -forced)
            reason = "guard_right"
            fast_response = True
        elif stats["left_min"] < stats["right_min"] * 0.92:
            steering = max(steering, forced)
            reason = "guard_left"
            fast_response = True

    front_min = stats["front_min"]
    if front_min < args.emergency_distance_px:
        steering = -0.85 if stats["left_mean"] > stats["right_mean"] else 0.85
        throttle = 0.0
        reason = "emergency_front"
        fast_response = True
    elif front_min < args.slow_distance_px:
        ratio = clamp(front_min / max(args.slow_distance_px, 1.0), 0.25, 1.0)
        throttle *= ratio
        if reason == "model":
            reason = "slow_front"

    turn_factor = 1.0 - args.turn_slowdown * clamp(abs(steering), 0.0, 1.0)
    throttle *= clamp(turn_factor, args.min_turn_factor, 1.0)
    throttle = clamp(throttle, 0.0, settings.vesc.max_duty)
    return throttle, steering, reason, fast_response


def make_run_dir(base_dir: Path, dry_run: bool) -> Path:
    base_dir = resolve_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dry" if dry_run else "live"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base_dir / "{}_{}".format(stamp, suffix)
    idx = 2
    while path.exists():
        path = base_dir / "{}_{}_{}".format(stamp, suffix, idx)
        idx += 1
    path.mkdir(parents=True)
    return path


def run_drive(args: argparse.Namespace) -> None:
    model = load_behavior_model(args.model)
    metadata = model["metadata"]
    if int(metadata["n_rays"]) > MAX_RAYS or float(metadata["fov"]) > MAX_FOV:
        raise RuntimeError(
            "Model violates limits: n_rays={} fov={} (max {} rays, {} deg)".format(
                metadata["n_rays"],
                metadata["fov"],
                MAX_RAYS,
                MAX_FOV,
            )
        )
    settings = configured_settings(
        load_settings(args.config),
        n_rays=int(metadata["n_rays"]),
        fov=float(metadata["fov"]),
        camera_source=args.camera_source,
    )
    perception = WhiteTapePerception(settings.perception)
    ray_fallback = RaycastLineFollower(settings.controller)

    start_stream_if_needed(args.stream, args.stream_port)

    run_dir = None if args.no_log else make_run_dir(args.log_dir, args.dry_run)
    log_file = None
    if run_dir is not None:
        (run_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "settings.json").write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log_file = (run_dir / "events.jsonl").open("w", encoding="utf-8")

    gamepad = None
    if not args.no_gamepad_stop:
        try:
            gamepad = build_gamepad(args.gamepad)
            print("Gamepad stop enabled")
        except Exception as exc:
            print("Gamepad stop unavailable: {}".format(exc))

    vesc = None if args.dry_run else VescController(settings.vesc)
    min_frame_time = 1.0 / max(args.max_fps, 1.0)
    quit_buttons = parse_csv_list(args.quit_button)

    print("Drive mode")
    print("model={}".format(model["path"]))
    print("dry_run={}".format(args.dry_run))
    print("target_throttle={:.4f}".format(args.target_throttle))
    print("hard_max_throttle={:.4f}".format(args.hard_max_throttle))
    print("ray_blend={:.2f} steering_gain={:.2f}".format(args.ray_blend, args.steering_gain))
    print("n_rays={} fov={}".format(settings.perception.n_rays, settings.perception.fov))
    if run_dir is not None:
        print("logs={}".format(run_dir))

    previous_prediction_time: Optional[float] = None
    previous_steering = 0.0
    frame_idx = 0
    start_mono_s = time.monotonic()
    start_wall_s = time.time()
    stop_reason = "completed"
    error_payload = None

    try:
        with OakCamera(settings.camera) as camera:
            while True:
                loop_start = time.monotonic()
                if gamepad is not None and safe_been_pressed(gamepad, quit_buttons):
                    stop_reason = "quit_button"
                    break

                frame_bgr = camera.read()
                now = time.monotonic()
                dt_s = None if previous_prediction_time is None else now - previous_prediction_time
                previous_prediction_time = now
                result = perception.process(frame_bgr)

                model_steering = model_predict_one(
                    model,
                    result,
                    prev_steering=previous_steering,
                    throttle=clamp(
                        min(args.target_throttle, args.hard_max_throttle)
                        / max(settings.vesc.max_duty, 1e-6),
                        0.0,
                        1.0,
                    ),
                )
                ray_command = ray_fallback.predict(
                    result.raycast,
                    result.mask_fraction,
                    dt_s=dt_s,
                    ray_fov=result.ray_fov,
                    track_center_offset=result.track_center_offset,
                    track_heading=result.track_heading,
                    track_confidence=result.track_confidence,
                )
                ray_blend = clamp(float(args.ray_blend), 0.0, 1.0)
                hybrid_steering = (
                    (1.0 - ray_blend) * model_steering
                    + ray_blend * ray_command.steering
                )
                hybrid_steering = clamp(hybrid_steering * args.steering_gain, -1.0, 1.0)
                throttle, steering_raw, reason, fast_response = apply_safety_supervisor(
                    hybrid_steering,
                    result,
                    settings,
                    args,
                    dt_s,
                )
                if reason == "model" and ray_blend > 0.0:
                    reason = "hybrid"

                alpha = args.steering_alpha
                if dt_s is not None and args.steering_tau > 0.0:
                    alpha = clamp(dt_s / (args.steering_tau + dt_s), 0.0, 1.0)
                if fast_response:
                    alpha = max(alpha, 0.70)
                steering = previous_steering + alpha * (steering_raw - previous_steering)
                steering = clamp(steering, -1.0, 1.0)
                previous_steering = steering

                if vesc is not None:
                    vesc.set_steering(steering)
                    vesc.set_motor(throttle)

                payload = {
                    "event": "frame",
                    "frame_idx": frame_idx,
                    "timestamp_s": time.time(),
                    "elapsed_s": time.monotonic() - start_mono_s,
                    "dt_s": dt_s,
                    "fps": None if dt_s is None else 1.0 / max(dt_s, 1e-6),
                    "model_steering": model_steering,
                    "ray_steering": ray_command.steering,
                    "ray_reason": ray_command.reason,
                    "hybrid_steering": hybrid_steering,
                    "steering_raw": steering_raw,
                    "steering": steering,
                    "throttle": throttle,
                    "reason": reason,
                    "fast_response": fast_response,
                    "mask_fraction": result.mask_fraction,
                    "track_center": result.track_center_offset,
                    "track_heading": result.track_heading,
                    "track_confidence": result.track_confidence,
                    "raycast": [float(v) for v in result.raycast],
                }
                if log_file is not None:
                    log_file.write(json.dumps(payload, sort_keys=True) + "\n")
                    log_file.flush()

                if args.headless:
                    print(
                        json.dumps(
                            {
                                "steering": steering,
                                "throttle": throttle,
                                "reason": reason,
                                "ray_reason": ray_command.reason,
                                "mask_fraction": result.mask_fraction,
                                "raycast": [int(v) for v in result.raycast],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

                debug = None
                if args.stream or args.window:
                    debug = draw_behavior_debug(
                        result,
                        throttle,
                        steering,
                        reason,
                        mode="DRIVE",
                        recording=False,
                        samples=frame_idx,
                    )
                if args.stream and debug is not None:
                    encode_stream_frame(debug)
                key = maybe_show_window("behavioral drive", debug, args.window) if debug is not None else None
                if key in (27, ord("q")):
                    stop_reason = "keyboard"
                    break

                frame_idx += 1
                if args.frames > 0 and frame_idx >= args.frames:
                    stop_reason = "frame_limit"
                    break

                elapsed = time.monotonic() - loop_start
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Drive interrupted")
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
            print("VESC safety stop")
            vesc.stop()
            vesc.close()
        if gamepad is not None:
            try:
                gamepad.disconnect()
            except Exception:
                pass
        if args.window:
            cv2.destroyAllWindows()
        if log_file is not None:
            summary = {
                "event": "summary",
                "timestamp_s": time.time(),
                "start_timestamp_s": start_wall_s,
                "stop_reason": stop_reason,
                "frames": frame_idx,
                "duration_s": time.monotonic() - start_mono_s,
                "dry_run": args.dry_run,
                "model": str(model["path"]),
                "error": error_payload,
            }
            log_file.write(json.dumps(summary, sort_keys=True) + "\n")
            log_file.flush()
            log_file.close()
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print("summary={}".format(run_dir / "summary.json"))


def add_common_live_args(parser: argparse.ArgumentParser, include_perception: bool = True) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    if include_perception:
        parser.add_argument("--n-rays", type=int, default=MAX_RAYS, help="Max {}.".format(MAX_RAYS))
        parser.add_argument("--fov", type=float, default=MAX_FOV, help="Max {} degrees.".format(MAX_FOV))
    parser.add_argument(
        "--camera-source",
        choices=("rgb_preview", "rgb_video", "rgb_isp", "mono_left", "mono_right"),
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Behavioral cloning pilot: record, train and drive with <=15 rays and <=180 FOV.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Manual driving dataset recorder.")
    add_common_live_args(record)
    record.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    record.add_argument("--output", type=Path, default=None)
    record.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    record.add_argument("--dry-run", action="store_true")
    record.add_argument(
        "--gamepad",
        choices=("auto", "xbox360", "logitech", "ps4", "ps3", "generic"),
        default="auto",
    )
    record.add_argument("--record-button", default="A,CROSS,0")
    record.add_argument("--quit-button", default="START,OPTIONS,7,9")
    record.add_argument("--steering-axis", default="LX,LEFT-X,0")
    record.add_argument("--throttle-axis", default="RT,R2,5")
    record.add_argument("--brake-axis", default="LT,L2,2")
    record.add_argument("--brake-scale", type=float, default=1.0)
    record.add_argument("--light-augments", type=int, default=1)
    record.add_argument("--seed", type=int, default=123)
    record.add_argument("--max-fps", type=float, default=20.0)
    record.add_argument("--flush-every", type=int, default=25)
    record.add_argument("--stream", action="store_true")
    record.add_argument("--stream-port", type=int, default=5000)
    record.add_argument("--window", action="store_true")
    record.set_defaults(func=run_record)

    train = subparsers.add_parser("train", help="Train a compact behavioral pilot model.")
    train.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    train.add_argument("--input", type=Path, nargs="*", default=[])
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument("--n-rays", type=int, default=MAX_RAYS)
    train.add_argument("--fov", type=float, default=MAX_FOV)
    train.add_argument("--min-rows", type=int, default=300)
    train.add_argument("--val-fraction", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--ray-scale-px", type=float, default=None)
    train.add_argument("--hidden1", type=int, default=64)
    train.add_argument("--hidden2", type=int, default=32)
    train.add_argument("--epochs", type=int, default=240)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=0.0025)
    train.add_argument("--weight-decay", type=float, default=0.0008)
    train.add_argument("--ridge-alpha", type=float, default=1.8)
    train.add_argument("--huber-delta", type=float, default=0.12)
    train.add_argument("--feature-noise", type=float, default=0.015)
    train.add_argument("--patience", type=int, default=28)
    train.add_argument("--print-every", type=int, default=10)
    train.add_argument("--blend-mlp", type=float, default=-1.0)
    train.set_defaults(func=run_train)

    drive = subparsers.add_parser("drive", help="Run the trained IA pilot.")
    drive.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    add_common_live_args(drive, include_perception=False)
    drive.add_argument("--dry-run", action="store_true")
    drive.add_argument("--target-throttle", type=float, default=0.045)
    drive.add_argument("--hard-max-throttle", type=float, default=0.045)
    drive.add_argument("--ray-blend", type=float, default=0.0)
    drive.add_argument("--steering-gain", type=float, default=1.25)
    drive.add_argument("--max-fps", type=float, default=12.0)
    drive.add_argument("--frames", type=int, default=0)
    drive.add_argument("--headless", action="store_true")
    drive.add_argument("--stream", action="store_true")
    drive.add_argument("--stream-port", type=int, default=5000)
    drive.add_argument("--window", action="store_true")
    drive.add_argument("--no-log", action="store_true")
    drive.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    drive.add_argument(
        "--gamepad",
        choices=("auto", "xbox360", "logitech", "ps4", "ps3", "generic"),
        default="auto",
    )
    drive.add_argument("--no-gamepad-stop", action="store_true")
    drive.add_argument("--quit-button", default="START,OPTIONS,7,9")
    drive.add_argument("--stale-timeout", type=float, default=1.0)
    drive.add_argument("--guard-distance-px", type=float, default=105.0)
    drive.add_argument("--guard-steering", type=float, default=0.68)
    drive.add_argument("--emergency-distance-px", type=float, default=42.0)
    drive.add_argument("--slow-distance-px", type=float, default=125.0)
    drive.add_argument("--turn-slowdown", type=float, default=0.58)
    drive.add_argument("--min-turn-factor", type=float, default=0.35)
    drive.add_argument("--steering-alpha", type=float, default=0.32)
    drive.add_argument("--steering-tau", type=float, default=0.22)
    drive.set_defaults(func=run_drive)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
