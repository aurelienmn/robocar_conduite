from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from live_settings import DEFAULT_CONFIG, load_settings
from oak_camera import OakCamera
from vesc_control import VescController, clamp


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "Gamepad"))

import Gamepad  # noqa: E402


def normalize_trigger(value: float) -> float:
    if value < 0:
        return (value + 1.0) / 2.0
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record real camera frames and manual commands.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "real")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not open the VESC, only record commands.")
    parser.add_argument("--max-fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    source_id = args.source_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / source_id
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.csv"

    gamepad = Gamepad.PS4()
    gamepad.startBackgroundUpdates()
    vesc = None if args.dry_run else VescController(settings.vesc)

    min_frame_time = 1.0 / max(args.max_fps, 1.0)
    last = time.monotonic()
    frame_idx = 0

    print("Enregistrement donnees reelles")
    print(f"run_dir={run_dir}")
    print(f"dry_run={args.dry_run}")
    print("Stick gauche = direction | R2 = avancer | L2 = reculer | OPTIONS = arret")

    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as f, OakCamera(settings.camera) as camera:
            writer = csv.writer(f)
            writer.writerow(["image_path", "throttle", "steering", "frame_idx", "source_id"])

            while gamepad.isConnected():
                frame_bgr = camera.read()

                steering = clamp(gamepad.axis("LEFT-X"), -1.0, 1.0)
                rt = normalize_trigger(gamepad.axis("R2"))
                lt = normalize_trigger(gamepad.axis("L2"))
                throttle = clamp((rt - lt) * settings.vesc.max_duty, -settings.vesc.max_duty, settings.vesc.max_duty)

                if vesc is not None:
                    vesc.set_steering(steering)
                    vesc.set_motor(throttle)

                image_rel = Path("frames") / f"frame_{frame_idx:06d}.png"
                image_path = run_dir / image_rel
                if not cv2.imwrite(str(image_path), frame_bgr):
                    raise RuntimeError(f"Could not write {image_path}")

                writer.writerow([image_rel, throttle, steering, frame_idx, source_id])
                if frame_idx % 20 == 0:
                    print(f"{frame_idx} frames | throttle={throttle:.3f} steering={steering:.2f}")

                if gamepad.beenPressed("OPTIONS"):
                    break

                frame_idx += 1
                elapsed = time.monotonic() - last
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
                last = time.monotonic()

    finally:
        if vesc is not None:
            print("Arret securite VESC")
            vesc.stop()
            vesc.close()
        gamepad.disconnect()


if __name__ == "__main__":
    main()
