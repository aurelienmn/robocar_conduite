from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from live_driver import LiveDriver
from live_settings import DEFAULT_CONFIG, load_settings
from oak_camera import OakCamera
from vesc_control import VescController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the real car with the live IA line follower.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without opening the VESC.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames. 0 means infinite.")
    parser.add_argument("--max-fps", type=float, default=3.0)
    parser.add_argument(
        "--camera-source",
        choices=("rgb_preview", "rgb_video", "rgb_isp", "mono_left", "mono_right"),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.camera_source is not None:
        settings = replace(settings, camera=replace(settings.camera, source=args.camera_source))
    driver = LiveDriver(settings)

    vesc = None if args.dry_run else VescController(settings.vesc)
    min_frame_time = 1.0 / max(args.max_fps, 1.0)
    last = time.monotonic()
    frame_idx = 0

    print("IA live demarree")
    print(f"dry_run={args.dry_run}")
    print(f"camera_source={settings.camera.source}")
    print(f"n_rays={settings.perception.n_rays} fov={settings.perception.fov}")
    print(f"max_throttle={settings.controller.max_throttle}")

    try:
        with OakCamera(settings.camera) as camera:
            while True:
                frame_bgr = camera.read()
                result = driver.predict_bgr(frame_bgr)

                throttle = result.command.throttle
                steering = result.command.steering

                if vesc is not None:
                    vesc.set_steering(steering)
                    vesc.set_motor(throttle)

                print(
                    json.dumps(
                        {
                            "throttle": throttle,
                            "steering": steering,
                            "confidence": result.command.confidence,
                            "reason": result.command.reason,
                            "camera_source": settings.camera.source,
                            "fps": None if result.dt_s is None else 1.0 / max(result.dt_s, 1e-6),
                            "mask_fraction": result.perception.mask_fraction,
                            "raycast": [int(v) for v in result.perception.raycast],
                        }
                    ),
                    flush=True,
                )

                frame_idx += 1
                if args.frames > 0 and frame_idx >= args.frames:
                    break

                elapsed = time.monotonic() - last
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
                last = time.monotonic()

    except KeyboardInterrupt:
        print("IA live interrompue")

    finally:
        if vesc is not None:
            print("Arret securite VESC")
            vesc.stop()
            vesc.close()


if __name__ == "__main__":
    main()
