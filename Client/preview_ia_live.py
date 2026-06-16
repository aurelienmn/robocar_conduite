from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2

from live_driver import LiveDriver
from live_perception import draw_debug
from live_settings import DEFAULT_CONFIG, load_settings
from oak_camera import OakCamera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview IA live without sending motor commands.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", help="Print JSON instead of opening a window.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames. 0 means infinite.")
    parser.add_argument("--save-debug", type=Path, default=None)
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

    if args.save_debug:
        args.save_debug.mkdir(parents=True, exist_ok=True)

    min_frame_time = 1.0 / max(args.max_fps, 1.0)
    last = time.monotonic()
    frame_idx = 0

    try:
        with OakCamera(settings.camera) as camera:
            while True:
                frame_bgr = camera.read()
                result = driver.predict_bgr(frame_bgr)
                payload = {
                    "throttle": result.command.throttle,
                    "steering": result.command.steering,
                    "confidence": result.command.confidence,
                    "reason": result.command.reason,
                    "camera_source": settings.camera.source,
                    "fps": None if result.dt_s is None else 1.0 / max(result.dt_s, 1e-6),
                    "mask_fraction": result.perception.mask_fraction,
                    "raycast": [int(v) for v in result.perception.raycast],
                }

                if args.headless:
                    print(json.dumps(payload), flush=True)
                    if args.save_debug and frame_idx % 20 == 0:
                        debug = draw_debug(
                            result.perception,
                            result.command.throttle,
                            result.command.steering,
                            result.command.reason,
                        )
                        cv2.imwrite(str(args.save_debug / f"frame_{frame_idx:06d}.png"), debug)
                else:
                    debug = draw_debug(
                        result.perception,
                        result.command.throttle,
                        result.command.steering,
                        result.command.reason,
                    )
                    cv2.imshow("IA live preview", debug)
                    if args.save_debug and frame_idx % 20 == 0:
                        cv2.imwrite(str(args.save_debug / f"frame_{frame_idx:06d}.png"), debug)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_idx += 1
                if args.frames > 0 and frame_idx >= args.frames:
                    break

                elapsed = time.monotonic() - last
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
                last = time.monotonic()
    except KeyboardInterrupt:
        print("Preview IA arretee")

    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
