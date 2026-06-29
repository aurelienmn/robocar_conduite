from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from live_controller import RaycastLineFollower
from live_perception import WhiteTapePerception, draw_debug
from live_settings import DEFAULT_CONFIG, PerceptionSettings, load_settings
from oak_camera import OakCamera


def frame_std(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values.reshape(-1), q))


def mask_fraction_for(frame_bgr: np.ndarray, settings: PerceptionSettings) -> float:
    mask, _ = WhiteTapePerception(settings).predict_mask(frame_bgr)
    return float(mask.mean())


def relaxed_settings(settings: PerceptionSettings, value_min: int, sat_max: int, min_rgb: int) -> PerceptionSettings:
    return PerceptionSettings(
        n_rays=settings.n_rays,
        fov=settings.fov,
        min_frame_std=settings.min_frame_std,
        white_value_min=value_min,
        saturation_max=sat_max,
        min_rgb=min_rgb,
        max_rgb_delta=160,
        roi_top_fraction=settings.roi_top_fraction,
        roi_bottom_fraction=settings.roi_bottom_fraction,
        morphology_kernel=settings.morphology_kernel,
        open_iterations=settings.open_iterations,
        dilate_iterations=settings.dilate_iterations,
        min_component_area=settings.min_component_area,
        max_component_area=settings.max_component_area,
        min_bottom_fraction=settings.min_bottom_fraction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one OAK-D frame and print white-tape diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=Path("debug_camera"))
    parser.add_argument("--attempts", type=int, default=30)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    attempt_stats = []
    with OakCamera(settings.camera) as camera:
        frame_bgr = camera.read()
        attempt_stats.append(frame_std(frame_bgr))
        for _ in range(max(0, args.attempts - 1)):
            if frame_std(frame_bgr) >= settings.perception.min_frame_std:
                break
            frame_bgr = camera.read()
            attempt_stats.append(frame_std(frame_bgr))

    perception = WhiteTapePerception(settings.perception)
    result = perception.process(frame_bgr)
    command = RaycastLineFollower(settings.controller).predict(
        result.raycast,
        result.mask_fraction,
        ray_fov=result.ray_fov,
        track_center_offset=result.track_center_offset,
        track_heading=result.track_heading,
        track_confidence=result.track_confidence,
    )

    raw_path = args.out_dir / "frame_raw.png"
    debug_path = args.out_dir / "frame_debug.png"
    mask_path = args.out_dir / "mask.png"
    cv2.imwrite(str(raw_path), frame_bgr)
    cv2.imwrite(str(debug_path), draw_debug(result, command.throttle, command.steering, command.reason))
    cv2.imwrite(str(mask_path), result.mask.astype(np.uint8) * 255)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    min_channel = frame_bgr.min(axis=2)
    channel_delta = frame_bgr.max(axis=2) - min_channel

    top = int(frame_bgr.shape[0] * settings.perception.roi_top_fraction)
    roi = slice(top, frame_bgr.shape[0])
    roi_value = value[roi, :]
    roi_saturation = saturation[roi, :]
    roi_min_channel = min_channel[roi, :]
    roi_delta = channel_delta[roi, :]

    candidates = {
        "current": settings.perception,
        "relaxed_120": relaxed_settings(settings.perception, 120, 150, 80),
        "relaxed_100": relaxed_settings(settings.perception, 100, 180, 60),
        "very_relaxed_80": relaxed_settings(settings.perception, 80, 220, 40),
    }

    payload = {
        "saved": {
            "raw": str(raw_path),
            "debug": str(debug_path),
            "mask": str(mask_path),
        },
        "camera_source": settings.camera.source,
        "frame_shape": list(frame_bgr.shape),
        "frame_min_bgr": frame_bgr.min(axis=(0, 1)).astype(int).tolist(),
        "frame_max_bgr": frame_bgr.max(axis=(0, 1)).astype(int).tolist(),
        "frame_std_bgr": [float(v) for v in frame_bgr.std(axis=(0, 1))],
        "frame_spatial_std": frame_std(frame_bgr),
        "attempts_used": len(attempt_stats),
        "attempt_std_min": min(attempt_stats),
        "attempt_std_max": max(attempt_stats),
        "warning": (
            "blank_or_startup_frame"
            if frame_std(frame_bgr) < settings.perception.min_frame_std
            else None
        ),
        "current_mask_fraction": result.mask_fraction,
        "current_raycast": [int(v) for v in result.raycast],
        "current_command": {
            "throttle": command.throttle,
            "steering": command.steering,
            "reason": command.reason,
        },
        "roi_stats": {
            "value_p90": percentile(roi_value, 90),
            "value_p95": percentile(roi_value, 95),
            "value_p99": percentile(roi_value, 99),
            "saturation_p50": percentile(roi_saturation, 50),
            "saturation_p90": percentile(roi_saturation, 90),
            "min_rgb_p90": percentile(roi_min_channel, 90),
            "min_rgb_p95": percentile(roi_min_channel, 95),
            "min_rgb_p99": percentile(roi_min_channel, 99),
            "channel_delta_p90": percentile(roi_delta, 90),
            "channel_delta_p99": percentile(roi_delta, 99),
        },
        "candidate_mask_fractions": {
            name: mask_fraction_for(frame_bgr, candidate) for name, candidate in candidates.items()
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
