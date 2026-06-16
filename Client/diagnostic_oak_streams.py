from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def frame_stats(frame: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return {
        "shape": list(frame.shape),
        "min": frame.min(axis=(0, 1)).astype(int).tolist() if frame.ndim == 3 else int(frame.min()),
        "max": frame.max(axis=(0, 1)).astype(int).tolist() if frame.ndim == 3 else int(frame.max()),
        "mean": [float(v) for v in frame.mean(axis=(0, 1))] if frame.ndim == 3 else float(frame.mean()),
        "channel_std": [float(v) for v in frame.std(axis=(0, 1))] if frame.ndim == 3 else [float(frame.std())],
        "global_std": float(frame.std()),
        "spatial_std": float(gray.std()),
    }


def camera_socket(dai, modern_name: str, legacy_name: str):
    sockets = dai.CameraBoardSocket
    if hasattr(sockets, modern_name):
        return getattr(sockets, modern_name)
    return getattr(sockets, legacy_name)


def build_pipeline(dai, source: str, width: int, height: int, fps: int):
    pipeline = dai.Pipeline()

    if source in ("mono_left", "mono_right"):
        cam = pipeline.create(dai.node.MonoCamera)
        if source == "mono_left":
            cam.setBoardSocket(camera_socket(dai, "CAM_B", "LEFT"))
        else:
            cam.setBoardSocket(camera_socket(dai, "CAM_C", "RIGHT"))
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam.setFps(fps)
        output = cam.out
    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(camera_socket(dai, "CAM_A", "RGB"))
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(fps)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        if source == "rgb_preview":
            cam.setPreviewSize(width, height)
            output = cam.preview
        elif source == "rgb_video":
            cam.setVideoSize(width, height)
            output = cam.video
        elif source == "rgb_isp":
            output = cam.isp
        else:
            raise ValueError(f"Unknown source: {source}")

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(source)
    output.link(xout.input)
    return pipeline


def capture_source(dai, source: str, args: argparse.Namespace) -> dict[str, Any]:
    pipeline = build_pipeline(dai, source, args.width, args.height, args.fps)
    stats: dict[str, Any] = {"source": source}

    try:
        with dai.Device(pipeline) as device:
            queue = device.getOutputQueue(name=source, maxSize=1, blocking=False)
            frame = None
            attempt_stds = []

            for _ in range(args.attempts):
                packet = queue.get()
                frame = packet.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                attempt_stds.append(float(gray.std()))
                if gray.std() >= args.min_std:
                    break

            if frame is None:
                stats["error"] = "no_frame"
                return stats

            out_path = args.out_dir / f"{source}.png"
            cv2.imwrite(str(out_path), frame)
            stats.update(frame_stats(frame))
            stats["saved"] = str(out_path)
            stats["attempts_used"] = len(attempt_stds)
            stats["attempt_std_min"] = min(attempt_stds)
            stats["attempt_std_max"] = max(attempt_stds)
            stats["warning"] = "flat_stream" if stats["spatial_std"] < args.min_std else None
            return stats
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose OAK-D RGB streams.")
    parser.add_argument("--out-dir", type=Path, default=Path("debug_oak_streams"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--min-std", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    import depthai as dai

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "depthai_version": dai.__version__,
        "out_dir": str(args.out_dir),
        "streams": [],
    }

    try:
        payload["available_devices"] = [device.name for device in dai.Device.getAllAvailableDevices()]
    except Exception as exc:
        payload["available_devices_error"] = f"{type(exc).__name__}: {exc}"

    for source in ("rgb_preview", "rgb_video", "rgb_isp", "mono_left", "mono_right"):
        payload["streams"].append(capture_source(dai, source, args))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
