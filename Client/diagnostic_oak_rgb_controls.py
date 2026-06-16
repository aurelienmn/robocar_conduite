from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


def spatial_std(frame_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).std())


def stats(frame_bgr: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(frame_bgr.shape),
        "min_bgr": frame_bgr.min(axis=(0, 1)).astype(int).tolist(),
        "max_bgr": frame_bgr.max(axis=(0, 1)).astype(int).tolist(),
        "mean_bgr": [float(v) for v in frame_bgr.mean(axis=(0, 1))],
        "spatial_std": spatial_std(frame_bgr),
    }


def board_socket(dai):
    if hasattr(dai.CameraBoardSocket, "CAM_A"):
        return dai.CameraBoardSocket.CAM_A
    return dai.CameraBoardSocket.RGB


def try_control(name: str, fn: Callable[[], None], applied: list[str], errors: list[str]) -> None:
    try:
        fn()
        applied.append(name)
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")


def build_pipeline(dai, variant: str, width: int, height: int, fps: int):
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(board_socket(dai))
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(width, height)
    cam.setFps(fps)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    applied: list[str] = []
    errors: list[str] = []

    if variant == "default":
        pass
    elif variant == "continuous_af":
        try_control(
            "continuous_af",
            lambda: cam.initialControl.setAutoFocusMode(
                dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
            ),
            applied,
            errors,
        )
    elif variant == "manual_focus_mid":
        try_control("manual_focus_130", lambda: cam.initialControl.setManualFocus(130), applied, errors)
    elif variant == "manual_focus_near":
        try_control("manual_focus_220", lambda: cam.initialControl.setManualFocus(220), applied, errors)
    elif variant == "manual_exposure_bright":
        try_control(
            "manual_exposure_10000_800",
            lambda: cam.initialControl.setManualExposure(10000, 800),
            applied,
            errors,
        )
    elif variant == "manual_exposure_dark":
        try_control(
            "manual_exposure_2000_400",
            lambda: cam.initialControl.setManualExposure(2000, 400),
            applied,
            errors,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    return pipeline, applied, errors


def capture_variant(dai, variant: str, args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {"variant": variant}

    try:
        pipeline, applied, errors = build_pipeline(dai, variant, args.width, args.height, args.fps)
        result["applied_controls"] = applied
        result["control_errors"] = errors

        with dai.Device(pipeline) as device:
            queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            frame_bgr = None
            attempt_stds = []
            for _ in range(args.attempts):
                frame_bgr = queue.get().getCvFrame()
                attempt_stds.append(spatial_std(frame_bgr))
                if attempt_stds[-1] >= args.min_std:
                    break

        if frame_bgr is None:
            result["error"] = "no_frame"
            return result

        out_path = args.out_dir / f"{variant}.png"
        cv2.imwrite(str(out_path), frame_bgr)
        result.update(stats(frame_bgr))
        result["saved"] = str(out_path)
        result["attempts_used"] = len(attempt_stds)
        result["attempt_std_min"] = min(attempt_stds)
        result["attempt_std_max"] = max(attempt_stds)
        result["warning"] = "flat_stream" if spatial_std(frame_bgr) < args.min_std else None
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RGB camera controls on OAK-D Lite.")
    parser.add_argument("--out-dir", type=Path, default=Path("debug_oak_controls"))
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

    variants = (
        "default",
        "continuous_af",
        "manual_focus_mid",
        "manual_focus_near",
        "manual_exposure_bright",
        "manual_exposure_dark",
    )

    payload = {
        "depthai_version": dai.__version__,
        "out_dir": str(args.out_dir),
        "variants": [capture_variant(dai, variant, args) for variant in variants],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
