import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "Client" / "live_config.json"


@dataclass(frozen=True)
class CameraSettings:
    source: str = "rgb_preview"
    width: int = 640
    height: int = 480
    fps: int = 6
    warmup_frames: int = 6
    exposure_compensation: int = 0


@dataclass(frozen=True)
class PerceptionSettings:
    n_rays: int = 12
    fov: float = 180.0
    min_frame_std: float = 3.0
    white_value_min: int = 150
    saturation_max: int = 95
    min_rgb: int = 115
    max_rgb_delta: int = 95
    roi_top_fraction: float = 0.30
    roi_bottom_fraction: float = 0.06
    morphology_kernel: int = 3
    open_iterations: int = 1
    dilate_iterations: int = 0
    min_component_area: int = 350
    max_component_area: int = 20000
    min_bottom_fraction: float = 0.45


@dataclass(frozen=True)
class ControllerSettings:
    base_throttle: float = 0.028
    min_throttle: float = 0.0
    max_throttle: float = 0.045
    steering_gain: float = 1.10
    avoid_gain: float = 0.45
    centerline_gain: float = 0.85
    heading_gain: float = 0.55
    centerline_blend: float = 0.75
    steering_smoothing: float = 0.35
    steering_time_constant_s: float = 0.25
    throttle_turn_slowdown: float = 0.65
    slow_frame_threshold_s: float = 0.45
    stale_frame_timeout_s: float = 1.20
    emergency_distance_px: float = 45.0
    lost_mask_fraction: float = 0.0008
    lost_line_throttle: float = 0.0


@dataclass(frozen=True)
class VescSettings:
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    max_duty: float = 0.08
    deadzone: float = 0.03
    servo_center: float = 0.50
    servo_range: float = 0.30
    servo_min: float = 0.20
    servo_max: float = 0.80


@dataclass(frozen=True)
class LiveSettings:
    camera: CameraSettings
    perception: PerceptionSettings
    controller: ControllerSettings
    vesc: VescSettings


def _section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    section = data.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be an object")
    return section


def load_settings(path: Union[str, Path] = DEFAULT_CONFIG) -> LiveSettings:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    return LiveSettings(
        camera=CameraSettings(**_section(data, "camera")),
        perception=PerceptionSettings(**_section(data, "perception")),
        controller=ControllerSettings(**_section(data, "controller")),
        vesc=VescSettings(**_section(data, "vesc")),
    )
