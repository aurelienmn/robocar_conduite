from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from live_settings import ControllerSettings


@dataclass(frozen=True)
class DriveCommand:
    throttle: float
    steering: float
    confidence: float
    reason: str


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class RaycastLineFollower:
    """Premier modele live: suivre l'espace libre entre les bandes blanches.

    Convention:
    - steering -1 = gauche, +1 = droite
    - ray 0 = droite, ray central = devant, dernier ray = gauche
    """

    def __init__(self, settings: ControllerSettings) -> None:
        self.settings = settings
        self.previous_steering = 0.0

    def predict(self, raycast: np.ndarray, mask_fraction: float, dt_s: float | None = None) -> DriveCommand:
        distances = np.asarray(raycast, dtype=np.float32).reshape(-1)
        if distances.size == 0 or not np.isfinite(distances).all():
            return self._lost("invalid_raycast")

        if dt_s is not None and dt_s > self.settings.stale_frame_timeout_s:
            return self._lost("stale_frame")

        if mask_fraction < self.settings.lost_mask_fraction:
            return self._lost("line_lost")

        n_rays = len(distances)
        middle = n_rays // 2
        angles = np.linspace(0.0, 180.0, n_rays, dtype=np.float32)

        max_distance = max(float(distances.max()), 1.0)
        normalized = np.clip(distances / max_distance, 0.0, 1.0)
        forward_bias = 1.0 - 0.25 * np.abs(angles - 90.0) / 90.0
        scores = normalized * forward_bias

        target_idx = int(np.argmax(scores))
        target_angle = float(angles[target_idx])
        steering = (90.0 - target_angle) / 90.0 * self.settings.steering_gain

        right_clear = float(distances[:middle].mean()) if middle > 0 else max_distance
        left_clear = float(distances[middle + 1 :].mean()) if middle + 1 < n_rays else max_distance
        balance = (left_clear - right_clear) / max(left_clear + right_clear, 1.0)
        steering -= self.settings.avoid_gain * balance

        center_window = distances[max(0, middle - 1) : min(n_rays, middle + 2)]
        front_min = float(center_window.min())
        reason = f"target_ray={target_idx}"

        if front_min < self.settings.emergency_distance_px:
            steering = -0.85 if left_clear > right_clear else 0.85
            reason = "emergency_avoid"

        smoothing = self._steering_alpha(dt_s)
        steering = self.previous_steering + smoothing * (steering - self.previous_steering)
        steering = clamp(steering, -1.0, 1.0)
        self.previous_steering = steering

        turn_factor = 1.0 - self.settings.throttle_turn_slowdown * abs(steering)
        throttle = self.settings.base_throttle * clamp(turn_factor, 0.0, 1.0)
        if dt_s is not None and dt_s > self.settings.slow_frame_threshold_s:
            throttle *= clamp(self.settings.slow_frame_threshold_s / dt_s, 0.25, 1.0)
        if front_min < self.settings.emergency_distance_px:
            throttle = min(throttle, self.settings.min_throttle)
        throttle = clamp(throttle, self.settings.min_throttle, self.settings.max_throttle)

        confidence = clamp(mask_fraction / max(self.settings.lost_mask_fraction * 8.0, 1e-6), 0.0, 1.0)
        return DriveCommand(throttle=throttle, steering=steering, confidence=confidence, reason=reason)

    def _steering_alpha(self, dt_s: float | None) -> float:
        if dt_s is None or self.settings.steering_time_constant_s <= 0.0:
            return clamp(self.settings.steering_smoothing, 0.0, 1.0)
        return clamp(dt_s / (self.settings.steering_time_constant_s + dt_s), 0.0, 1.0)

    def _lost(self, reason: str) -> DriveCommand:
        steering = self.previous_steering * 0.8
        self.previous_steering = steering
        return DriveCommand(
            throttle=self.settings.lost_line_throttle,
            steering=steering,
            confidence=0.0,
            reason=reason,
        )
