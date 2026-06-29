from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from live_settings import ControllerSettings


@dataclass(frozen=True)
class DriveCommand:
    throttle: float
    steering: float
    confidence: float
    reason: str
    diagnostics: Dict[str, Any]


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class RaycastLineFollower:
    """
    Controleur algorithmique : suit l'espace libre entre les bandes blanches.

    Convention rayons :
      ray 0 = droite extreme, ray central = devant, dernier ray = gauche extreme

    Parametres cles dans live_config.json (section controller) :
      base_throttle        : vitesse normale (augmenter si trop lent)
      max_throttle         : vitesse max absolue
      steering_gain        : amplification du braquage (augmenter si vire trop peu)
      avoid_gain           : centrage entre les deux bandes
      steering_smoothing   : lissage du volant (0=brusque, 1=tres lisse)
      throttle_turn_slowdown : freinage en virage (1=arret complet en virage)
      emergency_distance_px  : distance en pixels avant evitement urgence
    """

    def __init__(self, settings: ControllerSettings) -> None:
        self.settings = settings
        self.previous_steering = 0.0

    def predict(
        self,
        raycast: np.ndarray,
        mask_fraction: float,
        dt_s: Optional[float] = None,
        ray_fov: float = 180.0,
        track_center_offset: float = 0.0,
        track_heading: float = 0.0,
        track_confidence: float = 0.0,
    ) -> DriveCommand:
        distances = np.asarray(raycast, dtype=np.float32).reshape(-1)
        if distances.size == 0 or not np.isfinite(distances).all():
            return self._lost("invalid_raycast")

        if dt_s is not None and dt_s > self.settings.stale_frame_timeout_s:
            return self._lost("stale_frame")

        if mask_fraction < self.settings.lost_mask_fraction:
            return self._lost("line_lost")

        n_rays = len(distances)
        middle = n_rays // 2
        if n_rays == 1:
            angles = np.array([90.0], dtype=np.float32)
        else:
            ray_fov = clamp(float(ray_fov), 1.0, 180.0)
            angles = np.arange(n_rays, dtype=np.float32) * (ray_fov / (n_rays - 1)) + (180.0 - ray_fov) / 2.0

        max_distance = max(float(distances.max()), 1.0)
        normalized = np.clip(distances / max_distance, 0.0, 1.0)
        forward_bias = 1.0 - 0.10 * np.abs(angles - 90.0) / 90.0
        scores = normalized * forward_bias

        target_idx = int(np.argmax(scores))
        target_angle = float(angles[target_idx])
        ray_steering = (90.0 - target_angle) / 90.0 * self.settings.steering_gain
        ray_steering_before_balance = ray_steering

        right_clear = float(distances[:middle].mean()) if middle > 0 else max_distance
        left_clear = float(distances[middle + 1 :].mean()) if middle + 1 < n_rays else max_distance
        right_min = float(distances[:middle].min()) if middle > 0 else max_distance
        left_min = float(distances[middle + 1 :].min()) if middle + 1 < n_rays else max_distance
        balance = (left_clear - right_clear) / max(left_clear + right_clear, 1.0)
        ray_steering -= self.settings.avoid_gain * balance

        center_window = distances[max(0, middle - 1) : min(n_rays, middle + 2)]
        front_min = float(center_window.min())
        reason = "target_ray={}".format(target_idx)

        steering = ray_steering
        fast_response = False
        track_confidence = clamp(float(track_confidence), 0.0, 1.0)
        center_offset = clamp(float(track_center_offset), -1.0, 1.0)
        heading = clamp(float(track_heading), -1.0, 1.0)
        centerline_steering = None
        centerline_blend = 0.0
        if track_confidence > 0.20:
            centerline_steering = (
                self.settings.centerline_gain * center_offset
                + self.settings.heading_gain * heading
            )
            centerline_blend = clamp(self.settings.centerline_blend * track_confidence, 0.0, 1.0)
            steering = (1.0 - centerline_blend) * ray_steering + centerline_blend * centerline_steering
            reason = "centerline(blend={:.2f},ray={})".format(centerline_blend, target_idx)

        guard_distance = max(float(self.settings.boundary_guard_distance_px), 1.0)
        guard_steering = clamp(float(self.settings.boundary_guard_steering), 0.0, 1.0)
        closest_side = min(left_min, right_min)
        guard_side = None
        guard_forced_steering = None
        if closest_side < guard_distance:
            proximity = clamp((guard_distance - closest_side) / guard_distance, 0.0, 1.0)
            forced = guard_steering * (0.55 + 0.45 * proximity)
            if right_min < left_min * 0.92:
                steering = min(steering, -forced)
                reason = "boundary_guard_right"
                fast_response = True
                guard_side = "right"
                guard_forced_steering = -forced
            elif left_min < right_min * 0.92:
                steering = max(steering, forced)
                reason = "boundary_guard_left"
                fast_response = True
                guard_side = "left"
                guard_forced_steering = forced

        emergency_side = None
        if front_min < self.settings.emergency_distance_px:
            steering = -0.85 if left_clear > right_clear else 0.85
            reason = "emergency_avoid"
            fast_response = True
            emergency_side = "right" if left_clear > right_clear else "left"

        smoothing = self._steering_alpha(dt_s)
        if fast_response:
            smoothing = max(smoothing, 0.70)
        pre_smooth_steering = steering
        steering = self.previous_steering + smoothing * (steering - self.previous_steering)
        steering = clamp(steering, -1.0, 1.0)
        self.previous_steering = steering

        curve_intensity = max(
            abs(steering),
            track_confidence * (
                0.45 * abs(float(track_center_offset))
                + 0.75 * abs(float(track_heading))
            ),
        )
        turn_factor = 1.0 - self.settings.throttle_turn_slowdown * clamp(curve_intensity, 0.0, 1.0)
        throttle = self.settings.base_throttle * clamp(turn_factor, 0.0, 1.0)
        throttle_before_slow_frame = throttle
        if dt_s is not None and dt_s > self.settings.slow_frame_threshold_s:
            throttle *= clamp(self.settings.slow_frame_threshold_s / dt_s, 0.25, 1.0)
        if front_min < self.settings.emergency_distance_px:
            throttle = min(throttle, self.settings.min_throttle)
        throttle = clamp(throttle, self.settings.min_throttle, self.settings.max_throttle)

        confidence = clamp(mask_fraction / max(self.settings.lost_mask_fraction * 8.0, 1e-6), 0.0, 1.0)
        diagnostics = {
            "n_rays": int(n_rays),
            "ray_fov": float(ray_fov),
            "target_idx": int(target_idx),
            "target_angle": float(target_angle),
            "target_distance": float(distances[target_idx]),
            "ray_steering": float(ray_steering),
            "ray_steering_before_balance": float(ray_steering_before_balance),
            "balance": float(balance),
            "left_clear": float(left_clear),
            "right_clear": float(right_clear),
            "left_min": float(left_min),
            "right_min": float(right_min),
            "front_min": float(front_min),
            "track_center": float(center_offset),
            "track_heading": float(heading),
            "track_confidence": float(track_confidence),
            "centerline_steering": None if centerline_steering is None else float(centerline_steering),
            "centerline_blend": float(centerline_blend),
            "guard_side": guard_side,
            "guard_distance": float(guard_distance),
            "guard_forced_steering": (
                None if guard_forced_steering is None else float(guard_forced_steering)
            ),
            "emergency_side": emergency_side,
            "fast_response": bool(fast_response),
            "smoothing": float(smoothing),
            "pre_smooth_steering": float(pre_smooth_steering),
            "curve_intensity": float(curve_intensity),
            "turn_factor": float(turn_factor),
            "throttle_before_slow_frame": float(throttle_before_slow_frame),
        }
        return DriveCommand(
            throttle=throttle,
            steering=steering,
            confidence=confidence,
            reason=reason,
            diagnostics=diagnostics,
        )

    def _steering_alpha(self, dt_s: Optional[float]) -> float:
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
            diagnostics={"lost_reason": reason},
        )
