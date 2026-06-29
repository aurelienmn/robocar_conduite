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
        self.emergency_hold_remaining_s = 0.0
        self.emergency_steering = 0.0

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
        dt_step = 0.10 if dt_s is None else max(float(dt_s), 0.0)
        self.emergency_hold_remaining_s = max(0.0, self.emergency_hold_remaining_s - dt_step)
        if n_rays == 1:
            angles = np.array([90.0], dtype=np.float32)
        else:
            ray_fov = clamp(float(ray_fov), 1.0, 180.0)
            angles = np.arange(n_rays, dtype=np.float32) * (ray_fov / (n_rays - 1)) + (180.0 - ray_fov) / 2.0

        max_distance = max(float(distances.max()), 1.0)
        normalized = np.clip(distances / max_distance, 0.0, 1.0)
        forward_bias = 1.0 - 0.10 * np.abs(angles - 90.0) / 90.0
        scores = normalized * forward_bias

        navigation_margin = int(max(0, self.settings.navigation_margin_rays))
        if n_rays > navigation_margin * 2 + 1:
            nav_scores = scores.copy()
            nav_scores[:navigation_margin] = -1.0
            nav_scores[n_rays - navigation_margin:] = -1.0
            target_idx = int(np.argmax(nav_scores))
        else:
            target_idx = int(np.argmax(scores))
        target_angle = float(angles[target_idx])
        ray_steering = (90.0 - target_angle) / 90.0 * self.settings.steering_gain
        ray_steering_before_balance = ray_steering

        right_clear = float(distances[:middle].mean()) if middle > 0 else max_distance
        left_clear = float(distances[middle + 1 :].mean()) if middle + 1 < n_rays else max_distance
        right_min = float(distances[:middle].min()) if middle > 0 else max_distance
        left_min = float(distances[middle + 1 :].min()) if middle + 1 < n_rays else max_distance
        right_emergency_rays = distances[1:middle] if middle > 2 else distances[:middle]
        left_emergency_rays = distances[middle + 1 : -1] if middle + 2 < n_rays else distances[middle + 1 :]
        right_emergency_clear = (
            float(np.median(right_emergency_rays)) if right_emergency_rays.size else right_clear
        )
        left_emergency_clear = (
            float(np.median(left_emergency_rays)) if left_emergency_rays.size else left_clear
        )
        balance = (left_clear - right_clear) / max(left_clear + right_clear, 1.0)
        ray_steering -= self.settings.avoid_gain * balance

        center_window = distances[max(0, middle - 1) : min(n_rays, middle + 2)]
        front_min = float(center_window.min())
        reason = "target_ray={}".format(target_idx)
        boundary_distance = max(float(self.settings.boundary_avoidance_distance_px), 1.0)
        boundary_closeness = np.clip((boundary_distance - distances) / boundary_distance, 0.0, 1.0)
        if n_rays > navigation_margin * 2 + 1:
            boundary_closeness[:navigation_margin] *= 0.45
            boundary_closeness[n_rays - navigation_margin:] *= 0.45
        side_sign = np.sign(angles - 90.0)
        forward_weight = 1.0 - 0.45 * np.abs(angles - 90.0) / 90.0
        pressure = boundary_closeness * forward_weight
        left_mask = side_sign > 0.0
        right_mask = side_sign < 0.0
        if left_mask.any():
            left_avg = pressure[left_mask].sum() / max(float(forward_weight[left_mask].sum()), 1e-6)
            left_pressure = float(0.65 * pressure[left_mask].max() + 0.35 * left_avg)
        else:
            left_pressure = 0.0
        if right_mask.any():
            right_avg = pressure[right_mask].sum() / max(float(forward_weight[right_mask].sum()), 1e-6)
            right_pressure = float(0.65 * pressure[right_mask].max() + 0.35 * right_avg)
        else:
            right_pressure = 0.0
        boundary_avoidance_max = max(float(self.settings.boundary_avoidance_max_steering), 0.0)
        boundary_avoidance = clamp(
            (left_pressure - right_pressure) * float(self.settings.boundary_avoidance_gain),
            -boundary_avoidance_max,
            boundary_avoidance_max,
        )

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

        if abs(boundary_avoidance) > 0.02:
            steering = clamp(steering + boundary_avoidance, -1.0, 1.0)
            if abs(boundary_avoidance) > 0.12:
                reason = "boundary_avoidance"
            if abs(boundary_avoidance) > 0.20:
                fast_response = True

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

        anticipation_distance = max(float(self.settings.turn_anticipation_distance_px), 1.0)
        anticipation_steering = clamp(float(self.settings.turn_anticipation_steering), 0.0, 1.0)
        anticipation_active = front_min < anticipation_distance
        if anticipation_active:
            fast_response = True
            reason = "turn_anticipation"
            if abs(steering) < anticipation_steering:
                if abs(steering) > 0.05:
                    direction = 1.0 if steering > 0.0 else -1.0
                elif track_confidence > 0.20 and abs(heading) > 0.05:
                    direction = 1.0 if heading > 0.0 else -1.0
                elif left_min < right_min:
                    direction = 1.0
                elif right_min < left_min:
                    direction = -1.0
                elif abs(self.previous_steering) > 0.05:
                    direction = 1.0 if self.previous_steering > 0.0 else -1.0
                else:
                    direction = 0.0
                if direction != 0.0:
                    steering = direction * anticipation_steering

        emergency_side = None
        if front_min < self.settings.emergency_distance_px:
            switch_ratio = max(float(self.settings.emergency_switch_ratio), 1.0)
            if left_emergency_clear > right_emergency_clear * switch_ratio:
                steering = -0.85
                emergency_side = "right"
            elif right_emergency_clear > left_emergency_clear * switch_ratio:
                steering = 0.85
                emergency_side = "left"
            elif self.emergency_hold_remaining_s > 0.0 and self.emergency_steering != 0.0:
                steering = self.emergency_steering
                emergency_side = "hold"
            elif abs(self.previous_steering) > 0.12:
                steering = 0.85 if self.previous_steering > 0.0 else -0.85
                emergency_side = "previous"
            elif left_emergency_clear > right_emergency_clear:
                steering = -0.85
                emergency_side = "right"
            else:
                steering = 0.85
                emergency_side = "left"
            self.emergency_steering = steering
            self.emergency_hold_remaining_s = max(float(self.settings.emergency_hold_s), 0.0)
            reason = "emergency_avoid"
            fast_response = True

        smoothing = self._steering_alpha(dt_s)
        if fast_response:
            smoothing = max(smoothing, 0.90)
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
        if anticipation_active:
            throttle = min(
                throttle,
                self.settings.base_throttle
                * clamp(float(self.settings.turn_anticipation_throttle_scale), 0.0, 1.0),
            )
        if front_min < self.settings.emergency_distance_px:
            throttle = min(throttle, self.settings.min_throttle)
        throttle = clamp(throttle, self.settings.min_throttle, self.settings.max_throttle)

        confidence = clamp(mask_fraction / max(self.settings.lost_mask_fraction * 8.0, 1e-6), 0.0, 1.0)
        diagnostics = {
            "n_rays": int(n_rays),
            "ray_fov": float(ray_fov),
            "navigation_margin_rays": int(navigation_margin),
            "target_idx": int(target_idx),
            "target_angle": float(target_angle),
            "target_distance": float(distances[target_idx]),
            "ray_steering": float(ray_steering),
            "ray_steering_before_balance": float(ray_steering_before_balance),
            "balance": float(balance),
            "left_clear": float(left_clear),
            "right_clear": float(right_clear),
            "left_emergency_clear": float(left_emergency_clear),
            "right_emergency_clear": float(right_emergency_clear),
            "left_min": float(left_min),
            "right_min": float(right_min),
            "front_min": float(front_min),
            "boundary_avoidance": float(boundary_avoidance),
            "left_pressure": float(left_pressure),
            "right_pressure": float(right_pressure),
            "boundary_avoidance_distance": float(boundary_distance),
            "anticipation_active": bool(anticipation_active),
            "anticipation_distance": float(anticipation_distance),
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
            "emergency_hold_remaining_s": float(self.emergency_hold_remaining_s),
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
