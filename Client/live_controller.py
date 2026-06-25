from dataclasses import dataclass
from typing import Optional

import numpy as np

from live_settings import ControllerSettings
from corner_model import CornerModel


@dataclass(frozen=True)
class DriveCommand:
    throttle: float
    steering: float
    confidence: float
    reason: str


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class RaycastLineFollower:
    """
    Controleur algorithmique : suit l'espace libre entre les bandes blanches.

    Convention rayons :
      ray 0 = droite extreme, ray central = devant, dernier ray = gauche extreme

    Mode hybride :
      Si corner_model_weights.npz est present dans Client/,
      le modele ML prend progressivement le relais dans les virages serres.
      Sinon, l'algo seul est utilise (pas d'erreur).

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
        self.corner_model = CornerModel.load_if_available()

    def predict(self, raycast: np.ndarray, mask_fraction: float, dt_s: Optional[float] = None) -> DriveCommand:
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
        reason = "target_ray={}".format(target_idx)

        # Blend modele ML en virage serre si disponible
        if self.corner_model is not None:
            asymmetry = abs(left_clear - right_clear) / max(left_clear + right_clear, 1.0)
            if asymmetry > 0.45:
                ml_steering = self.corner_model.predict(distances)
                blend = min((asymmetry - 0.45) / 0.55, 1.0)
                steering = (1.0 - blend) * steering + blend * ml_steering
                reason = "ml_corner(blend={:.2f})".format(blend)

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
        )
