import math
from dataclasses import dataclass

import cv2
import numpy as np

from live_settings import PerceptionSettings
from mask_generator_bridge import load_cast_rays


cast_rays = load_cast_rays()


@dataclass(frozen=True)
class PerceptionResult:
    frame_bgr: np.ndarray
    mask: np.ndarray
    mask_rejected: np.ndarray
    raycast: np.ndarray
    mask_fraction: float


class WhiteTapePerception:
    """Camera frame -> white-tape mask -> 9 raycasts."""

    def __init__(self, settings: PerceptionSettings) -> None:
        self.settings = settings

    def predict_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Frame BGR attendue en (H,W,3), obtenu {frame_bgr.shape}")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < self.settings.min_frame_std:
            return np.zeros(frame_bgr.shape[:2], dtype=bool)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        min_channel = frame_bgr.min(axis=2)
        channel_delta = frame_bgr.max(axis=2) - min_channel

        mask = (
            (hsv[:, :, 2] >= self.settings.white_value_min)
            & (hsv[:, :, 1] <= self.settings.saturation_max)
            & (min_channel >= self.settings.min_rgb)
            & (channel_delta <= self.settings.max_rgb_delta)
        )

        if self.settings.roi_top_fraction > 0.0:
            top = int(mask.shape[0] * self.settings.roi_top_fraction)
            mask[:top, :] = False

        if self.settings.roi_bottom_fraction > 0.0:
            bottom = int(mask.shape[0] * (1.0 - self.settings.roi_bottom_fraction))
            mask[bottom:, :] = False

        if self.settings.morphology_kernel > 1:
            kernel = np.ones(
                (self.settings.morphology_kernel, self.settings.morphology_kernel),
                dtype=np.uint8,
            )
            mask_u8 = mask.astype(np.uint8) * 255
            if self.settings.open_iterations > 0:
                mask_u8 = cv2.morphologyEx(
                    mask_u8,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=self.settings.open_iterations,
                )
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            if self.settings.dilate_iterations > 0:
                mask_u8 = cv2.dilate(mask_u8, kernel, iterations=self.settings.dilate_iterations)
            mask = mask_u8 > 0

        mask_before_filter = mask.copy()

        if self.settings.min_component_area > 0 and mask.any():
            mask = self._filter_components(
                mask,
                self.settings.min_component_area,
                self.settings.max_component_area,
                self.settings.min_bottom_fraction,
                self.settings.n_closest,
            )

        return mask, mask_before_filter

    @staticmethod
    def _filter_components(mask, min_area, max_area, min_bottom_fraction, n_closest=4):
        """
        Garde uniquement les composantes blanches qui sont :
          1. Assez grandes (>= min_area pixels)
          2. Pas trop grandes (<= max_area pixels) — rejette murs/sols blancs
          3. Proches de la voiture (bord bas >= min_bottom_fraction * hauteur)
          4. Parmi les n_closest plus proches de la camera (bord bas le plus bas)

        Reglages dans live_config.json :
          min_component_area  : augmenter pour ignorer le bruit
          max_component_area  : diminuer si les murs sont encore detectes
          min_bottom_fraction : augmenter si le decor haut est detecte
        """
        h = mask.shape[0]
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        if labels_count <= 1:
            return mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        bottom_y = stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT]

        valid = (
            (areas >= min_area)
            & (areas <= max_area)
            & (bottom_y >= int(h * min_bottom_fraction))
        )

        if not valid.any():
            return np.zeros_like(mask, dtype=bool)

        valid_indices = np.where(valid)[0]
        order = np.argsort(-bottom_y[valid_indices])
        closest = valid_indices[order[:n_closest]]

        keep = np.zeros(labels_count, dtype=bool)
        keep[closest + 1] = True
        return keep[labels]

    def process(self, frame_bgr: np.ndarray) -> PerceptionResult:
        mask, mask_before_filter = self.predict_mask(frame_bgr)
        rejected = mask_before_filter & ~mask
        raycast = cast_rays(mask, n_rays=self.settings.n_rays, fov=self.settings.fov).astype(np.float32)
        return PerceptionResult(
            frame_bgr=frame_bgr,
            mask=mask,
            mask_rejected=rejected,
            raycast=raycast,
            mask_fraction=float(mask.mean()),
        )


def draw_debug(result: PerceptionResult, throttle: float, steering: float, reason: str) -> np.ndarray:
    frame = result.frame_bgr.copy()
    frame[result.mask_rejected] = (255, 80, 0)   # bleu = elimine
    frame[result.mask] = (0, 0, 255)              # rouge = capte

    h, w = result.mask.shape
    ox, oy = w / 2.0, h - 1.0
    n_rays = len(result.raycast)
    angles = np.array([90.0], dtype=np.float32) if n_rays == 1 else np.linspace(0.0, 180.0, n_rays)

    for idx, distance in enumerate(result.raycast):
        rad = math.radians(float(angles[idx]))
        x = int(ox + float(distance) * math.cos(rad))
        y = int(oy - float(distance) * math.sin(rad))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        cv2.line(frame, (int(ox), int(oy)), (x, y), (255, 180, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    cv2.putText(
        frame,
        f"thr={throttle:.3f} steer={steering:.2f} mask={result.mask_fraction:.4f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        reason,
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return frame
