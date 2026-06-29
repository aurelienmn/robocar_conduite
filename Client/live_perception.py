import math
from dataclasses import dataclass
from typing import Tuple

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
    ray_fov: float
    mask_fraction: float
    track_center_offset: float
    track_heading: float
    track_confidence: float
    track_width_px: float
    track_points: Tuple[Tuple[int, int], ...]


class WhiteTapePerception:
    """Camera frame -> white-tape mask -> configured raycasts."""

    def __init__(self, settings: PerceptionSettings) -> None:
        self.settings = settings

    def predict_mask(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Frame BGR attendue en (H,W,3), obtenu {frame_bgr.shape}")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < self.settings.min_frame_std:
            empty = np.zeros(frame_bgr.shape[:2], dtype=bool)
            return empty, empty.copy()

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

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
            )

        return mask, mask_before_filter

    @staticmethod
    def _estimate_track_center(mask):
        h, w = mask.shape
        if not mask.any():
            return 0.0, 0.0, 0.0, 0.0, tuple()

        y_near = int(h * 0.88)
        y_far = int(h * 0.42)
        if y_far >= y_near:
            return 0.0, 0.0, 0.0, 0.0, tuple()

        sample_ys = np.linspace(y_near, y_far, 9).astype(int)
        center_x = w / 2.0
        nominal_width = w * 0.58
        min_width = w * 0.18
        max_width = w * 0.92
        previous_center = center_x
        widths = []

        def contiguous_runs(cols):
            if cols.size == 0:
                return []
            split_at = np.flatnonzero(np.diff(cols) > 1) + 1
            groups = np.split(cols, split_at)
            return [
                (float(group[0]), float(group[-1]))
                for group in groups
                if group.size >= 3
            ]

        centers = []
        weighted_sum = 0.0
        weight_total = 0.0
        quality_total = 0.0

        for idx, y in enumerate(sample_ys):
            band = mask[max(0, y - 2): min(h, y + 3), :]
            cols = np.flatnonzero(band.any(axis=0))
            runs = contiguous_runs(cols)
            if not runs:
                continue

            if widths:
                nominal_width = float(np.median(widths[-5:]))

            candidates = []
            for left_idx, left_run in enumerate(runs[:-1]):
                for right_run in runs[left_idx + 1:]:
                    width = right_run[0] - left_run[1]
                    if min_width <= width <= max_width:
                        center = (left_run[1] + right_run[0]) * 0.5
                        score = abs(center - previous_center) + 0.25 * abs(width - nominal_width)
                        candidates.append((score, center, width, 1.0))

            for start_x, end_x in runs:
                left_edge_center = end_x + nominal_width * 0.5
                if 0.0 <= left_edge_center < w:
                    score = abs(left_edge_center - previous_center) + 0.45 * nominal_width
                    candidates.append((score, left_edge_center, nominal_width, 0.45))

                right_edge_center = start_x - nominal_width * 0.5
                if 0.0 <= right_edge_center < w:
                    score = abs(right_edge_center - previous_center) + 0.45 * nominal_width
                    candidates.append((score, right_edge_center, nominal_width, 0.45))

            if not candidates:
                continue

            _, center, width, quality = min(candidates, key=lambda c: c[0])
            if centers and quality < 1.0 and abs(center - previous_center) > w * 0.32:
                continue

            smooth = 0.72 if quality >= 1.0 else 0.45
            center = smooth * center + (1.0 - smooth) * previous_center
            center = float(np.clip(center, 0.0, w - 1.0))
            previous_center = center
            if quality >= 1.0:
                widths.append(float(width))

            progress = idx / max(len(sample_ys) - 1, 1)
            weight = quality * (1.0 + 0.8 * progress)
            weighted_sum += center * weight
            weight_total += weight
            quality_total += quality
            centers.append((center, float(y), quality))

        if len(centers) < 2 or weight_total <= 0.0:
            return 0.0, 0.0, 0.0, nominal_width, tuple()

        target_center = weighted_sum / weight_total
        near = np.median([c[0] for c in centers[: min(3, len(centers))]])
        far = np.median([c[0] for c in centers[max(0, len(centers) - 3):]])

        offset = (target_center - center_x) / max(center_x, 1.0)
        heading = (far - near) / max(center_x, 1.0)
        confidence = min(1.0, quality_total / 6.0)
        points = tuple((int(round(c[0])), int(round(c[1]))) for c in centers)
        return (
            float(np.clip(offset, -1.0, 1.0)),
            float(np.clip(heading, -1.0, 1.0)),
            float(confidence),
            float(nominal_width),
            points,
        )

    @staticmethod
    def _filter_components(mask, min_area, max_area, min_bottom_fraction, n_closest=6):
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
        center_offset, heading, confidence, width_px, points = self._estimate_track_center(mask)
        return PerceptionResult(
            frame_bgr=frame_bgr,
            mask=mask,
            mask_rejected=rejected,
            raycast=raycast,
            ray_fov=float(self.settings.fov),
            mask_fraction=float(mask.mean()),
            track_center_offset=center_offset,
            track_heading=heading,
            track_confidence=confidence,
            track_width_px=width_px,
            track_points=points,
        )


def draw_debug(result: PerceptionResult, throttle: float, steering: float, reason: str) -> np.ndarray:
    frame = result.frame_bgr.copy()
    frame[result.mask_rejected] = (255, 80, 0)   # bleu = elimine
    frame[result.mask] = (0, 0, 255)              # rouge = capte

    h, w = result.mask.shape
    ox, oy = w / 2.0, h - 1.0
    n_rays = len(result.raycast)
    if n_rays == 1:
        angles = np.array([90.0], dtype=np.float32)
    else:
        fov = max(1.0, min(180.0, float(result.ray_fov)))
        angles = np.arange(n_rays, dtype=np.float32) * (fov / (n_rays - 1)) + (180.0 - fov) / 2.0

    for idx, distance in enumerate(result.raycast):
        rad = math.radians(float(angles[idx]))
        x = int(ox + float(distance) * math.cos(rad))
        y = int(oy - float(distance) * math.sin(rad))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        cv2.line(frame, (int(ox), int(oy)), (x, y), (255, 180, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    if result.track_points:
        pts = np.array(result.track_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
        for point in result.track_points:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)
        if len(result.track_points) >= 2:
            cv2.arrowedLine(
                frame,
                result.track_points[0],
                result.track_points[-1],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
                tipLength=0.18,
            )

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
    cv2.putText(
        frame,
        "center={:.2f} head={:.2f} conf={:.2f}".format(
            result.track_center_offset,
            result.track_heading,
            result.track_confidence,
        ),
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return frame
