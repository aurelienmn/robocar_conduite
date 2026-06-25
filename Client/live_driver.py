import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from live_controller import DriveCommand, RaycastLineFollower
from live_perception import PerceptionResult, WhiteTapePerception
from live_settings import LiveSettings


@dataclass(frozen=True)
class LiveResult:
    command: DriveCommand
    perception: PerceptionResult
    dt_s: Optional[float]


class LiveDriver:
    """Camera frame -> 9 raycasts -> throttle/steering command."""

    def __init__(self, settings: LiveSettings) -> None:
        self.perception = WhiteTapePerception(settings.perception)
        self.controller = RaycastLineFollower(settings.controller)
        self.previous_prediction_time: Optional[float] = None

    def predict_bgr(self, frame_bgr: np.ndarray) -> LiveResult:
        now = time.monotonic()
        dt_s = None if self.previous_prediction_time is None else now - self.previous_prediction_time
        self.previous_prediction_time = now

        perception = self.perception.process(frame_bgr)
        command = self.controller.predict(perception.raycast, perception.mask_fraction, dt_s=dt_s)
        return LiveResult(command=command, perception=perception, dt_s=dt_s)
