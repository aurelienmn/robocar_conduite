import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_controller import RaycastLineFollower
from live_settings import ControllerSettings


class RaycastLineFollowerTest(unittest.TestCase):
    def test_predict_does_not_crash_when_guard_distance_is_reached(self):
        settings = ControllerSettings(
            boundary_guard_distance_px=160,
            turn_anticipation_distance_px=120,
            emergency_distance_px=40,
        )
        controller = RaycastLineFollower(settings)
        distances = np.full(15, 420, dtype=np.float32)
        distances[1] = 130

        command = controller.predict(distances, mask_fraction=0.02, dt_s=0.05, ray_fov=160)

        self.assertTrue(np.isfinite(command.steering))
        self.assertIn(command.reason, {"boundary_guard_right", "target_ray=7", "turn_memory"})

    def test_turn_anticipation_forces_steering_before_emergency(self):
        settings = ControllerSettings(
            base_throttle=0.065,
            turn_anticipation_distance_px=320,
            turn_anticipation_steering=0.95,
            turn_anticipation_throttle_scale=0.35,
            emergency_distance_px=95,
            steering_time_constant_s=0.05,
        )
        controller = RaycastLineFollower(settings)
        distances = np.full(15, 430, dtype=np.float32)
        distances[6:9] = 180
        distances[8:] = 150

        command = controller.predict(distances, mask_fraction=0.02, dt_s=0.05, ray_fov=160)

        self.assertEqual(command.reason, "turn_anticipation")
        self.assertGreater(command.steering, 0.35)
        self.assertLess(command.throttle, settings.base_throttle)

    def test_recovery_throttle_does_not_speed_up_near_boundary(self):
        settings = ControllerSettings(
            base_throttle=0.065,
            recovery_throttle=0.05,
            boundary_avoidance_distance_px=280,
            boundary_guard_distance_px=80,
            emergency_distance_px=40,
            throttle_turn_slowdown=0.85,
        )
        controller = RaycastLineFollower(settings)
        distances = np.full(15, 420, dtype=np.float32)
        distances[8:] = 90

        command = controller.predict(distances, mask_fraction=0.02, dt_s=0.05, ray_fov=160)

        self.assertLessEqual(command.throttle, settings.recovery_throttle)


if __name__ == "__main__":
    unittest.main()
