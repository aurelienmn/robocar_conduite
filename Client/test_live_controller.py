import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_controller import RaycastLineFollower
from live_settings import ControllerSettings


class RaycastLineFollowerTest(unittest.TestCase):
    def test_boundary_guard_does_not_reverse_confirmed_right_turn(self):
        settings = ControllerSettings(
            boundary_guard_distance_px=120,
            boundary_guard_reverse_distance_px=45,
            emergency_distance_px=35,
            centerline_blend=0.0,
            steering_time_constant_s=0.01,
        )
        controller = RaycastLineFollower(settings)
        distances = np.array(
            [70, 180, 420, 450, 410, 360, 300, 260, 220, 200, 190, 180],
            dtype=np.float32,
        )

        command = controller.predict(distances, mask_fraction=0.02, dt_s=0.05, ray_fov=180)

        self.assertGreater(command.steering, 0.25)
        self.assertNotEqual(command.reason, "boundary_guard_right")
        self.assertEqual(command.diagnostics["guard_candidate_side"], "right")
        self.assertEqual(command.diagnostics["guard_suppressed_side"], "right")

    def test_boundary_guard_still_applies_when_it_matches_planned_turn(self):
        settings = ControllerSettings(
            boundary_guard_distance_px=120,
            boundary_guard_reverse_distance_px=45,
            emergency_distance_px=35,
            centerline_blend=0.0,
            steering_time_constant_s=0.01,
        )
        controller = RaycastLineFollower(settings)
        distances = np.array(
            [55, 90, 120, 130, 145, 160, 260, 420, 450, 430, 410, 390],
            dtype=np.float32,
        )

        command = controller.predict(distances, mask_fraction=0.02, dt_s=0.05, ray_fov=180)

        self.assertEqual(command.reason, "boundary_guard_right")
        self.assertLess(command.steering, -0.25)
        self.assertEqual(command.diagnostics["guard_side"], "right")
        self.assertIsNone(command.diagnostics["guard_suppressed_side"])

    def test_boundary_guard_can_reverse_when_side_is_critical(self):
        settings = ControllerSettings(
            boundary_guard_distance_px=120,
            boundary_guard_reverse_distance_px=45,
            emergency_distance_px=35,
            centerline_blend=0.0,
            steering_time_constant_s=0.01,
        )
        controller = RaycastLineFollower(settings)
        distances = np.array(
            [35, 180, 420, 450, 410, 360, 300, 260, 220, 200, 190, 180],
            dtype=np.float32,
        )

        command = controller.predict(distances, mask_fraction=0.02, dt_s=0.05, ray_fov=180)

        self.assertEqual(command.reason, "boundary_guard_right")
        self.assertLess(command.steering, -0.25)
        self.assertEqual(command.diagnostics["guard_side"], "right")


if __name__ == "__main__":
    unittest.main()
