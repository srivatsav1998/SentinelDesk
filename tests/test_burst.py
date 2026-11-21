"""Tests for the live outage / burst detector."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np

    from burst import BurstDetector
except ImportError as error:  # numpy not installed (bare CI)
    raise unittest.SkipTest(f"burst dependencies unavailable: {error}")


def vec(*values):
    v = np.zeros(8, dtype=np.float32)
    v[: len(values)] = values
    return v


class BurstDetection(unittest.TestCase):
    def test_three_similar_tickets_raise_alert(self):
        detector = BurstDetector(similarity=0.9, min_cluster=3)
        self.assertIsNone(detector.add("a", "VPN down", vec(1, 0.1), now=0))
        self.assertIsNone(detector.add("b", "VPN fails", vec(1, 0.2), now=10))
        alert = detector.add("c", "VPN offline", vec(1, 0.15), now=20)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.count, 3)

    def test_unrelated_tickets_do_not_alert(self):
        detector = BurstDetector(similarity=0.9, min_cluster=3)
        detector.add("a", "VPN down", vec(1, 0), now=0)
        detector.add("b", "Invoice", vec(0, 1), now=1)
        self.assertIsNone(detector.add("c", "Password", vec(0, 0, 1), now=2))

    def test_resubmitting_the_same_ticket_does_not_count_twice(self):
        detector = BurstDetector(similarity=0.9, min_cluster=3)
        detector.add("a", "VPN down", vec(1, 0.1), now=0)
        detector.add("b", "VPN fails", vec(1, 0.2), now=1)
        self.assertIsNone(detector.add("b", "VPN fails", vec(1, 0.2), now=2))

    def test_old_tickets_leave_the_window(self):
        detector = BurstDetector(window_s=60, similarity=0.9, min_cluster=3)
        detector.add("a", "VPN down", vec(1, 0.1), now=0)
        detector.add("b", "VPN fails", vec(1, 0.2), now=10)
        self.assertIsNone(detector.add("c", "VPN offline", vec(1, 0.15), now=500))


if __name__ == "__main__":
    unittest.main()
