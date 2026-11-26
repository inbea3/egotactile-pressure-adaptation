import unittest
import numpy as np

from scripts.run_stage1_signal_diagnostics import deduplicate_episode, find_sustained_onset, detect_contact_segments


class Stage1SignalDiagnosticsTests(unittest.TestCase):
    def test_deduplicates_identical_pressure_at_same_timestamp(self):
        ts = np.array([0.0, 0.1, 0.1, 0.2])
        p = np.array([[1.0], [2.0], [2.0], [3.0]])
        new_ts, new_p, qc = deduplicate_episode(ts, p)
        self.assertEqual(len(new_ts), 3)
        self.assertEqual(qc["duplicate_rows_removed"], 1)
        self.assertTrue(qc["timestamp_pressure_integrity_ok"])

    def test_flags_conflicting_pressure_at_same_timestamp(self):
        ts = np.array([0.0, 0.1, 0.1, 0.2])
        p = np.array([[1.0], [2.0], [9.0], [3.0]])
        _, _, qc = deduplicate_episode(ts, p)
        self.assertEqual(qc["duplicate_pressure_conflict_count"], 1)
        self.assertFalse(qc["timestamp_pressure_integrity_ok"])

    def test_detects_three_sample_sustained_onset(self):
        ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        condition = np.array([False, True, True, True, False])
        self.assertEqual(find_sustained_onset(ts, condition), 1)

    def test_rejects_onset_when_sample_gap_is_too_large(self):
        ts = np.array([0.0, 0.1, 1.0, 1.1, 1.2])
        condition = np.array([False, True, True, True, False])
        self.assertIsNone(find_sustained_onset(ts, condition, max_gap=0.5))

    def test_detects_two_contact_cycles_separated_by_release(self):
        ts = np.arange(12, dtype=np.float64) * 0.1
        high = np.array([False, True, True, True, True, False, False, False, True, True, True, True])
        low = ~high
        segments = detect_contact_segments(ts, high, low, start_run=3, release_run=3)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["start_index"], 1)
        self.assertEqual(segments[1]["start_index"], 8)


if __name__ == "__main__":
    unittest.main()
