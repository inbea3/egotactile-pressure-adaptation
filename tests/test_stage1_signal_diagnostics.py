import unittest
import numpy as np

from scripts.run_stage1_signal_diagnostics import deduplicate_episode, find_sustained_onset


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


if __name__ == "__main__":
    unittest.main()
