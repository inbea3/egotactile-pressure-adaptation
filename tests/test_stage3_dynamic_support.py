"""Small split and timing checks for the Stage 3 derived-pressure experiment."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_stage3_dynamic_support import (
    contact_row, evaluate_rule, recording_vector,
)


def bin_record(index, loads):
    names = ("thumb", "index", "middle", "ring", "little", "palm")
    return {
        "time_bin_start_seconds": index * 0.5,
        "region_load_mean_N": dict(zip(names, loads)),
    }


def episode(subject, future_thumb=12.0):
    trace = [bin_record(i, [12.0, 8.0, 0.0, 0.0, 0.0, 0.0]) for i in range(8)]
    trace[3] = bin_record(3, [future_thumb, 8.0, 0.0, 0.0, 0.0, 0.0])
    trace[4] = bin_record(4, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return {
        "subject_id": subject,
        "excluded_from_signal_diagnostics": False,
        "candidate_onsets": {
            "two_sensors_gt_5N_3samples": {
                "first_onset": {"onset_seconds_from_episode_start": 0.25}
            },
        },
        "time_binned_pressure_trace_0p5s": trace,
    }


class Stage3Tests(unittest.TestCase):
    def test_future_bin_does_not_enter_observation(self):
        left, reason = recording_vector(episode("p0", future_thumb=12.0), "two_sensors_gt_5N_3samples")
        right, _ = recording_vector(episode("p0", future_thumb=90.0), "two_sensors_gt_5N_3samples")
        self.assertIsNone(reason)
        np.testing.assert_allclose(left["early"], right["early"])
        self.assertEqual(left["cutoff_seconds_from_recording_start"], 1.5)
        self.assertNotAlmostEqual(left["future"][0, 0], right["future"][0, 0])
        self.assertEqual(left["future"][1, 6], 1.0)
        np.testing.assert_allclose(left["future"].sum(axis=1), np.ones(4))

    def test_missing_bin_is_excluded(self):
        row = episode("p0")
        row["time_binned_pressure_trace_0p5s"].pop(5)
        vector, reason = recording_vector(row, "two_sensors_gt_5N_3samples")
        self.assertIsNone(vector)
        self.assertEqual(reason, "missing_required_time_bin")

    def test_contact_threshold_encodes_no_contact(self):
        encoded = contact_row(bin_record(0, [2, 2, 0, 0, 0, 0]))
        np.testing.assert_array_equal(encoded, np.array([0, 0, 0, 0, 0, 0, 1]))

    def test_query_subject_is_blocked_and_shots_are_recordings(self):
        data = {}
        for object_index, name in enumerate(("A", "B", "C")):
            data[name] = {}
            for subject in range(6):
                early = np.tile(np.array([0.6, 0.4, 0, 0, 0, 0, 0]), (2, 1))
                future = np.tile(np.array([0.55 + 0.005 * object_index + 0.002 * subject,
                                           0.45 - 0.005 * object_index - 0.002 * subject,
                                           0, 0, 0, 0, 0]), (4, 1))
                data[name][f"p{subject}"] = {
                    "early": early, "future": future,
                    "delta": future - early[-1][None, :],
                }
        with patch("scripts.run_stage3_dynamic_support.extract_vectors", return_value=(data, {})):
            report = evaluate_rule({"revision": "synthetic"}, "two_sensors_gt_5N_3samples", repetitions=40)
        self.assertEqual(report["shot_summaries"]["5"]["object_count"], 3)
        self.assertTrue(all(row["support_query_combinations"] == 6
                            for row in report["object_metrics"]["5"]))
        self.assertIn("blocked_global_minus_fixed_quarter_shrinkage_blocked",
                      report["shot_summaries"]["5"]["paired_differences_positive_favors_adaptation"])


if __name__ == "__main__":
    unittest.main()

