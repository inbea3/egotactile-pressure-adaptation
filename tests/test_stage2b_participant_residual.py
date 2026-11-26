"""Small deterministic checks for the Stage 2B split and aggregation."""
import unittest
from unittest.mock import patch

import numpy as np

from run_stage2b_participant_residual import evaluate_policy, global_mean


def synthetic_data():
    result = {}
    for object_index, name in enumerate(("A", "B", "C")):
        result[name] = {}
        for subject in range(6):
            early = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
            delta = np.array([0.01 * object_index + 0.002 * subject,
                              -0.01 * object_index - 0.002 * subject, 0, 0, 0, 0])
            result[name][f"p{subject}"] = {
                "early": early, "future": early + delta, "delta": delta,
                "valid_cycle_count": 5,
            }
    return result


class Stage2BTests(unittest.TestCase):
    def test_global_excludes_query_subject_across_training_objects(self):
        data = synthetic_data()
        known = global_mean(data, ["B", "C"], "delta")
        blocked = global_mean(data, ["B", "C"], "delta", exclude_subject="p5")
        self.assertGreater(known[0], blocked[0])
        expected = np.mean([data[name][f"p{i}"]["delta"]
                            for name in ("B", "C") for i in range(5)], axis=0)
        np.testing.assert_allclose(blocked, expected)

    def test_shots_and_object_equal_aggregation(self):
        data = synthetic_data()
        with patch("run_stage2b_participant_residual.extract_participant_vectors", return_value=data):
            report = evaluate_policy({"revision": "synthetic"}, "all_detected_events", repetitions=40)
        self.assertEqual(report["object_count"], 3)
        for shot, combinations in ((1, 30), (2, 60), (5, 6)):
            rows = report["object_metrics"][str(shot)]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["support_query_combinations"] == combinations for row in rows))
            self.assertIn("global_query_subject_blocked", report["shot_summaries"][str(shot)]["methods"])
        self.assertIn("global_minus_object_residual_template", report["residual_summary"])


if __name__ == "__main__":
    unittest.main()

