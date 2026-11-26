import unittest
import numpy as np

from scripts.run_no_model_support_test import project_to_simplex, run_no_model_support_test


class NoModelSupportTests(unittest.TestCase):
    def test_simplex_projection(self):
        result = project_to_simplex(np.array([-0.2, 0.3, 0.4, 0.5, 0.1, 0.2]))
        self.assertTrue(np.all(result >= 0))
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_runs_participant_disjoint_leave_one_object_out_support_test(self):
        objects = []
        for obj_idx in range(4):
            episodes = []
            for person_idx in range(6):
                early = np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
                shift = 0.01 * obj_idx
                future = np.array([0.2 + shift, 0.2, 0.2, 0.15, 0.15 - shift, 0.1])
                cycles = [{
                    "early_region_load_share": early.tolist(),
                    "future_region_load_share": future.tolist(),
                    "early_future_window_covered_without_next_cycle_overlap": True,
                }]
                episodes.append({
                    "subject_id": f"p{person_idx}",
                    "excluded_from_signal_diagnostics": False,
                    "grasp_cycle_candidates": {"0.25": {"cycles": cycles}},
                })
            objects.append({"object_name": f"Object{obj_idx}", "episodes": episodes})
        report = {
            "revision": "test",
            "objects": objects,
        }
        result = run_no_model_support_test(report, bootstrap_reps=100, seed=1)
        self.assertEqual(result["object_count_common_5shot_set"], 4)
        self.assertEqual(result["shot_summaries"]["5"]["object_count"], 4)
        self.assertTrue(result["support_query_participants_are_disjoint_within_object"])
        self.assertFalse(result["model_training_performed"])
        for shot, rows in result["object_level_metrics"].items():
            for row in rows:
                self.assertAlmostEqual(row["shrinkage_mae_by_lambda"]["0.0"], row["global_transition_mae"], places=10)
                self.assertAlmostEqual(row["shrinkage_mae_by_lambda"]["1.0"], row["support_correction_mae"], places=10)


if __name__ == "__main__":
    unittest.main()
