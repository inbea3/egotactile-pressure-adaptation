"""Invariants for the exploratory Stage 4 object signal audit."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_stage4_object_signal_audit import (  # noqa: E402
    cross_actor_prediction, episode_targets, fixed_subsets, variance_decomposition,
)


class Stage4SignalAuditTests(unittest.TestCase):
    def test_episode_targets_use_only_future_minus_last_observed_bin(self):
        bins = []
        for index, force in enumerate((10., 20., 30., 40., 50., 60.)):
            bins.append({"time_bin_start_seconds": index * 0.5,
                         "region_load_mean_N": {
                             "thumb": force, "index": 0., "middle": 0.,
                             "ring": 0., "little": 0., "palm": 0.,
                         }})
        episode = {
            "excluded_from_signal_diagnostics": False,
            "candidate_onsets": {"two_sensors_gt_5N_3samples": {
                "first_onset": {"onset_seconds_from_episode_start": 0.0}}},
            "time_binned_pressure_trace_0p5s": bins,
        }
        targets, reason = episode_targets(episode, "two_sensors_gt_5N_3samples")
        self.assertIsNone(reason)
        self.assertEqual(targets["share_change"].shape, (4, 6))
        np.testing.assert_allclose(targets["share_change"], 0.)
        np.testing.assert_allclose(
            targets["log_total_change"][:, 0],
            np.log1p([30., 40., 50., 60.]) - np.log1p(20.),
        )

    def test_variance_decomposition_recovers_additive_object_and_actor_signal(self):
        data = {}
        for object_index in range(4):
            data[f"obj{object_index}"] = {}
            for actor_index in range(6):
                data[f"obj{object_index}"][f"actor{actor_index}"] = {
                    "share_change": np.full((4, 6), object_index * 2. + actor_index * 0.5),
                }
        result = variance_decomposition(data, "share_change", permutation_reps=100)
        self.assertAlmostEqual(result["additive_r2"], 1., places=10)
        self.assertGreater(result["incremental_object_after_participant_fraction"], 0.)
        self.assertGreater(result["incremental_participant_after_object_fraction"], 0.)
        self.assertGreater(result["incremental_object_after_participant_fraction"],
                           result["object_increment_permutation_null"]["quantile_95"])

    def test_support_query_separation_and_object_signal(self):
        actors = [f"actor{i}" for i in range(6)]
        subset = fixed_subsets("obj0", "actor0", actors[1:], 2, 5, 20261127)
        self.assertEqual(subset, fixed_subsets("obj0", "actor0", actors[1:], 2, 5, 20261127))
        self.assertTrue(all("actor0" not in row for row in subset))
        data = {}
        for object_index, object_effect in enumerate((-3., -2., -1., 1., 2., 3.)):
            data[f"obj{object_index}"] = {
                actor: {"share_change": np.full((4, 6), object_effect + actor_index * 0.2)}
                for actor_index, actor in enumerate(actors)
            }
        summaries, _ = cross_actor_prediction(data, "share_change", 5, 100, 7)
        gain = summaries["5"]["paired_differences_positive_favors_support"][
            "actor_history_minus_quarter_support"]
        self.assertGreater(gain["mean"], 0.)
        self.assertEqual(gain["objects_improved"], 6)


if __name__ == "__main__":
    unittest.main()

