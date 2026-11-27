"""Check task boundaries and first-order meta update for the Stage 5 pilot."""
import sys
from pathlib import Path
import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_stage5_fomaml_pilot import choose_task, fomaml_train, make_model  # noqa: E402


@unittest.skipIf(torch is None, "requires PyTorch; run on AutoDL")
class Stage5FomamlPilotTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "object_a": {
                f"actor_{index}": {
                    "early_log_region": np.full((2, 6), index / 10),
                    "log_region_change": np.full((4, 6), 0.5 + index / 20),
                }
                for index in range(7)
            }
        }

    def test_support_and_query_are_distinct_participant_records(self):
        rng = np.random.default_rng(14)
        for _ in range(100):
            name, support, query = choose_task(
                self.data, ["object_a"], rng, query_size=3)
            self.assertEqual(name, "object_a")
            self.assertIn(len(support), (1, 2, 5))
            self.assertTrue(query)
            self.assertTrue(set(support).isdisjoint(query))

    def test_query_gradient_changes_initialization(self):
        torch.set_num_threads(1)
        initial = make_model(19)
        trained, diagnostics = fomaml_train(
            self.data, ["object_a"], seed=19, iterations=3,
            inner_steps=2, inner_lr=0.05, outer_lr=0.001, query_size=2)
        self.assertTrue(np.isfinite(diagnostics["last_100_training_query_mse"]))
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(initial.parameters(), trained.parameters())
        ))


if __name__ == "__main__":
    unittest.main()

