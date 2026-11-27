"""Shape, adaptation, and object-boundary checks for the Stage 4 model pilot."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_stage4_log_region_model_pilot import (  # noqa: E402
    actor_history, adapted_predict, make_model, predict,
)


class Stage4LogRegionPilotTests(unittest.TestCase):
    def test_support_adaptation_uses_two_early_bins_and_four_future_bins(self):
        model = make_model(31)
        record = {"early_log_region": np.ones((2, 6), dtype=np.float64),
                  "log_region_change": np.ones((4, 6), dtype=np.float64)}
        before = predict(model, record)
        after = adapted_predict(model, [record], record, inner_steps=3, inner_lr=0.05)
        self.assertEqual(before.shape, (4, 6))
        self.assertEqual(after.shape, (4, 6))
        self.assertLess(np.mean((after - record["log_region_change"]) ** 2),
                        np.mean((before - record["log_region_change"]) ** 2))
        np.testing.assert_allclose(predict(model, record), before)

    def test_actor_history_only_uses_named_training_objects(self):
        data = {
            "train": {"p001": {"log_region_change": np.full((4, 6), 2.)}},
            "heldout": {"p001": {"log_region_change": np.full((4, 6), 100.)}},
        }
        np.testing.assert_allclose(actor_history(data, ["train"], "p001"), 2.)
        self.assertIsNone(actor_history(data, ["train"], "p002"))


if __name__ == "__main__":
    unittest.main()

