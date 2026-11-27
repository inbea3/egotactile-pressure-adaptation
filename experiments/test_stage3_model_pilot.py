"""Split and adaptation checks for the bounded Stage 3 model pilot."""
import unittest

import numpy as np

from scripts.run_stage3_model_pilot import (
    adapted_predict, fixed_subsets, make_model, object_folds, predict,
)


class Stage3ModelPilotTests(unittest.TestCase):
    def test_object_folds_are_disjoint_and_complete(self):
        names = [f"object_{index}" for index in range(63)]
        folds = object_folds(names, 20261127, 5)
        self.assertEqual(sorted(name for fold in folds for name in fold), sorted(names))
        self.assertEqual(len(set(name for fold in folds for name in fold)), 63)
        self.assertLessEqual(max(map(len, folds)) - min(map(len, folds)), 1)

    def test_support_subsets_are_repeatable_and_exclude_query(self):
        supports = [f"p{index}" for index in range(1, 11)]
        first = fixed_subsets("object_A", "p0", supports, 5, 5, 20261127)
        second = fixed_subsets("object_A", "p0", supports, 5, 5, 20261127)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all("p0" not in subset and len(subset) == 5 for subset in first))

    def test_shared_initialization_and_fine_tuning(self):
        early = np.tile(np.array([0.6, 0.4, 0, 0, 0, 0, 0]), (2, 1))
        future = np.tile(np.array([0.4, 0.6, 0, 0, 0, 0, 0]), (4, 1))
        support = {"early": early, "future": future,
                   "delta": future - early[-1][None, :]}
        query = {"early": early, "future": future,
                 "delta": future - early[-1][None, :]}
        model = make_model(5)
        before = predict(model, query)
        np.testing.assert_allclose(before, np.tile(early[-1], (4, 1)), atol=1e-6)
        after = adapted_predict(model, [support], query, inner_steps=3, inner_lr=0.05)
        self.assertLess(np.mean(np.abs(after - future)), np.mean(np.abs(before - future)))


if __name__ == "__main__":
    unittest.main()

