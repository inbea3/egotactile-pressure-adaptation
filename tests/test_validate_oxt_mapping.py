import unittest
import numpy as np

from scripts.validate_oxt_mapping import validate_mapping_arrays


class ValidateMappingArraysTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "layout_channels": [
                {"source_raw_index_1based": 10, "grid_y": 0, "grid_x": 0},
                {"source_raw_index_1based": 20, "grid_y": 1, "grid_x": 0},
                {"source_raw_index_1based": 20, "grid_y": 2, "grid_x": 0},
            ],
            "physical_sensors": [
                {"multiplicity": 1, "layout_channels": [0]},
                {"multiplicity": 2, "layout_channels": [1, 2]},
            ],
        }

    def test_passes_stable_official_mapping_and_equal_virtual_cells(self):
        raw = np.tile(np.array([10, 20, 20]), (3, 1))
        grid = np.tile(np.array([[0, 0], [0, 1], [0, 2]]), (3, 1, 1))
        pressure = np.array([[1, 2, 2], [1, 3, 3], [1, 4, 4]], dtype=np.float32)
        result = validate_mapping_arrays(self.mapping, raw, grid, pressure)
        self.assertTrue(result["all_checks_pass"])
        self.assertEqual(result["bend_value_mismatch_count"] if "bend_value_mismatch_count" in result else result["bend_virtual_cell_pressure_mismatch_count"], 0)

    def test_detects_mapping_or_virtual_pressure_mismatch(self):
        raw = np.tile(np.array([10, 20, 21]), (2, 1))
        grid = np.tile(np.array([[0, 0], [0, 1], [0, 2]]), (2, 1, 1))
        pressure = np.array([[1, 2, 9], [1, 3, 3]], dtype=np.float32)
        result = validate_mapping_arrays(self.mapping, raw, grid, pressure)
        self.assertFalse(result["all_checks_pass"])
        self.assertFalse(result["raw_indices_match_official_note"])
        self.assertEqual(result["bend_virtual_cell_pressure_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
