import unittest

from scripts.build_sensor_mapping import build_sensor_mapping


class BuildSensorMappingTests(unittest.TestCase):
    def test_builds_physical_groups_and_regions(self):
        areas = {"0": "thumb_tip", "1": "thumb_bend", "2": "palm"}
        descriptions = {
            "sensor_idx_0": "OXT channel 0: thumb_tip (functional_area_idx=0), grid (y=0, x=0), source raw pressure index=10 (1-based).",
            "sensor_idx_1": "OXT channel 1: thumb_bend (functional_area_idx=1), grid (y=1, x=0), source raw pressure index=20 (1-based), value divided by 6 across virtual bend cells.",
            "sensor_idx_2": "OXT channel 2: thumb_bend (functional_area_idx=1), grid (y=2, x=0), source raw pressure index=20 (1-based), value divided by 6 across virtual bend cells.",
            "sensor_idx_3": "OXT channel 3: thumb_bend (functional_area_idx=1), grid (y=3, x=0), source raw pressure index=20 (1-based), value divided by 6 across virtual bend cells.",
            "sensor_idx_4": "OXT channel 4: thumb_bend (functional_area_idx=1), grid (y=4, x=0), source raw pressure index=20 (1-based), value divided by 6 across virtual bend cells.",
            "sensor_idx_5": "OXT channel 5: thumb_bend (functional_area_idx=1), grid (y=5, x=0), source raw pressure index=20 (1-based), value divided by 6 across virtual bend cells.",
            "sensor_idx_6": "OXT channel 6: thumb_bend (functional_area_idx=1), grid (y=6, x=0), source raw pressure index=20 (1-based), value divided by 6 across virtual bend cells.",
            "sensor_idx_7": "OXT channel 7: palm (functional_area_idx=2), grid (y=7, x=0), source raw pressure index=30 (1-based).",
        }
        note = {
            "EgoTactilePressureGlove": {
                "sensor_num": 8,
                "functional_area_idx_description": areas,
                "sensor_idx_description": descriptions,
            }
        }
        result = build_sensor_mapping(note, expected_layout=8, expected_unique=3)
        self.assertEqual(result["layout_positions"], 8)
        self.assertEqual(result["unique_physical_sensors"], 3)
        self.assertEqual(result["multiplicity_distribution"], {1: 2, 6: 1})
        thumb = next(x for x in result["regions"] if x["region"] == "thumb")
        self.assertEqual(thumb["unique_physical_sensor_count"], 2)
        self.assertEqual(thumb["layout_channel_count"], 7)
        bend = next(x for x in result["physical_sensors"] if x["source_raw_index_1based"] == 20)
        self.assertEqual(bend["aggregation_rule"], "sum_virtual_cells")


if __name__ == "__main__":
    unittest.main()
