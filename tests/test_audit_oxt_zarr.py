import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from scripts.audit_oxt_zarr import audit_oxt_zarr_archive


class AuditOxtZarrTests(unittest.TestCase):
    def test_counts_complete_gloved_direct_episodes_without_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Toy.zip"
            store = zarr.ZipStore(str(path), mode="w")
            root = zarr.group(store=store)
            obj = root.create_group("Toy.zarr")
            data = obj.create_group("data")
            meta = obj.create_group("meta")
            n = 8
            data.array("acquisition_mode", np.array(["gloved_hand"] * n, dtype="U12"), chunks=(4,))
            data.array("pairing_quality", np.array(["direct"] * n, dtype="U8"), chunks=(4,))
            data.array("subject_id", np.array(["p001"] * n, dtype="U8"), chunks=(4,))
            data.array("timestamps", np.arange(n, dtype=np.float64) / 2, chunks=(4,))
            data.array("tactile_timestamps", np.arange(n, dtype=np.float64) / 2, chunks=(4,))
            data.array("right_tactile_data_pressure", np.ones((n, 162), dtype=np.float32), chunks=(4, 162))
            data.array("right_tactile_raw_pressure_256", np.ones((n, 256), dtype=np.uint16), chunks=(4, 256))
            idx = np.tile(np.arange(162, dtype=np.int64), (n, 1))
            idx[:, -25:] = np.arange(137, 162)
            data.array("right_tactile_raw_index_pressure", idx, chunks=(4, 162))
            meta.array("episode_ends", np.array([4, 8], dtype=np.int64), chunks=(2,))
            store.close()

            result = audit_oxt_zarr_archive(path, "Toy")
            self.assertEqual(result["episode_count"], 2)
            self.assertEqual(result["episode_counts"]["gloved_hand_only"], 2)
            self.assertEqual(result["episode_counts"]["direct_pairing_only"], 2)
            self.assertEqual(result["unique_raw_sensor_indices_in_first_frame"], 162)
            self.assertFalse(result["video_decoded"])


if __name__ == "__main__":
    unittest.main()
