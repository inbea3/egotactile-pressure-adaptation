import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_full_egotactile_inventory import run_full_inventory


class FakeInfo:
    sha = "pinned"


class FakeFile:
    def __init__(self, path, size, sha):
        self.path = path
        self.size = size
        self.lfs = type("Lfs", (), {"sha256": sha})()


class FullInventoryTests(unittest.TestCase):
    @patch("scripts.run_full_egotactile_inventory.audit_oxt_zarr_archive")
    @patch("scripts.run_full_egotactile_inventory.sha256_file")
    @patch("scripts.run_full_egotactile_inventory.hf_hub_download")
    @patch("scripts.run_full_egotactile_inventory.HfApi")
    def test_requires_exactly_63_object_archives(self, api_cls, download, sha, audit):
        api = api_cls.return_value
        api.dataset_info.return_value = FakeInfo()
        api.list_repo_tree.return_value = [FakeFile("EgoTactile/Apple.zip", 1, "x")]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_full_inventory(Path(tmp) / "out.json", Path(tmp) / "cache")

    @patch("scripts.run_full_egotactile_inventory.audit_oxt_zarr_archive")
    @patch("scripts.run_full_egotactile_inventory.sha256_file")
    @patch("scripts.run_full_egotactile_inventory.hf_hub_download")
    @patch("scripts.run_full_egotactile_inventory.HfApi")
    def test_aggregates_object_shot_gates(self, api_cls, download, sha, audit):
        api = api_cls.return_value
        api.dataset_info.return_value = FakeInfo()
        api.list_repo_tree.return_value = [
            FakeFile(f"EgoTactile/Object{i:02}.zip", 1, "x") for i in range(63)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "object.zip"
            local.write_bytes(b"x")
            download.return_value = str(local)
            sha.return_value = "x"
            audit.side_effect = lambda path, name: {
                "object_name": name,
                "frame_count": 10,
                "episode_count": 6,
                "episode_counts": {"gloved_hand_only": 6, "preliminary_3s_window_eligible": 6},
                "episodes": [{"preliminary_3s_window_eligible": True} for _ in range(6)],
                "eligible_trajectory_counts_by_participant": {f"p{i}": 1 for i in range(6)},
                "unique_raw_sensor_indices_in_first_frame": 137,
                "raw_index_mapping_constant_across_frames": True,
                "tactile_timestamps": {
                    "negative_interval_count": 0,
                    "duplicate_interval_count": 0,
                    "duplicate_pressure_rows_identical_to_previous": 0,
                    "duplicate_pressure_rows_different_from_previous": 0,
                },
                "pressure_statistics_162": {"finite_fraction": 1.0},
            }
            report = run_full_inventory(Path(tmp) / "out.json", Path(tmp) / "cache")
            self.assertEqual(report["object_archive_count"], 63)
            self.assertEqual(report["objects_meeting_preliminary_shot_gate"]["5"], 63)
            self.assertFalse(report["video_decoded"])


if __name__ == "__main__":
    unittest.main()
