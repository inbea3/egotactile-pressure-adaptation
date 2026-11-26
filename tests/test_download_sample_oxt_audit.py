import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_sample_oxt_audit import run_sample_audit


class FakeInfo:
    sha = "pinned-sha"


class FakeFile:
    def __init__(self, path, size):
        self.path = path
        self.size = size


class SampleAuditTests(unittest.TestCase):
    @patch("scripts.download_sample_oxt_audit.audit_oxt_zarr_archive")
    @patch("scripts.download_sample_oxt_audit.inspect_archive")
    @patch("scripts.download_sample_oxt_audit.hf_hub_download")
    @patch("scripts.download_sample_oxt_audit.HfApi")
    def test_uses_pinned_revision_and_reports_metadata_only(self, api_cls, download, inspect, pressure_audit):
        api = api_cls.return_value
        api.dataset_info.return_value = FakeInfo()
        api.list_repo_tree.return_value = [
            FakeFile("EgoTactile/Apple.zip", 4),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "Apple.zip"
            local.write_bytes(b"data")
            download.return_value = str(local)
            inspect.return_value = {"archive": "Apple.zip", "member_count": 1}
            pressure_audit.return_value = {"episode_count": 1}
            output = Path(tmp) / "out.json"
            report = run_sample_audit(["Apple"], output, Path(tmp) / "cache")
            self.assertEqual(report["revision"], "pinned-sha")
            self.assertFalse(report["pressure_arrays_decoded"])
            self.assertEqual(report["objects_completed"], 1)
            self.assertEqual(json.loads(output.read_text())["objects_requested"], ["Apple"])

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_sample_audit(["../bad"], Path(tmp) / "out.json", Path(tmp) / "cache")


if __name__ == "__main__":
    unittest.main()
