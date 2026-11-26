import unittest
from unittest.mock import patch

from scripts.build_egotactile_manifest import build_manifest


class FakeInfo:
    sha = "deadbeef"


class FakeFile:
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.lfs = None


class ManifestTests(unittest.TestCase):
    @patch("scripts.build_egotactile_manifest.HfApi")
    def test_manifest_pins_revision_and_only_lists_target_files(self, api_cls):
        api = api_cls.return_value
        api.dataset_info.return_value = FakeInfo()
        api.list_repo_tree.return_value = [
            FakeFile("EgoTactile/Apple.zip", 100),
            FakeFile("EgoTactile/Bottle.zip", 200),
            FakeFile("README.md", 50),
            FakeFile("supplementary_tactile_note.json", 20),
        ]
        result = build_manifest()
        self.assertEqual(result["revision"], "deadbeef")
        self.assertEqual(result["object_archive_count"], 2)
        self.assertEqual(result["object_archives_total_bytes"], 300)
        self.assertFalse(result["downloaded_any_dataset_archive"])
        self.assertEqual(len(result["files"]), 3)


if __name__ == "__main__":
    unittest.main()
