import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.inspect_oxt_archive import inspect_archive


class InspectOxtArchiveTests(unittest.TestCase):
    def test_reads_only_small_metadata_and_does_not_decode_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "toy.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("object.zarr/.zgroup", json.dumps({"zarr_format": 2}))
                archive.writestr(
                    "object.zarr/pressure/.zarray",
                    json.dumps({
                        "shape": [100, 256],
                        "chunks": [10, 256],
                        "dtype": "<f4",
                        "compressor": None,
                        "filters": None,
                        "order": "C",
                    }),
                )
                archive.writestr(
                    "object.zarr/meta/.zattrs",
                    json.dumps({"episode_ends": [40, 100], "other": "ignored"}),
                )
                archive.writestr("object.zarr/video.mp4", b"not decoded")

            result = inspect_archive(archive_path)

            self.assertTrue(result["zip_readable"])
            self.assertEqual(result["member_count"], 4)
            self.assertEqual(result["zarr_arrays"][0]["shape"], [100, 256])
            self.assertEqual(
                result["zarr_attributes"][0]["selected_values"]["episode_ends"],
                [40, 100],
            )
            self.assertEqual(len(result["video_members_not_decoded"]), 1)
            self.assertIn("sha256", result)

    def test_reports_malformed_metadata_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("object.zarr/.zattrs", "{bad json")
            result = inspect_archive(archive_path)
            self.assertEqual(len(result["metadata_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
