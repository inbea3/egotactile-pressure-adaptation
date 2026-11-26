import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.fetch_tactile_mapping_note import inspect_mapping_note


class FakeDownload:
    pass


class MappingNoteTests(unittest.TestCase):
    @patch("scripts.fetch_tactile_mapping_note.sha256_file", return_value="abc123")
    @patch("scripts.fetch_tactile_mapping_note.hf_hub_download")
    def test_fetches_pinned_mapping_note_and_preserves_raw_json(self, download, sha):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "note.json"
            source.write_text(json.dumps({"layout": [1, 2], "regions": {"palm": [1]}}))
            download.return_value = str(source)
            output = Path(tmp) / "out.json"
            report = inspect_mapping_note(Path(tmp) / "cache", output)
            self.assertEqual(report["revision"], "5b05f19e613758e76c330671f4c9f3c06af8af44")
            self.assertEqual(report["sha256"], "abc123")
            self.assertEqual(report["raw_mapping_note"]["layout"], [1, 2])
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
