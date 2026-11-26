import unittest
from unittest.mock import patch

from scripts.run_full_stage1_event_inventory import discover_object_names


class FullStage1InventoryTests(unittest.TestCase):
    @patch("scripts.run_full_stage1_event_inventory.HfApi")
    def test_discovers_exactly_63_object_archives(self, api_cls):
        api = api_cls.return_value
        api.list_repo_tree.return_value = [
            type("File", (), {"path": f"EgoTactile/Object{i:02}.zip"})()
            for i in range(63)
        ]
        names = discover_object_names()
        self.assertEqual(len(names), 63)
        self.assertEqual(names[0], "Object00")
        self.assertEqual(names[-1], "Object62")

    @patch("scripts.run_full_stage1_event_inventory.HfApi")
    def test_stops_if_dataset_object_count_changes(self, api_cls):
        api_cls.return_value.list_repo_tree.return_value = [
            type("File", (), {"path": "EgoTactile/OnlyOne.zip"})()
        ]
        with self.assertRaises(RuntimeError):
            discover_object_names()


if __name__ == "__main__":
    unittest.main()
