import unittest

from tools.run_campaign import campaign_names, summarize


class CampaignTests(unittest.TestCase):
    def test_names_are_sorted(self):
        manifest = {"states": {"b": {}, "a": {}, "c": {}}}
        self.assertEqual(campaign_names(manifest), ["a", "b", "c"])

    def test_summarize_counts_map_changes(self):
        rows = [
            {"name": "a", "outcome": "advanced", "start_map": (1, 1), "end_map": (1, 2)},
            {"name": "b", "outcome": "budget", "start_map": (2, 1), "end_map": (2, 1)},
        ]
        self.assertEqual(summarize(rows), {"ran": 2, "advanced": 1, "left_map": 1})


if __name__ == "__main__":
    unittest.main()
