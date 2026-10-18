import json
import tempfile
import unittest
from pathlib import Path

from src.knowledge import lookup_boss
from src.saycan import evaluate


class EvaluateTests(unittest.TestCase):
    def test_veto_empty_party(self):
        result = evaluate({"badges": 0, "party_count": 0, "levels": []}, "Falkner")
        self.assertEqual(result["verdict"], "VETO")
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"])

    def test_approve_level_20_party_of_two(self):
        party = {
            "badges": 1,
            "party_count": 2,
            "levels": [20, 16],
            "types": ["fire", "normal"],
        }
        result = evaluate(party, "Bugsy")
        self.assertEqual(result["verdict"], "APPROVE")
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"])


class LookupTests(unittest.TestCase):
    def test_missing_file_returns_none(self):
        missing = Path(tempfile.gettempdir()) / "tensormew-missing-bosses.json"
        if missing.exists():
            missing.unlink()
        self.assertIsNone(lookup_boss("Falkner", path=missing))

    def test_unknown_name_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bosses.json"
            path.write_text("{}", encoding="utf-8")
            self.assertIsNone(lookup_boss("Whitney", path=path))
        self.assertIsNone(lookup_boss("Whitney"))

    def test_temp_file_finds_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bosses.json"
            path.write_text(
                json.dumps({"Falkner": {"gym": "Violet"}}),
                encoding="utf-8",
            )
            self.assertEqual(lookup_boss("Falkner", path=path), {"gym": "Violet"})
            self.assertIsNone(lookup_boss("Bugsy", path=path))
