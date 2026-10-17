import json
import tempfile
import unittest
from pathlib import Path

from app import load_decisions, summarize


def _row(**overrides):
    base = {
        "timestamp": "2026-10-17T22:00:00+00:00",
        "frame": 120,
        "map": [3, 2],
        "pos": [10, 20],
        "badges": 1,
        "actor": "llm",
        "trigger": "CUT_BLOCKED",
        "tool": "use_cut",
        "arguments": {"pokemon": 0},
        "reason": "tree blocks the path",
        "ok": True,
        "est_usd": 0.001,
        "input_tokens": 100,
        "output_tokens": 20,
    }
    base.update(overrides)
    return base


class LoadDecisionsTests(unittest.TestCase):
    def test_two_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            rows = [
                _row(reason="first", est_usd=0.001),
                _row(reason="second", tool="go_to", map=[21, 14], pos=[4, 8]),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            loaded = load_decisions(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["reason"], "first")
        self.assertEqual(loaded[1]["tool"], "go_to")
        self.assertEqual(loaded[1]["map"], [21, 14])
        self.assertEqual(loaded[1]["pos"], [4, 8])
        self.assertEqual(loaded[1]["arguments"], {"pokemon": 0})

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_decisions(Path(tmp) / "missing.jsonl"), [])

    def test_malformed_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            good = json.dumps(_row(reason="kept"))
            path.write_text(f"not json\n{good}\n{{broken\n\n[1, 2]\n", encoding="utf-8")
            loaded = load_decisions(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["reason"], "kept")


class SummarizeTests(unittest.TestCase):
    def test_sums_cost_tokens_and_last_reason(self):
        rows = [
            _row(reason="first", est_usd=0.001, input_tokens=10, output_tokens=2, badges=1),
            _row(reason="cut the tree", est_usd=0.004, input_tokens=30, output_tokens=8, badges=2),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["steps"], 2)
        self.assertAlmostEqual(summary["est_usd"], 0.005)
        self.assertEqual(summary["tokens"], 50)
        self.assertEqual(summary["latest_reason"], "cut the tree")
        self.assertEqual(summary["badges"], 2)

    def test_empty_rows(self):
        summary = summarize([])
        self.assertEqual(
            summary,
            {"steps": 0, "est_usd": 0.0, "tokens": 0, "latest_reason": "", "badges": 0},
        )


if __name__ == "__main__":
    unittest.main()
