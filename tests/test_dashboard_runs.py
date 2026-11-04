import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.dashboard import classify_run, list_runs

PAGE = Path(__file__).resolve().parents[1] / "tools" / "dashboard_static" / "index.html"
NOW = 1_800_000_000.0


def mk(root: Path, name: str, age: float, finished: bool = False) -> None:
    d = root / name
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"started": NOW - 9999, "finished": finished, "updates_logged": 5}))
    h = d / "history.jsonl"
    h.write_text("{}\n")
    os.utime(h, (NOW - age, NOW - age))


class RunStatusTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_run(False, NOW - 10, True, NOW), "live")
        self.assertEqual(classify_run(False, NOW - 1000, True, NOW), "stopped")
        self.assertEqual(classify_run(False, NOW - 10, False, NOW), "stopped")
        self.assertEqual(classify_run(True, NOW - 10, True, NOW), "finished")

    def test_order_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mk(root, "old", 90000)
            mk(root, "mid", 5000)
            mk(root, "fresh", 20)
            mk(root, "done", 100, finished=True)
            runs = list_runs(root, "mid", NOW)
            self.assertEqual([r["id"] for r in runs], ["fresh", "done", "mid", "old"])  # nothing live: by recency
            mk(root, "cur", 30)
            runs = list_runs(root, "cur", NOW)
            self.assertEqual(runs[0]["id"], "cur")
            self.assertEqual(runs[0]["status"], "live")
            self.assertEqual({r["id"]: r["status"] for r in runs}["old"], "stopped")
            self.assertEqual({r["id"]: r["status"] for r in runs}["done"], "finished")


class RunPickerPageTests(unittest.TestCase):
    def test_wiring(self):
        html = PAGE.read_text(encoding="utf-8")
        for s in ('id="backLive"', 'id="staleBanner"', "LIVE \\u2022", "localStorage", "Live feed stopped", 'status === "live"'):
            self.assertIn(s, html)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_script_parses(self):
        html = PAGE.read_text(encoding="utf-8")
        script = html[html.rindex("<script>") + 8:html.rindex("</script>")]
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "page.js"
            f.write_text(script, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
