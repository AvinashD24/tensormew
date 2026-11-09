import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "tools" / "dashboard_static" / "index.html"


class DashboardChartsPageTests(unittest.TestCase):
    def test_new_elements_present(self):
        html = PAGE.read_text(encoding="utf-8")
        for el in ("learnChecks", "rewardVerdict", "smoothSeg", "cycleNote", "ppoGrid", "chReward", "rewardLegend"):
            self.assertIn(f'id="{el}"', html)
        for fn in ("rewardVerdict", "cycleDetected", "movAvg", "renderPpo", "renderLearning"):
            self.assertIn(f"function {fn}(", html)

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
