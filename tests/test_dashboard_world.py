import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.dashboard import STATIC_DIR, make_handler


class WorldLayoutTests(unittest.TestCase):
    """The coverage view draws every route map in one picture from the ROM-derived layout file."""

    def test_layout_file_has_the_whole_early_route(self):
        data = json.loads((STATIC_DIR / "world_layout.json").read_text(encoding="utf-8"))
        maps = {m["id"]: m for m in data["maps"]}
        for key in ("24,4", "24,3", "26,3", "26,1", "26,2", "10,5"):  # New Bark ... Violet City
            self.assertTrue(maps[key]["outdoor"], key)
            self.assertIn("ox", maps[key])
            self.assertEqual(len(maps[key]["rows"]), maps[key]["h"])
        for key in ("24,5", "26,10", "10,7"):  # Elm's lab, Mr. Pokemon's house, Violet gym
            self.assertFalse(maps[key]["outdoor"], key)
        self.assertEqual([s["n"] for s in data["stars"]], [1, 2, 3, 4, 5])

    def test_world_route_and_page_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Path(tmp) / "live", Path(tmp) / "m.jsonl", Path(tmp) / "runs"))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                world = json.loads(urllib.request.urlopen(base + "/api/world", timeout=5).read())
                self.assertGreaterEqual(len(world["maps"]), 12)
                page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
                self.assertIn("renderWorld", page)
                self.assertNotIn('id="mapTabs"', page)      # no per-map tabs any more
                self.assertNotIn('id="envs"', page)         # no per-agent screen grid
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
