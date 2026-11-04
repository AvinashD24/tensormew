import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.dashboard import STATIC_DIR, lineage_prior, make_handler


class WorldLayoutTests(unittest.TestCase):
    """The coverage view draws every route map in one picture from the ROM-derived layout file."""

    def test_layout_file_has_the_whole_early_route(self):
        data = json.loads((STATIC_DIR / "world_layout.json").read_text(encoding="utf-8"))
        maps = {m["id"]: m for m in data["maps"]}
        for key in ("24,4", "24,3", "26,3", "26,1", "26,2", "10,5"):  # New Bark ... Violet City
            self.assertTrue(maps[key]["outdoor"], key)
            self.assertIn("ox", maps[key])
            self.assertEqual(len(maps[key]["rows"]), maps[key]["h"])
        for key in ("26,11", "10,7"):  # Violet gatehouse, Violet gym
            self.assertFalse(maps[key]["outdoor"], key)
        self.assertEqual([s["id"] for s in data["stars"]], ["route29", "cherrygrove", "route30", "route31", "violet_gate", "violet_city", "violet_gym", "beat_falkner"])
        self.assertEqual([s["n"] for s in data["stars"]], list(range(1, 9)))
        self.assertEqual(sum(s["reward"] for s in data["stars"]), 140)   # 5+10+10+10+5+15+15 plus 70 for the badge and event
        self.assertEqual(len(data["legs"]), 1)
        self.assertEqual(data["legs"][0]["steps"], 319)
        blob = json.dumps(data["stars"]).lower()
        for word in ("elm", "egg", " rival"):
            self.assertNotIn(word, blob)

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


def _mk_run(runs: Path, name: str, started: float, first: int, last: int) -> None:
    d = runs / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"run_id": name, "started": started}), encoding="utf-8")
    lines = [json.dumps({"update": u, "metrics": {"update": u, "run": name}}) for u in range(first, last + 1)]
    (d / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class LineageTests(unittest.TestCase):
    """A resumed run continues the update numbering of the run it was restored from."""

    def test_chain_of_three_trims_replaced_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            _mk_run(runs, "a", 100.0, 0, 50)    # ran to 50, but c resumed from the 30 checkpoint
            _mk_run(runs, "b", 200.0, 31, 80)   # resumed from a (a.last+1 >= 31), ran past c's start
            _mk_run(runs, "c", 300.0, 61, 90)
            got = lineage_prior(runs, "c")
            self.assertEqual([m["update"] for m in got], list(range(0, 61)))  # b trimmed to < 61, a to < 31
            self.assertEqual([m["run"] for m in got[:31]], ["a"] * 31)
            self.assertEqual({m["run"] for m in got[31:]}, {"b"})
            self.assertEqual([m["update"] for m in got if m.get("resumed")], [31])  # divider where b took over

    def test_replaced_updates_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            _mk_run(runs, "old", 100.0, 0, 103)
            _mk_run(runs, "new", 200.0, 100, 146)
            got = lineage_prior(runs, "new")
            self.assertEqual([m["update"] for m in got], list(range(0, 100)))
            self.assertEqual(lineage_prior(runs, "old"), [])

    def test_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            _mk_run(runs, "a", 100.0, 0, 20)
            _mk_run(runs, "gap", 200.0, 40, 60)     # would need a.last + 1 >= 40
            _mk_run(runs, "fresh", 300.0, 0, 10)    # restarts at 0: nobody starts before it
            _mk_run(runs, "early", 50.0, 45, 70)    # started earlier than a and begins after gap's first update
            self.assertEqual(lineage_prior(runs, "gap"), [])
            self.assertEqual(lineage_prior(runs, "fresh"), [])
            self.assertEqual(lineage_prior(runs, "missing"), [])
            self.assertEqual(lineage_prior(runs, None), [])

    def test_latest_started_candidate_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            _mk_run(runs, "p1", 100.0, 0, 120)
            _mk_run(runs, "p2", 150.0, 0, 110)
            _mk_run(runs, "r", 200.0, 100, 130)
            self.assertEqual({m["run"] for m in lineage_prior(runs, "r")}, {"p2"})

    def test_routes_prepend_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _mk_run(root / "runs", "old", 100.0, 0, 5)
            _mk_run(root / "runs", "new", 200.0, 4, 8)
            (root / "live").mkdir()
            (root / "live" / "state.json").write_text(json.dumps({"run_id": "new"}), encoding="utf-8")
            (root / "m.jsonl").write_text("\n".join(json.dumps({"update": u}) for u in range(4, 9)) + "\n", encoding="utf-8")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root / "live", root / "m.jsonl", root / "runs"))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                full = json.loads(urllib.request.urlopen(base + "/api/metrics?since=-1", timeout=5).read())
                self.assertEqual([m["update"] for m in full], list(range(0, 9)))
                self.assertTrue(full[4].get("resumed"))
                inc = json.loads(urllib.request.urlopen(base + "/api/metrics?since=6", timeout=5).read())
                self.assertEqual([m["update"] for m in inc], [7, 8])
                hist = json.loads(urllib.request.urlopen(base + "/api/run/new/history", timeout=5).read())
                self.assertEqual([m["update"] for m in hist["prior"]], [0, 1, 2, 3])
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
