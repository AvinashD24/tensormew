import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from src.live_feed import LiveFeed, dex_ids
from tools.dashboard import make_handler


def _bits(*ids):
    b = bytearray(32)
    for sp in ids:
        b[(sp - 1) // 8] |= 1 << ((sp - 1) % 8)
    return bytes(b)


def _info(bm=0, es=0, own=(), seen=(), lvl=5, comps=None, map_=(24, 4)):
    return {"map": list(map_), "x": 1, "y": 1, "state": "s", "battle": bool(bm), "badges": 0, "ended_maps": 1,
            "reward_components": comps or {},
            "fun": {"bm": bm, "es": es, "ps": [155], "pl": [lvl], "dc": _bits(*own), "ds": _bits(*seen), "mo": 300}}


class FunFactsTests(unittest.TestCase):
    def test_dex_ids(self):
        self.assertEqual(dex_ids(_bits(1, 9, 251)), {1, 9, 251})

    def test_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = LiveFeed(Path(tmp) / "live", num_envs=1, action_names=["a"], history_dir=Path(tmp) / "h")
            steps = [
                _info(own=(155,), seen=(155,)),
                _info(bm=1, es=16, own=(155,), seen=(155, 16)),        # wild Pidgey starts
                _info(bm=1, es=16, own=(155,), seen=(155, 16)),        # same fight: not recounted
                _info(bm=0, own=(155,), seen=(155, 16)),
                _info(bm=2, es=19, own=(155,), seen=(155, 16, 19), lvl=9),   # trainer Rattata
                _info(bm=0, own=(155, 16), seen=(155, 16, 19), comps={"whiteout": -5.0}),   # caught Pidgey
            ]
            for inf in steps:
                feed.observe(infos=[inf], actions=np.array([0]), rewards=np.array([0.0]), dones=np.array([False]))
            f = feed.fun_payload()
            self.assertEqual(f["enc"], {"16": 1, "19": 1})
            self.assertEqual(f["got"], {"16": 1})
            self.assertEqual((f["wild"], f["trn"], f["wo"], f["lvl"], f["seen"], f["own"]), (1, 1, 1, 9, 3, 2))
            self.assertEqual(f["first"][0][0], "New Bark Town")
            # An episode reset re-baselines the dex rather than counting the start state as new catches.
            feed.observe(infos=[_info(own=(155,))], actions=np.array([0]), rewards=np.array([0.0]), dones=np.array([True]))
            feed.observe(infos=[_info(own=(155,))], actions=np.array([0]), rewards=np.array([0.0]), dones=np.array([False]))
            self.assertEqual(feed.fun_payload()["got"], {"16": 1})
            feed.on_update(0, 10, {"update": 0})
            rec = json.loads((Path(tmp) / "h" / "history.jsonl").read_text().splitlines()[0])
            self.assertEqual(rec["fun"]["wild"], 1)
            meta = json.loads((Path(tmp) / "h" / "meta.json").read_text())
            self.assertIn("16", meta["species_names"])

    def test_party_payload_picks_strongest_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = LiveFeed(Path(tmp) / "live", num_envs=2, action_names=["a"])
            a, b = _info(), _info()
            a["fun"].update(ps=[155, 0xFF], pl=[6], ph=[[10, 20]])
            b["fun"].update(ps=[155, 16, 19, 0, 0, 0], pl=[12, 8, 3], ph=[[30, 40], [0, 0], [5, 5]])
            feed.observe(infos=[a, b], actions=np.array([0, 0]), rewards=np.array([0.0, 0.0]), dones=np.array([False, False]))
            f = feed.fun_payload()
            self.assertEqual(f["pe"], 1)
            self.assertEqual(f["party"], [[155, 12, 30, 40], [16, 8], [19, 3, 5, 5]])
            self.assertIn("16", feed._species_names())
            self.assertEqual(json.loads(json.dumps(f))["party"][1], [16, 8])

    def test_sprite_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Path(tmp) / "live", Path(tmp) / "m.jsonl", Path(tmp)))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                r = urllib.request.urlopen(base + "/sprites/25.png", timeout=5)
                self.assertEqual(r.headers["Content-Type"], "image/png")
                self.assertEqual(r.read(4), b"\x89PNG")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(base + "/sprites/999.png", timeout=5)
                self.assertEqual(ctx.exception.code, 404)
                page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
                for token in ("partyHtml", "/sprites/", "image-rendering: pixelated"):
                    self.assertIn(token, page)
            finally:
                server.shutdown()
                server.server_close()

    def test_no_fun_without_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = LiveFeed(Path(tmp) / "live", num_envs=1, action_names=["a"])
            feed.observe(infos=[{"map": [1, 1], "x": 0, "y": 0}], actions=np.array([0]), rewards=np.array([0.0]), dones=np.array([False]))
            self.assertIsNone(feed.fun_payload())
            self.assertIsNone(feed.snapshot()["fun"])

    def test_page_wiring_and_history_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            (runs / "r1").mkdir(parents=True)
            (runs / "r1" / "meta.json").write_text(json.dumps({"run_id": "r1", "species_names": {"16": "Pidgey"}}))
            (runs / "r1" / "history.jsonl").write_text(json.dumps({"update": 0, "fun": {"enc": {"16": 2}}}) + "\n")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Path(tmp) / "live", Path(tmp) / "m.jsonl", runs))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
                for token in ('id="funCard"', "renderFun", "species_names"):
                    self.assertIn(token, page)
                h = json.loads(urllib.request.urlopen(base + "/api/run/r1/history", timeout=5).read())
                self.assertEqual(h["records"][0]["fun"]["enc"], {"16": 2})
                self.assertEqual(h["meta"]["species_names"]["16"], "Pidgey")
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
