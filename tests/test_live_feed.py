import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from src.live_feed import LiveFeed
from tools.dashboard import make_handler, read_metrics

ACTIONS = ("up", "down", "left", "right", "a")


def _info(x, y, map_id=(24, 4), **extra):
    return {"map": list(map_id), "x": x, "y": y, "state": "route29_entry", "battle": False, "badges": 0, **extra}


class LiveFeedTests(unittest.TestCase):
    def _feed(self, tmp, n=2):
        return LiveFeed(tmp, num_envs=n, action_names=ACTIONS, state_interval=0.0, frame_interval=0.0)

    def test_heatmap_trails_and_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = self._feed(tmp)
            screens = np.zeros((2, 144, 160, 3), dtype=np.uint8)
            for step in range(3):
                feed.observe(
                    infos=[_info(5 + step, 9), _info(1, 1, map_id=(24, 3))],
                    actions=np.array([3, 0]),
                    rewards=np.array([0.5, -0.1], dtype=np.float32),
                    dones=np.array([False, False]),
                    screens=screens,
                )
            snap = json.loads((Path(tmp) / "state.json").read_text())
            self.assertEqual(snap["maps_seen"], 2)
            self.assertEqual(snap["actions"]["total"], [3, 0, 0, 3, 0])
            self.assertEqual(len(snap["trails"][0]), 3)
            heat = {h["map"]: h for h in snap["heat"]}
            self.assertEqual(heat["24,3"]["cells"], [[1, 1, 3]])
            self.assertEqual(sorted(c[:2] for c in heat["24,4"]["cells"]), [[5, 9], [6, 9], [7, 9]])
            self.assertTrue((Path(tmp) / "env_0.png").is_file())
            self.assertAlmostEqual(snap["envs"][0]["ep_return"], 1.5)

    def test_episode_finish_resets_and_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = self._feed(tmp, n=1)
            feed.observe(infos=[_info(1, 1)], actions=np.array([0]), rewards=np.array([2.0]), dones=np.array([False]))
            feed.observe(
                infos=[_info(2, 2, ended_maps=3, ended_state="route29_entry")],
                actions=np.array([1]),
                rewards=np.array([1.0]),
                dones=np.array([True]),
            )
            snap = feed.snapshot()
            self.assertEqual(len(snap["episodes"]), 1)
            self.assertAlmostEqual(snap["episodes"][0]["return"], 3.0)
            self.assertEqual(snap["episodes"][0]["maps"], 3)
            self.assertEqual(snap["envs"][0]["ep_return"], 0.0)
            self.assertEqual(snap["best_maps"], 3)
            kinds = {e["kind"] for e in snap["events"]}
            self.assertIn("map", kinds)
            self.assertIn("record", kinds)

    def test_on_update_clears_recent_actions_and_finish_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = self._feed(tmp, n=1)
            feed.observe(infos=None, actions=np.array([2]), rewards=np.array([0.0]), dones=np.array([False]))
            feed.on_update(4, 128)
            snap = feed.snapshot()
            self.assertEqual(snap["actions"]["recent"], [0, 0, 0, 0, 0])
            self.assertEqual(snap["actions"]["total"][2], 1)
            self.assertEqual((snap["update"], snap["samples"]), (4, 128))
            feed.finish()
            self.assertTrue(json.loads((Path(tmp) / "state.json").read_text())["finished"])


class RunHistoryTests(unittest.TestCase):
    def _run(self, tmp, updates=3):
        feed = LiveFeed(
            Path(tmp) / "live",
            num_envs=2,
            action_names=ACTIONS,
            state_interval=0.0,
            history_dir=Path(tmp) / "runs" / "r1",
            frame_every=2,
        )
        screens = np.full((2, 144, 160, 3), 90, dtype=np.uint8)
        for u in range(updates):
            for s in range(2):
                feed.observe(
                    infos=[_info(u + s, 5), _info(1, 1, map_id=(24, 3))],
                    actions=np.array([3, 0]),
                    rewards=np.array([0.5, 0.5], dtype=np.float32),
                    dones=np.array([False, False]),
                    screens=screens,
                )
            feed.on_update(u, 2, metrics={"update": u, "mean_reward": float(u)})
        feed.finish()
        return Path(tmp) / "runs" / "r1"

    def test_history_records_per_update_deltas_and_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp)
            records = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines()]
            self.assertEqual([r["update"] for r in records], [0, 1, 2])
            # heat is a per-update delta, not cumulative: 2 steps x 2 envs per update
            self.assertEqual(sum(c for cells in records[0]["heat"].values() for _, _, c in cells), 4)
            self.assertEqual(records[1]["metrics"]["mean_reward"], 1.0)
            self.assertEqual(len(records[1]["envs"]), 2)
            frames = sorted(p.name for p in (run / "frames").glob("*.jpg"))
            self.assertEqual(frames, ["u000000.jpg", "u000002.jpg"])  # every 2 updates + final
            meta = json.loads((run / "meta.json").read_text())
            self.assertTrue(meta["finished"])
            self.assertEqual(meta["map_names"]["24,4"], "New Bark Town")

    def test_no_history_dir_means_no_history_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = LiveFeed(Path(tmp) / "live", num_envs=1, action_names=ACTIONS, state_interval=0.0)
            feed.observe(infos=None, actions=np.array([0]), rewards=np.array([0.0]), dones=np.array([False]))
            feed.on_update(0, 1, metrics={"update": 0})
            feed.finish()
            self.assertFalse((Path(tmp) / "runs").exists())

    def test_runs_api_history_frames_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(Path(tmp) / "live", Path(tmp) / "m.jsonl", Path(tmp) / "runs")
            )
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                get = lambda path: urllib.request.urlopen(base + path, timeout=5).read()
                runs = json.loads(get("/api/runs"))
                self.assertEqual([r["id"] for r in runs], ["r1"])
                self.assertTrue(runs[0]["finished"])
                hist = json.loads(get("/api/run/r1/history"))
                self.assertEqual(len(hist["records"]), 3)
                self.assertEqual(hist["frames"], [0, 2])
                self.assertEqual(get("/runframes/r1/u000000.jpg")[:2], b"\xff\xd8")  # JPEG magic
                for bad in ("/api/run/..%2f..%2fetc/history", "/api/run/nope/history", "/runframes/r1/evil.jpg"):
                    with self.assertRaises(urllib.error.HTTPError):
                        get(bad)
            finally:
                server.shutdown()
                server.server_close()


class CloudSyncTests(unittest.TestCase):
    def test_pull_commands_cover_live_runs_and_metrics(self):
        from tools.cloud_train_deep import LOG_PULLS, build_pull_cmd

        self.assertEqual({r for r, _ in LOG_PULLS}, {"logs/live/", "logs/runs/", "logs/training_metrics.jsonl"})
        cmd = build_pull_cmd("ssh -p 1", "1.2.3.4", "logs/runs/", "logs/runs/")
        self.assertEqual(cmd[:4], ["rsync", "-az", "-e", "ssh -p 1"])
        self.assertEqual(cmd[4], "root@1.2.3.4:/workspace/tensormew/logs/runs/")
        self.assertEqual(cmd[5], "./logs/runs/")


class DashboardServerTests(unittest.TestCase):
    def test_read_metrics_filters_step_records_and_since(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.jsonl"
            lines = [
                {"type": "step", "step": 1, "update": 99},
                {"update": 0, "mean_reward": 1.0},
                "not json",
                {"update": 1, "mean_reward": 2.0},
            ]
            p.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines), encoding="utf-8")
            self.assertEqual([r["update"] for r in read_metrics(p)], [0, 1])
            self.assertEqual([r["update"] for r in read_metrics(p, since=0)], [1])
            self.assertEqual(read_metrics(Path(tmp) / "missing.jsonl"), [])

    def test_http_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = Path(tmp) / "live"
            live.mkdir()
            (live / "state.json").write_text('{"t":1}')
            (live / "env_0.png").write_bytes(b"\x89PNG")
            metrics = Path(tmp) / "m.jsonl"
            metrics.write_text(json.dumps({"update": 0, "mean_reward": 1.0}) + "\n")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(live, metrics))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                get = lambda path: urllib.request.urlopen(base + path, timeout=5).read()
                self.assertIn(b"TensorMew", get("/"))
                self.assertEqual(json.loads(get("/api/live")), {"t": 1})
                self.assertEqual(json.loads(get("/api/metrics"))[0]["update"], 0)
                self.assertEqual(get("/frames/env_0.png"), b"\x89PNG")
                with self.assertRaises(urllib.error.HTTPError):
                    get("/frames/env_9.png")
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
