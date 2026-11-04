"""Fastest-to-milestone records (src.milestone_stats), their persistence (LiveFeed state/history), the dashboard event
derivation (tools.dashboard_events), its endpoints and the page wiring."""

import json
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from src.config import ROM_PATH
from tools import train_deep_ppo as t
from src.live_feed import LiveFeed
from src.milestone_stats import MilestoneRecords, is_base_start
from tools.dashboard import make_handler
from tools.dashboard_events import derive, run_events

PAGE = Path(__file__).resolve().parents[1] / "tools" / "dashboard_static" / "index.html"
ACTIONS = ("up", "down", "left", "right", "a")


class RecordLogicTests(unittest.TestCase):
    def test_base_start_detection(self):
        self.assertTrue(is_base_start("new_bark_outside"))
        self.assertTrue(is_base_start("states/new_bark_outside.state"))
        for name in ("swarm:stage3_0", "battle:x", "route:b12", "route29_entry", "", None):
            self.assertFalse(is_base_start(name))

    def test_first_reach_then_faster_then_slower_ignored(self):
        r = MilestoneRecords()
        ev = r.observe(env=3, m=1, steps=740, update=5, from_start=True)
        self.assertEqual([(e["scope"], e["first"], e["prev"]) for e in ev], [("start", True, None), ("any", True, None)])
        ev = r.observe(env=4, m=1, steps=612, update=6, from_start=True)
        self.assertEqual(ev[0]["prev"], 740)
        self.assertEqual(r.start[1], {"steps": 612, "update": 6, "env": 4, "prev": 740, "delta": -128})
        self.assertEqual(r.observe(env=1, m=1, steps=700, update=7, from_start=True), [])   # slower: ignored
        self.assertEqual(r.observe(env=1, m=1, steps=612, update=7, from_start=True), [])   # equal: not a record
        self.assertEqual(r.start[1]["steps"], 612)

    def test_archive_start_episodes_only_count_for_any_start(self):
        r = MilestoneRecords()
        ev = r.observe(env=0, m=2, steps=40, update=1, from_start=False)    # swarm / route-checkpoint start
        self.assertEqual([e["scope"] for e in ev], ["any"])
        self.assertIsNone(r.start[2])
        r.observe(env=0, m=2, steps=900, update=2, from_start=True)
        self.assertEqual(r.start[2]["steps"], 900)        # not beaten by the fake 40
        self.assertEqual(r.any[2]["steps"], 40)

    def test_stagger_partial_episode_excluded(self):
        r = MilestoneRecords()
        self.assertEqual(r.observe(env=0, m=0, steps=10, update=0, from_start=True, partial=True), [])
        self.assertIsNone(r.start[0])
        self.assertIsNone(r.any[0])

    def test_log_is_bounded_and_take_new_drains(self):
        r = MilestoneRecords(log_max=5)
        for k in range(20):
            r.observe(env=0, m=0, steps=1000 - k, update=k, from_start=True)
        self.assertEqual(len(r.log), 5)
        self.assertEqual(len(r.take_new()), 40)           # 20 start + 20 any events since the last take
        self.assertEqual(r.take_new(), [])

    def test_rebuild_from_history_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Path(tmp) / "history.jsonl"
            lines = [{"update": 1, "metrics": {}}, {"update": 2, "records": [
                {"u": 2, "m": 1, "steps": 700, "prev": None, "env": 1, "scope": "start", "first": True},
                {"u": 2, "m": 1, "steps": 700, "prev": None, "env": 1, "scope": "any", "first": True}]},
                {"update": 3, "records": [{"u": 3, "m": 1, "steps": 650, "prev": 700, "env": 2, "scope": "start", "first": False}]}]
            h.write_text("\n".join(json.dumps(x) for x in lines) + "\n{partial", encoding="utf-8")
            r = MilestoneRecords.from_history(h)
            self.assertEqual(r.start[1]["steps"], 650)
            self.assertEqual(r.start[1]["prev"], 700)
            self.assertEqual(r.any[1]["steps"], 700)
            self.assertEqual(r.take_new(), [])
            self.assertEqual(MilestoneRecords.from_history(Path(tmp) / "nope.jsonl").start, [None] * 8)


@unittest.skipUnless(Path(ROM_PATH).exists(), "needs the Pokemon Crystal ROM")
class EnvHookTests(unittest.TestCase):
    """EnvWorker.step reports the milestones first reached in the episode as info['ms_new'] = [[m, steps, from_base_start]]."""

    def _worker(self, **cfg):
        w = t.EnvWorker(0, ["new_bark_outside"], max_steps=500, cfg=cfg)
        self.addCleanup(w.close)
        w.reset()
        return w

    def _steps(self, w, n_before, stage):
        infos = []
        for _ in range(n_before):
            *_, info = w.step(0)
            infos.append(info)
        with mock.patch.object(t, "current_stage", return_value=stage):
            *_, info = w.step(0)
        infos.append(info)
        return infos

    def test_reports_first_arrival_with_step_count_from_base_start(self):
        w = self._worker()
        infos = self._steps(w, 3, 1)
        self.assertTrue(all("ms_new" not in i for i in infos[:3]))
        self.assertEqual(infos[3]["ms_new"], [[0, 4, True]])
        with mock.patch.object(t, "current_stage", return_value=1):
            *_, again = w.step(0)
        self.assertNotIn("ms_new", again)                      # already reached in this episode

    def test_a_jump_over_two_stages_reports_both(self):
        w = self._worker()
        self.assertEqual(self._steps(w, 1, 2)[-1]["ms_new"], [[0, 2, True], [1, 2, True]])

    def test_archive_start_is_not_a_base_start(self):
        w = self._worker()
        w.state_name = "route:b3"
        self.assertEqual(self._steps(w, 1, 1)[-1]["ms_new"], [[0, 2, False]])

    def test_stagger_partial_first_episode_reports_nothing(self):
        w = t.EnvWorker(0, ["new_bark_outside"], max_steps=500, cfg={"stagger_episodes": True, "num_envs": 4})
        self.addCleanup(w.close)
        w.reset()
        w._partial = True
        self.assertNotIn("ms_new", self._steps(w, 1, 1)[-1])


class PersistenceTests(unittest.TestCase):
    def test_state_and_history_carry_records_only_when_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            live, hist = Path(tmp) / "live", Path(tmp) / "runs" / "r1"
            feed = LiveFeed(live, num_envs=2, action_names=ACTIONS, history_dir=hist, state_interval=0.0, frame_every=1000)
            rec = MilestoneRecords()
            for e in rec.observe(env=1, m=1, steps=740, update=0, from_start=True):
                feed.record_milestone(e, rec.snapshot())
            feed.set_records(rec.snapshot(), rec.take_new())
            feed.on_update(0, 10, metrics={"update": 0})
            for e in rec.observe(env=0, m=1, steps=612, update=1, from_start=True):
                feed.record_milestone(e, rec.snapshot())
            feed.set_records(rec.snapshot(), rec.take_new())
            feed.on_update(1, 10, metrics={"update": 1})
            feed.set_records(rec.snapshot(), rec.take_new())      # update 2: nothing new
            feed.on_update(2, 10, metrics={"update": 2})
            lines = [json.loads(x) for x in (hist / "history.jsonl").read_text().splitlines()]
            self.assertEqual([("records" in x) for x in lines], [True, True, False])
            self.assertEqual({e["scope"] for e in lines[0]["records"]}, {"start", "any"})
            state = json.loads((live / "state.json").read_text())
            self.assertEqual(state["records"]["start"][1]["steps"], 612)
            self.assertEqual(state["records"]["start"][1]["delta"], -128)
            texts = [e["text"] for e in state["events"] if e["kind"] == "milestone"]
            self.assertEqual(texts, ["First time reaching Cherrygrove: 740 steps (env 1)",
                                     "New record: Cherrygrove in 612 steps (was 740, -128)"])


def _hist(n=40):
    recs = []
    for u in range(n):
        m = {"update": u, "mean_reward": -1.0 + (0.05 if u % 2 else -0.05), "entropy": 1.5, "value_loss": 0.01, "maps_discovered": 3 + (u >= 20),
             "path_best_max_cum": 10 + (30 if u >= 25 else 0), "ms_reach_cum": [0.5 if u >= 10 else 0] + [0] * 7,
             "battle_a_bias": 0.3 if u < 15 else 0.0, "entropy_boost": 1.0 if 30 <= u < 33 else 0.0, "route_ckpt_size": 4 if u >= 12 else 0,
             "intrinsic_weight": 1.0 - u * 0.02, "whiteout_episodes": 6 if u == 30 else 1}
        if u == 30:
            m["mean_reward"] = 4.0
        rec = {"update": u, "metrics": m, "fun": {"first": [["New Bark", 1], ["Route 29", 2]] + ([["Cherrygrove", 3]] if u >= 20 else []), "lvl": 5 + (u >= 22), "own": 1, "seen": 2},
               "components": {"pos": {"path": 5.0 + (20 if u == 30 else 0)}, "neg": {"step": -4.0, "whiteout": -15.0 if u == 30 else 0.0}}}
        if u == 31:
            rec["records"] = [{"u": 31, "m": 1, "name": "cherrygrove", "steps": 612, "prev": 740, "env": 2, "scope": "start", "first": False}]
        recs.append(rec)
    return recs


class EventDerivationTests(unittest.TestCase):
    def test_event_types_from_old_style_data(self):
        out = derive(_hist())
        types = {e["t"] for e in out["events"]}
        for t in ("ms", "path", "map", "wo", "abias", "boost", "intr", "lvl", "rck", "spike", "rec"):
            self.assertIn(t, types)
        by = lambda t: [e for e in out["events"] if e["t"] == t]
        self.assertEqual(by("ms")[0]["u"], 10)
        self.assertIn("Route 29", by("ms")[0]["x"])
        self.assertEqual(by("abias")[0]["u"], 15)
        self.assertEqual([e["u"] for e in by("boost")], [30, 33])
        self.assertIn("Cherrygrove", by("map")[0]["x"])
        self.assertTrue(by("path")[-1]["x"].endswith("40/309 tiles"))
        self.assertEqual(len(out["comps"]), 40)

    def test_spike_gets_likely_cause_and_component_deltas(self):
        out = derive(_hist())
        sp = [e for e in out["events"] if e["t"] == "spike" and e["u"] == 30 and e["v"]["metric"] == "mean_reward"]
        self.assertTrue(sp)
        self.assertTrue(any("white-out wave" in x.lower() for x in sp[0]["lk"]))      # a wave, not the routine 1 per update
        self.assertEqual(sp[0]["top"][0][0], "whiteout")
        self.assertTrue(any(k == "path" for k, _ in sp[0]["top"]))

    def test_missing_fields_degrade_quietly(self):
        out = derive([{"update": u, "metrics": {"update": u, "mean_reward": 0.1}} for u in range(20)])
        self.assertEqual(out["events"], [])
        self.assertEqual(derive([])["events"], [])

    def test_run_events_reads_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "history.jsonl").write_text("\n".join(json.dumps(r) for r in _hist()) + "\n{broken", encoding="utf-8")
            a = run_events(Path(tmp))
            self.assertEqual(a["n"], 40)
            self.assertIs(run_events(Path(tmp)), a)
            self.assertEqual(run_events(Path(tmp) / "none")["events"], [])


class EndpointTests(unittest.TestCase):
    def test_events_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "r9"
            run.mkdir(parents=True)
            (run / "history.jsonl").write_text("\n".join(json.dumps(r) for r in _hist()) + "\n", encoding="utf-8")
            live = root / "live"
            live.mkdir()
            (live / "state.json").write_text(json.dumps({"run_id": "r9", "t": 1}))
            (root / "m.jsonl").write_text("")
            srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(live, root / "m.jsonl", root / "runs"))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                base = f"http://127.0.0.1:{srv.server_address[1]}"
                for path in ("/api/events", "/api/run/r9/events"):
                    d = json.loads(urllib.request.urlopen(base + path).read())
                    self.assertEqual(d["run_id"], "r9")
                    self.assertTrue(d["events"])
                    self.assertEqual(len(d["comps"]), 40)
                with self.assertRaises(Exception):
                    urllib.request.urlopen(base + "/api/run/nope/events")
            finally:
                srv.shutdown()


class PageWiringTests(unittest.TestCase):
    def setUp(self):
        self.html = PAGE.read_text(encoding="utf-8")

    def test_ids_and_hooks(self):
        for ident in ("recCard", "recBody", "recSub", "toasts", "evToggle", "evLegend", "whyBtn", "whyPanel",
                      "evl_reward", "evl_entropy", "evl_kl", "evl_loss"):
            self.assertIn(f'id="{ident}"', self.html, ident)
        self.assertRegex(self.html, r'id="toasts"[^>]*aria-live="polite"')
        for fn in ("function renderRecords", "function recordsAt", "function toastNew", "function placeMarkers", "function renderWhy", "/api/events", "/events`", "tm_ev_on"):
            self.assertIn(fn, self.html, fn)
        self.assertIn("Not recorded for this run", self.html)
        self.assertIn("not reached yet", self.html)
        self.assertIn("resumed from checkpoint", self.html)           # the existing resume lines are kept

    def test_script_parses(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        js = re.findall(r"<script>(.*?)</script>", self.html, re.S)[-1]
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "page.js"
            f.write_text(js, encoding="utf-8")
            res = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
