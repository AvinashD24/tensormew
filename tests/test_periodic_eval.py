"""Periodic from-start evaluation (flag ``periodic_eval``): bookkeeping, isolation from the training state, base start only,
output fields, flag off = no change, state.json / dashboard wiring."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import periodic_eval as pe
from src import train_config

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "tools" / "dashboard_static" / "index.html"

EVAL_KEYS = ("eval_update", "eval_episodes", "eval_steps", "eval_best_progress_mean", "eval_best_progress_max", "eval_ms_reach",
             "eval_ms_steps", "eval_whiteout_frac", "eval_return_mean", "eval_seconds")


class BookkeepingTests(unittest.TestCase):
    def test_schedule(self):
        hits = [u for u in range(200) if pe.is_eval_update(u, 50)]
        self.assertEqual(hits, [49, 99, 149, 199])
        self.assertFalse(any(pe.is_eval_update(u, 1, enabled=False) for u in range(10)))   # flag off: never
        self.assertTrue(all(pe.is_eval_update(u, 1) for u in range(5)))

    def test_aggregate(self):
        eps = [pe.episode_record(10.0, 1024, 2, 30.5, None, {0: 200, 1: 500}),
               pe.episode_record(-4.0, 300, 0, 4.0, "whiteout", {}),
               pe.episode_record(6.0, 1024, 1, 12.0, None, {0: 400}),
               pe.episode_record(0.0, 1024, 0, 0.0, None, {})]
        out = pe.aggregate(eps, 8, update=49, seconds=12.345)
        for k in EVAL_KEYS:
            self.assertIn(k, out)
        self.assertEqual(out["eval_update"], 49)
        self.assertEqual(out["eval_episodes"], 4)
        self.assertEqual(out["eval_steps"], 1024 * 3 + 300)
        self.assertAlmostEqual(out["eval_best_progress_mean"], (30.5 + 4 + 12) / 4, places=2)
        self.assertEqual(out["eval_best_progress_max"], 30.5)
        self.assertEqual(out["eval_ms_reach"][:3], [0.5, 0.25, 0.0])
        self.assertEqual(len(out["eval_ms_reach"]), 8)
        self.assertEqual(out["eval_ms_steps"][:3], [300.0, 500.0, None])
        self.assertEqual(out["eval_whiteout_frac"], 0.25)
        self.assertEqual(out["eval_return_mean"], 3.0)
        json.dumps(out)   # JSON-able (metrics.jsonl / history.jsonl / state.json)
        with self.assertRaises(ValueError):
            pe.aggregate([], 8, 0)

    def test_base_start_is_the_only_eval_state(self):
        self.assertEqual(pe.EVAL_STATE, "new_bark_outside")

    def test_flag_defaults(self):
        for name in ("blueprint_v1", "long_v3", "long_v6", "long_v7"):
            cfg = train_config.preset(name)
            self.assertFalse(cfg.periodic_eval)
        d = train_config.TrainConfig()
        self.assertEqual((d.periodic_eval, d.eval_every, d.eval_episodes, d.eval_max_steps), (False, 50, 4, 1024))
        self.assertTrue(train_config.preset("long_v8").periodic_eval)


try:
    import torch
    from src.deep_policy import SpatialVisionPolicy
    from src.rl_env import DEEP_OBS_DIM
    from tools.train_deep_ppo import EVAL_WORKER_INDEX, PeriodicEvaluator, eval_env_cfg
    HAVE = (ROOT / "rom" / "pokemon_crystal.gbc").is_file()
except Exception:   # pragma: no cover
    HAVE = False


@unittest.skipUnless(HAVE, "torch / ROM / PyBoy not available")
class IsolationTests(unittest.TestCase):
    def _cfg(self, **kw):
        cfg = train_config.preset("long_v8")
        cfg.eval_episodes, cfg.eval_max_steps = 2, 10
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg.validate()

    def test_env_cfg_has_no_other_start_source(self):
        env = eval_env_cfg(train_config.preset("long_v8"))
        for k in ("swarm_dir", "battle_dir", "route_dir"):
            self.assertIsNone(env[k], k)
        for k in ("swarm_prob", "battle_prob", "route_prob", "jitter_max", "episode_len_jitter"):
            self.assertEqual(env[k], 0, k)
        self.assertFalse(env["stagger_episodes"])
        self.assertFalse(env["stage_balanced"])
        self.assertEqual(env["num_envs"], 0)
        self.assertTrue(env["action_mask"])   # the evaluation samples from the same masked distribution as training

    def test_run_does_not_touch_training_state(self):
        cfg = self._cfg()
        torch.manual_seed(5)
        np.random.seed(5)
        policy = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=6, use_gru=True, gru_hidden_dim=32)
        policy.train()
        policy.set_battle_a_bias(1.5)
        hx = torch.randn(1, 3, 32)
        hx_before = hx.clone()
        params = [p.detach().clone() for p in policy.parameters()]
        bufs = [b.detach().clone() for b in policy.buffers()]
        torch_state, np_state = torch.get_rng_state(), np.random.get_state()
        with tempfile.TemporaryDirectory() as tmp:
            before = sorted(Path(tmp).rglob("*"))
            ev = PeriodicEvaluator(cfg, torch.device("cpu"))
            try:
                out = ev.run(policy, 49)
                # a second evaluation reuses the env and starts from the base state again
                out2 = ev.run(policy, 99)
                self.assertEqual(ev.worker.state_name, pe.EVAL_STATE)
                self.assertIsNone(ev.worker.archive)
                self.assertIsNone(ev.worker.battle_archive)
                self.assertIsNone(ev.worker.route_archive)
                self.assertEqual(ev.worker.index, EVAL_WORKER_INDEX)
            finally:
                ev.close()
            self.assertEqual(sorted(Path(tmp).rglob("*")), before)
        for k in EVAL_KEYS:
            self.assertIn(k, out)
        self.assertEqual((out["eval_update"], out2["eval_update"]), (49, 99))
        self.assertEqual(out["eval_episodes"], 2)
        self.assertEqual(out["eval_steps"], 20)
        self.assertEqual(len(out["eval_ms_reach"]), len(out["eval_ms_steps"]))
        # nothing the training loop owns changed
        for a, b in zip(params, policy.parameters()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(bufs, policy.buffers()):
            self.assertTrue(torch.equal(a, b), "BatchNorm statistics changed")
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()), "global torch RNG consumed")
        s2 = np.random.get_state()
        self.assertTrue(np_state[0] == s2[0] and np.array_equal(np_state[1], s2[1]) and np_state[2:] == s2[2:], "global numpy RNG consumed")
        self.assertTrue(torch.equal(hx, hx_before))
        self.assertEqual(policy.battle_a_bias, 1.5, "A-bias not restored")
        self.assertTrue(policy.training, "train/eval mode not restored")

    def test_same_policy_same_update_is_reproducible(self):
        cfg = self._cfg(eval_episodes=1, eval_max_steps=12)
        torch.manual_seed(2)
        policy = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=6, use_gru=True, gru_hidden_dim=32)
        outs = []
        for _ in range(2):
            ev = PeriodicEvaluator(cfg, torch.device("cpu"))
            try:
                o = ev.run(policy, 7)
            finally:
                ev.close()
            o.pop("eval_seconds")
            outs.append(o)
        self.assertEqual(outs[0], outs[1])


class LiveFeedEvalTests(unittest.TestCase):
    def test_set_eval_reaches_state_json_and_events(self):
        from src.live_feed import LiveFeed
        with tempfile.TemporaryDirectory() as tmp:
            feed = LiveFeed(tmp, num_envs=1, action_names=("up", "down", "left", "right", "a", "b"), state_interval=0.0, frame_interval=0.0)
            self.assertIsNone(feed.snapshot()["eval"])   # older runs / before the first evaluation
            payload = pe.aggregate([pe.episode_record(1.0, 10, 1, 3.0, None, {0: 5})], 8, 49, 2.0)
            feed.set_eval(payload, "From-start evaluation at update 49")
            snap = feed.snapshot()
            self.assertEqual(snap["eval"]["eval_update"], 49)
            self.assertTrue(any(e["kind"] == "eval" for e in snap["events"]))
            json.dumps(snap["eval"])


class DashboardWiringTests(unittest.TestCase):
    def test_page_ids_and_function(self):
        html = PAGE.read_text(encoding="utf-8")
        for el in ("evalCard", "evalH", "evalSub", "evalBody"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn("function renderEval(", html)
        self.assertRegex(html, r"renderSplit\(\); renderEval\(\);")
        self.assertIn("eval_best_progress_mean", html)
        self.assertRegex(html, r'id="evalCard"[^>]*\bhidden\b')   # hidden by default: older runs show nothing

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_script_parses(self):
        html = PAGE.read_text(encoding="utf-8")
        script = html[html.rindex("<script>") + 8:html.rindex("</script>")]
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "page.js"
            f.write_text(script, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_render_eval_with_stub_dom(self):
        """renderEval on old-run data (no eval fields) hides the card; with eval fields it draws the table and chart."""
        html = PAGE.read_text(encoding="utf-8")
        start = html.index("function renderEval(")
        end = html.index("/* ---------- smoothing, reward verdict")
        fn = html[start:end]
        prog = r"""
const els = {};
const $ = (id) => els[id] || (els[id] = { hidden: false, textContent: "", innerHTML: "" });
const okv = (v) => v != null && isFinite(v);
const fmt = (v, d = 2) => (v == null || !isFinite(v)) ? "–" : Number(v).toFixed(d);
const MS_NAMES = ["Route 29", "Cherrygrove", "Route 30", "Route 31", "Violet gate", "Violet City", "Violet Gym", "Falkner beaten"];
let metrics = [{ update: 1, mean_reward: 1 }], live = null;
""" + fn + r"""
renderEval(); if (!$("evalCard").hidden) throw new Error("old run must hide the card");
live = { run: { config: { periodic_eval: true, eval_every: 50 } } }; renderEval();
if ($("evalCard").hidden || !/Waiting for the first evaluation/.test($("evalBody").innerHTML)) throw new Error("planned run shows a waiting note");
metrics = [{ update: 49, eval_update: 49, eval_episodes: 4, eval_best_progress_mean: 12.5, eval_best_progress_max: 30, eval_whiteout_frac: 0.25, eval_return_mean: 3.2, eval_ms_reach: [0.5, 0, null, 0, 0, 0, 0, 0], eval_ms_steps: [300, null, null, null, null, null, null, null] },
           { update: 99, eval_update: 99, eval_episodes: 4, eval_best_progress_mean: 20, eval_best_progress_max: 44, eval_whiteout_frac: 0, eval_return_mean: 5, eval_ms_reach: [1, 0.25, 0, 0, 0, 0, 0, 0], eval_ms_steps: [250, 700, null, null, null, null, null, null] }];
renderEval();
const h = $("evalBody").innerHTML;
if ($("evalCard").hidden || !/<svg/.test(h) || !/Route 29/.test(h) || !/20\.0 \/ 44\.0/.test(h)) throw new Error("table/chart missing: " + h.slice(0, 300));
console.log("ok");
"""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "t.js"
            f.write_text(prog, encoding="utf-8")
            r = subprocess.run(["node", str(f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
