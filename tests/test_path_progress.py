"""long_v4: the dense best-ever path-progress reward (src.path_progress) on the real ROM route graph."""

import argparse
import json
import re
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from src import path_progress as pp
from src import route_graph as rg
from src.config import ROM_PATH
from src.rl_env import ACTIONS
from src.train_config import PRESETS, ConfigError, apply_overrides, from_dict, load_config
from tools import cloud_train_deep as c
from tools import train_deep_ppo as t

HAVE_ROM = Path(ROM_PATH).exists()
SCALE = 0.08


def st(node, fainted=False):
    mid, x, y = node
    return SimpleNamespace(map_group=mid[0], map_number=mid[1], x=x, y=y, fainted=fainted)


class PathFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = rg._graph()
        if cls.g is None:
            raise unittest.SkipTest("ROM not available")
        cls.d = cls.g.dist[0]
        n = rg.PATH_START
        cls.path = [n]
        while cls.d[n] > 0:
            n = [o for o in cls.g.fwd[n] if o is not None and cls.d.get(o, 1e9) == cls.d[n] - 1][0]
            cls.path.append(n)

    def run_walk(self, nodes, scale=SCALE):
        tr = pp.PathProgress(scale)
        tr.reset(st(nodes[0]))
        paid = [tr.update(st(a), st(b)) for a, b in zip(nodes, nodes[1:])]
        return tr, paid

    def bfs(self, a, b):
        prev, q = {a: None}, deque([a])
        while q:
            n = q.popleft()
            if n == b:
                break
            for o in self.g.fwd.get(n, ()):
                if o is not None and o not in prev:
                    prev[o] = n
                    q.append(o)
        out, n = [], b
        while n is not None:
            out.append(n)
            n = prev[n]
        return out[::-1]


class ProgressMeasureTests(PathFixture):
    def test_zero_at_start_and_309_in_front_of_falkner(self):
        from src.story_path import STORY_PATH_STEPS
        self.assertEqual(STORY_PATH_STEPS, 309)
        self.assertEqual(rg.path_total(), float(STORY_PATH_STEPS))
        self.assertEqual(pp.progress(st(rg.PATH_START)), 0.0)
        self.assertEqual(pp.progress(st(self.path[-1])), 309.0)
        self.assertEqual(len(self.path), 310)

    def test_rises_by_exactly_one_per_step_along_the_whole_real_path(self):
        values = [pp.progress(st(n)) for n in self.path]
        self.assertEqual(values, [float(i) for i in range(310)])

    def test_path_passes_every_story_milestone_map_in_order(self):
        from src.story_path import MILESTONES
        maps = []
        for n in self.path:
            if n[0] not in maps:
                maps.append(n[0])
        self.assertEqual([m for m in maps if m in {x.map_id for x in MILESTONES}], [x.map_id for x in MILESTONES if x.map_id])

    def test_unknown_positions_have_no_progress(self):
        self.assertIsNone(pp.progress(st(((1, 1), 3, 3))))

    def test_buildings_off_the_path_never_score_above_their_door(self):
        # every warp into a spur map: the interior tiles are at most the door's progress minus the walk out
        for (mid, x, y), (dest, dx, dy) in self.g.warp_at.items():
            if dest in rg.ROUTE_MAPS:
                continue
            door = pp.progress(st((mid, x, y)))
            inside = pp.progress(st((dest, dx, dy)))
            if door is not None and inside is not None:
                self.assertLessEqual(inside, door + 1, (mid, x, y, dest))


class RewardTests(PathFixture):
    def test_full_path_pays_scale_times_309(self):
        tr, paid = self.run_walk(self.path)
        self.assertAlmostEqual(sum(paid), SCALE * 309, places=6)
        self.assertTrue(all(abs(p - SCALE) < 1e-9 for p in paid))
        self.assertEqual(tr.gained, 309.0)

    def test_stepping_back_pays_nothing_and_forward_again_pays_nothing_until_a_new_best(self):
        p = self.path
        walk = p[:21] + p[19::-1][:5] + p[16:26]    # forward to 20, back to 16, forward again past 20 to 25
        _, paid = self.run_walk(walk)
        self.assertAlmostEqual(sum(paid[:20]), SCALE * 20)
        back_and_forward = paid[20:25 + 4]
        self.assertTrue(all(v == 0.0 for v in back_and_forward), back_and_forward)
        self.assertAlmostEqual(sum(paid[29:]), SCALE * 5)   # 21..25 are new bests, one tile each
        self.assertAlmostEqual(sum(paid), SCALE * 25)

    def test_oscillating_forever_cannot_farm(self):
        p = self.path
        walk = p[:11] + (p[10:8:-1] + p[9:11]) * 50
        _, paid = self.run_walk(walk)
        self.assertAlmostEqual(sum(paid), SCALE * 10)

    def test_dead_end_corridor_on_route_29_pays_nothing_beyond_the_junction(self):
        onpath = set(self.path)
        best = None
        for e in self.d:
            if e[0] != (24, 3) or e in onpath or sum(1 for o in self.g.fwd[e] if o is not None and o != e) != 1:
                continue
            n, chain = e, [e]
            while n not in onpath:
                n = [o for o in self.g.fwd[n] if o is not None and self.d.get(o, 1e9) == self.d[n] - 1][0]
                chain.append(n)
            walk_in = self.bfs(n, e)
            if len(walk_in) >= 4 and max(pp.progress(st(x)) for x in walk_in) <= pp.progress(st(n)):
                if best is None or len(walk_in) > len(best[2]):
                    best = (n, e, walk_in)
        self.assertIsNotNone(best, "no dead-end corridor found on Route 29")
        junction, end, walk_in = best
        i = self.path.index(junction)
        walk = self.path[:i + 1] + walk_in[1:] + walk_in[-2::-1]   # to the junction, down the corridor, and back
        _, paid = self.run_walk(walk)
        self.assertGreaterEqual(len(walk_in), 4)
        self.assertAlmostEqual(sum(paid), SCALE * i)           # exactly the junction's progress, nothing for the corridor
        self.assertEqual(sum(paid[i:]), 0.0)
        self.assertLess(pp.progress(st(end)), pp.progress(st(junction)))

    def test_start_progress_is_free_for_a_mid_route_archive_state(self):
        p = self.path
        tr = pp.PathProgress(SCALE)
        tr.reset(st(p[150]))
        self.assertEqual(tr.best, 150.0)
        self.assertEqual(tr.update(st(p[150]), st(p[150])), 0.0)
        self.assertEqual(tr.update(st(p[150]), st(p[149])), 0.0)
        self.assertAlmostEqual(tr.update(st(p[149]), st(p[151])), SCALE)    # only beyond the start's own best
        total = sum(tr.update(st(a), st(b)) for a, b in zip(p[151:], p[152:]))
        self.assertAlmostEqual(total + SCALE, SCALE * (309 - 150))

    def test_off_graph_start_pays_nothing_for_the_first_known_tile(self):
        tr = pp.PathProgress(SCALE)
        tr.reset(st(((1, 1), 3, 3)))
        self.assertIsNone(tr.best)
        self.assertEqual(tr.update(st(((1, 1), 3, 3)), st(self.path[100])), 0.0)
        self.assertAlmostEqual(tr.update(st(self.path[100]), st(self.path[101])), SCALE)

    def test_whiteout_teleport_neither_pays_nor_moves_the_best(self):
        p = self.path
        tr = pp.PathProgress(SCALE)
        tr.reset(st(p[0]))
        self.assertEqual(tr.update(st(p[0]), st(p[200], fainted=True)), 0.0)
        self.assertEqual(tr.best, 0.0)

    def test_scale_zero_and_negative(self):
        _, paid = self.run_walk(self.path[:10], scale=0.0)
        self.assertEqual(sum(paid), 0.0)
        with self.assertRaises(ValueError):
            pp.PathProgress(-1.0)


class ConfigTests(unittest.TestCase):
    def test_long_v3_keeps_path_progress_off(self):
        self.assertIn("long_v3", PRESETS)
        for name in PRESETS:
            m = re.fullmatch(r"long_v(\d+)", name)
            if m is None or int(m.group(1)) < 4:   # long_v4 and every later long_vN turn the path reward on
                self.assertFalse(load_config(name).path_progress, name)
        self.assertIsNone(t.worker_env_cfg(load_config("long_v3"), None)["path_progress"])

    def test_long_v4_is_long_v3_plus_path_progress_and_the_documented_learning_changes(self):
        v3, v4 = load_config("long_v3").to_dict(), load_config("long_v4").to_dict()
        self.assertEqual(v4["name"], "long_v4")
        self.assertTrue(v4["path_progress"])
        self.assertAlmostEqual(v4["path_progress_scale"], 0.08)
        # the learning-speed changes (tests/test_long_v4_learning.py) are the only other differences
        changed = {"name", "path_progress", "entropy_start", "entropy_end", "explore_budget", "intrinsic_weight_end",
                   "intrinsic_decay_updates", "heal_reward", "battle_lowhp_weight", "aux_loss"}
        v3p, v4p = dict(v3.pop("ppo")), dict(v4.pop("ppo"))
        self.assertEqual({k for k in v3p if v3p[k] != v4p[k]}, {"gamma"})
        for k in v3:
            if k not in changed:
                self.assertEqual(v3[k], v4[k], k)
        self.assertFalse(v4["goal_potential"])
        self.assertTrue(v4["dwell_penalty"])

    def test_worker_env_cfg_and_overrides(self):
        self.assertAlmostEqual(t.worker_env_cfg(load_config("long_v4"), None)["path_progress"], 0.08)
        off = apply_overrides(load_config("long_v4"), {"path_progress": False})
        self.assertIsNone(t.worker_env_cfg(off, None)["path_progress"])
        big = apply_overrides(load_config("long_v4"), {"path_progress_scale": 0.2})
        self.assertAlmostEqual(t.worker_env_cfg(big, None)["path_progress"], 0.2)

    def test_validation(self):
        with self.assertRaises(ConfigError):
            apply_overrides(load_config("long_v4"), {"path_progress_scale": -0.1})
        with self.assertRaises(ConfigError):
            apply_overrides(load_config("long_v4"), {"goal_potential": True})

    def test_json_roundtrip_and_preset_key(self):
        cfg = load_config("long_v4")
        again = from_dict(json.loads(cfg.to_json()))
        self.assertTrue(again.path_progress)
        self.assertEqual(again.to_dict(), cfg.to_dict())
        self.assertTrue(from_dict({"preset_unused": None} if False else {"path_progress": True}).path_progress)

    def test_cloud_train_accepts_long_v4(self):
        self.assertEqual(c.resolve_config(argparse.Namespace(config="long_v4")), ("long_v4", False))
        args = argparse.Namespace(updates=1600, rollout=128, max_steps=1024, batch_size=256, lr=3e-4,
                                  state="new_bark_outside", save="s.pt", envs=24, curriculum=False, recurrent=True,
                                  states="", config="long_v4", explicit={"updates", "envs", "state", "curriculum"},
                                  resume=False, run_id="v4-1")
        self.assertIn("--config long_v4", c.build_remote_train_cmd(args, 24, "long_v4"))

    def test_path_progress_module_and_config_import_without_numpy(self):
        import subprocess
        import sys
        code = ("import sys; sys.modules['numpy'] = None; "
                "from src.train_config import PRESETS, load_config; import src.path_progress; "
                "assert load_config('long_v4').path_progress; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)


@unittest.skipUnless(HAVE_ROM, "needs the Pokemon Crystal ROM")
class EnvWorkerPathTests(PathFixture):
    STATE = "new_bark_outside"

    def make(self, **cfg):
        w = t.EnvWorker(0, [self.STATE], max_steps=200, cfg=cfg)
        self.addCleanup(w.close)
        w.reset()
        return w

    def test_real_env_pays_the_path_component_while_walking_the_route(self):
        w = self.make(path_progress=SCALE)
        total, last = 0.0, None
        for a, b in zip(self.path[:40], self.path[1:41]):
            if a[0] == b[0]:
                dx, dy = b[1] - a[1], b[2] - a[2]
                last = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}[(dx, dy)]
            *_, info = w.step(ACTIONS.index(last))
            total += info["reward_components"]["path"]
        self.assertGreater(total, 0.0)
        self.assertAlmostEqual(total, SCALE * w.path_progress.gained, places=6)
        self.assertAlmostEqual(w.path_progress.gained, w.path_progress.best)   # the start state's progress is 0
        w.reset()
        self.assertEqual(w.path_progress.gained, 0.0)

    def test_pressing_b_pays_no_path_reward_and_without_the_flag_there_is_no_component(self):
        w = self.make(path_progress=SCALE)
        *_, info = w.step(ACTIONS.index("b"))
        self.assertEqual(info["reward_components"]["path"], 0.0)
        w3 = self.make()
        *_, info = w3.step(ACTIONS.index("b"))
        self.assertNotIn("path", info["reward_components"])

    def test_a_new_best_tile_counts_as_progress_for_the_dwell_tracker(self):
        dwell = {"window": 8, "max_tiles": 2, "min_penalty": 0.01, "max_penalty": 0.02, "ramp_steps": 4,
                 "budget": 15.0, "grind_grace": 0, "level_targets": [0] * 9}
        w = self.make(path_progress=SCALE, dwell=dwell)
        w.dwell.overworld = 5
        w.dwell.last_progress = 0
        a, b = self.path[0], self.path[1]
        dx, dy = b[1] - a[1], b[2] - a[2]
        name = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}[(dx, dy)]
        *_, info = w.step(ACTIONS.index(name))
        if info["reward_components"]["path"] > 0:
            self.assertEqual(w.dwell.last_progress, w.dwell.overworld)


if __name__ == "__main__":
    unittest.main()
