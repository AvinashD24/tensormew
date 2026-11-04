"""long_v4 learning-speed changes: milestone success rates, horizon, auxiliary heads, exploration budget, heal reward,
low-HP battle practice, entropy schedule, preset loading. long_v3 must stay exactly as it was."""

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from src import aux_tasks
from src.config import ROM_PATH
from src.deep_policy import SpatialVisionPolicy
from src.heal_reward import heal_gain, heal_payment, hp_value
from src.live_feed import LiveFeed
from src.milestone_stats import MILESTONE_NAMES, MilestoneStats
from src.ppo_update import EntropySchedule, PPOHParams, RolloutBatch, ppo_update
from src.ram_parser import GameState
from src.rl_env import BUDGET_CAPS, DEEP_OBS_DIM, EXPLORE_BUDGET, N_ACTIONS, budget_pay, progress_reward
from src.story_path import MILESTONE_COUNT
from src.train_config import ConfigError, PRESETS, TrainConfig, apply_overrides, from_dict, load_config
from tools import train_deep_ppo as t

ROOT = Path(__file__).resolve().parents[1]
HAVE_ROM = Path(ROM_PATH).exists()


def _gs(**kw):
    base = {"map_group": 26, "map_number": 5, "x": 3, "y": 9, "tiles": {}, "event_flags": [0] * 300,
            "party_count": 1, "party_species": [155], "party_hp": [[19, 19]], "party_levels": [5],
            "party_moves": [[33, 43, 0, 0]]}
    base.update(kw)
    return GameState.from_dict(base)


def small_policy(aux=True, gru=True, default_dims=False):
    torch.manual_seed(0)
    dims = {} if default_dims else {"ram_hidden_dim": 16, "fusion_dim": 32}   # load() rebuilds the default trunk widths
    return SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=N_ACTIONS, use_gru=gru, gru_hidden_dim=16,
                               aux_heads=aux, **dims)


def make_rams(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    rams = torch.rand(B, T, DEEP_OBS_DIM, generator=g) * 0.2
    rams[..., aux_tasks.IDX_BATTLE] = 0.0
    rams[..., aux_tasks.IDX_MAP_GROUP] = 26 / 32.0
    rams[..., aux_tasks.IDX_MAP_NUMBER] = 3 / 32.0
    rams[..., aux_tasks.IDX_LEAD_HP] = 1.0
    return rams


# ------------------------------------------------------------------------------------ 1. milestone success rates
class MilestoneStatsTests(unittest.TestCase):
    def test_reach_rates_use_only_eligible_episodes(self):
        ms = MilestoneStats()
        ms.add_episode(0, 2, 40.0)           # started at New Bark, got to Cherrygrove (stage 2)
        ms.add_episode(0, 0, 3.0)            # never left
        ms.add_episode(3, 6, 150.0)          # archive start on Route 30, reached Violet City (stage 6)
        out = ms.end_update([0, 2, 6, 8])
        # Route 29 (stage 1): eligible for the first two only (the Route 30 start is past it)
        self.assertEqual(out["ms_elig"][0], 2)
        self.assertEqual(out["ms_reach"][0], 0.5)
        self.assertEqual(out["ms_elig"][3], 3)               # Route 31 (stage 4): all three started before it
        self.assertAlmostEqual(out["ms_reach"][3], 0.333, places=3)
        self.assertEqual(out["ms_reach"][5], round(1 / 3, 3))  # Violet City
        self.assertEqual(out["ms_eps"], 3)
        self.assertEqual(out["path_best_max"], 150.0)
        self.assertAlmostEqual(out["path_best_mean"], round((40 + 3 + 150) / 3, 2))
        self.assertEqual(len(out["ms_reach"]), MILESTONE_COUNT)
        self.assertEqual(len(MILESTONE_NAMES), MILESTONE_COUNT)

    def test_envs_at_or_above_and_window_reset_and_cumulative(self):
        ms = MilestoneStats()
        ms.add_episode(0, 1)
        first = ms.end_update([0, 1, 2, 2])
        self.assertEqual(first["ms_envs_at"][0], 0.75)       # 3 of 4 envs at stage >= 1
        self.assertEqual(first["ms_envs_at"][1], 0.5)        # 2 of 4 at stage >= 2
        second = ms.end_update([0])
        self.assertEqual(second["ms_eps"], 0)
        self.assertIsNone(second["ms_reach"][0])             # empty window: no rate, not 0
        self.assertEqual(second["ms_reach_cum"][0], 1.0)     # the cumulative rate survives
        self.assertEqual(second["ms_eps_cum"], 1)
        self.assertIsNone(second["path_best_mean"])

    def test_partial_and_missing_episodes_are_skipped(self):
        ms = MilestoneStats()
        self.assertFalse(ms.add_episode(0, 3, 10, partial=True))
        self.assertFalse(ms.add_episode(None, None))
        self.assertEqual(ms.end_update([])["ms_eps"], 0)

    def test_payload_is_compact_json(self):
        import json
        ms = MilestoneStats()
        ms.add_episode(0, 4, 100.0)
        s = json.dumps(ms.end_update([4, 4]), separators=(",", ":"))
        self.assertLess(len(s), 700)

    def test_live_feed_state_carries_the_milestones(self):
        with tempfile.TemporaryDirectory() as d:
            feed = LiveFeed(d, num_envs=1, action_names=["up"], state_interval=0)
            self.assertIsNone(feed.snapshot()["milestones"])
            feed.set_milestones({"ms_reach": [0.5]})
            self.assertEqual(feed.snapshot()["milestones"], {"ms_reach": [0.5]})

    def test_worker_info_has_the_episode_summary_fields(self):
        src = (ROOT / "tools" / "train_deep_ppo.py").read_text(encoding="utf-8")
        for key in ("ended_start_stage", "ended_furthest", "ended_best_path", "cur_furthest"):
            self.assertIn(key, src)

    def test_dashboard_has_the_milestone_panel_and_learning_check(self):
        html = (ROOT / "tools" / "dashboard_static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="msCard"', html)
        self.assertIn("function renderMilestones(", html)
        self.assertIn("Reaching further milestones?", html)
        self.assertIn("Array.isArray(metrics[i].ms_reach_cum)", html)   # old runs: panel degrades quietly


# ------------------------------------------------------------------------------------ presets / config
class PresetTests(unittest.TestCase):
    def test_long_v4_values(self):
        cfg = load_config("long_v4")
        self.assertEqual(cfg.name, "long_v4")
        self.assertAlmostEqual(cfg.ppo.gamma, 0.999)
        self.assertAlmostEqual(cfg.ppo.gae_lambda, 0.95)
        self.assertEqual(cfg.rollout_steps, 256)
        self.assertEqual(cfg.max_steps, 2048)
        self.assertAlmostEqual(cfg.entropy_start, 0.01)
        self.assertAlmostEqual(cfg.entropy_end, 0.001)
        self.assertAlmostEqual(cfg.explore_budget, 12.0)
        self.assertAlmostEqual(cfg.intrinsic_weight_end, 0.4)
        self.assertEqual(cfg.intrinsic_decay_updates, 800)
        self.assertTrue(cfg.heal_reward and cfg.aux_loss and cfg.path_progress)
        self.assertAlmostEqual(cfg.battle_lowhp_weight, 3.0)
        self.assertTrue(cfg.battle_archive)

    def test_long_v3_is_exactly_unchanged(self):
        cfg = load_config("long_v3")
        self.assertAlmostEqual(cfg.ppo.gamma, 0.997)
        self.assertAlmostEqual(cfg.ppo.gae_lambda, 0.95)
        self.assertAlmostEqual(cfg.entropy_start, 0.04)
        self.assertAlmostEqual(cfg.entropy_end, 0.01)
        self.assertFalse(cfg.path_progress)
        self.assertFalse(cfg.heal_reward)
        self.assertFalse(cfg.aux_loss)
        self.assertEqual(cfg.battle_lowhp_weight, 0.0)
        self.assertIsNone(cfg.explore_cap_override())          # workers keep the stock budget
        self.assertEqual(cfg.intrinsic_decay_updates, 0)
        self.assertEqual(t.worker_env_cfg(cfg, None)["explore_cap"], None)
        self.assertEqual(t.worker_env_cfg(cfg, None, None)["battle_lowhp_weight"], 0.0)

    def test_explore_default_matches_the_env_constant(self):
        self.assertEqual(TrainConfig().explore_budget, EXPLORE_BUDGET)
        self.assertEqual(BUDGET_CAPS["explore"], EXPLORE_BUDGET)

    def test_json_round_trip_and_old_configs_without_new_keys(self):
        cfg = load_config("long_v4")
        again = from_dict(cfg.to_dict(), TrainConfig())
        self.assertEqual(cfg.to_dict(), again.to_dict())
        old = cfg.to_dict()
        for k in ("explore_budget", "intrinsic_weight_end", "intrinsic_decay_updates", "heal_reward", "aux_loss",
                  "battle_lowhp_weight"):
            old.pop(k)
        self.assertFalse(from_dict(old, TrainConfig()).aux_loss)

    def test_validation(self):
        v4 = load_config("long_v4")
        with self.assertRaises(ConfigError):
            apply_overrides(v4, {"recurrent": False})                  # aux needs the recurrent policy
        with self.assertRaises(ConfigError):
            apply_overrides(v4, {"aux_horizon": 256})
        with self.assertRaises(ConfigError):
            apply_overrides(v4, {"battle_archive": False})             # low-HP weight needs the archive
        with self.assertRaises(ConfigError):
            apply_overrides(v4, {"intrinsic_weight_end": 1.5})
        self.assertEqual(apply_overrides(v4, {"aux_loss": False, "recurrent": False}).aux_loss, False)

    def test_cli_accepts_long_v4(self):
        args = t.build_parser().parse_args(["--config", "long_v4", "--run-id", "v4-1", "--envs", "24", "--updates", "1600",
                                            "--no-curriculum", "--state", "new_bark_outside"])
        cfg = t.build_config(args)
        self.assertEqual(cfg.name, "long_v4")
        self.assertEqual(cfg.envs, 24)
        self.assertEqual(cfg.ppo.num_minibatches, 8)
        self.assertIn("long_v4", PRESETS)

    def test_train_config_imports_without_numpy_or_torch(self):
        code = ("import sys\nfrom src.train_config import PRESETS, load_config\nload_config('long_v4')\n"
                "assert 'numpy' not in sys.modules and 'torch' not in sys.modules, sorted(m for m in sys.modules if m in ('numpy','torch'))")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


# ------------------------------------------------------------------------------------ 2. horizon
class HorizonTests(unittest.TestCase):
    def test_discount_horizon_numbers(self):
        v3, v4 = load_config("long_v3"), load_config("long_v4")
        h3, h4 = 1 / (1 - v3.ppo.gamma), 1 / (1 - v4.ppo.gamma)
        self.assertAlmostEqual(h3, 333.3, places=0)
        self.assertAlmostEqual(h4, 1000.0, places=0)
        self.assertGreater(h4, 319)                                   # longer than the 319-step story path
        self.assertGreater(v4.ppo.gamma ** 1000, 0.3)                 # a milestone 1000 steps ahead still counts
        self.assertLess(v3.ppo.gamma ** 1000, 0.06)

    def test_recurrent_state_carries_across_a_rollout_boundary(self):
        """Evaluating the second half of a rollout from the hidden state at the split equals the full pass."""
        policy = small_policy(aux=False).eval()
        B, T = 1, 6
        torch.manual_seed(1)
        screens = torch.rand(B, T, 3, 144, 160)
        rams = make_rams(B, T)
        actions = torch.zeros(B, T, dtype=torch.long)
        with torch.no_grad():
            _, _, v_full = policy.evaluate_actions_sequence(screens, rams, actions, None)
            hx = None
            for step in range(3):
                _, _, hx = policy.forward_recurrent(screens[:, step], rams[:, step], hx)
            _, _, v_tail = policy.evaluate_actions_sequence(screens[:, 3:], rams[:, 3:], actions[:, 3:], hx)
        self.assertTrue(torch.allclose(v_full[:, 3:], v_tail, atol=1e-4))
        # without the carried state the tail differs: the state really matters
        with torch.no_grad():
            _, _, v_cold = policy.evaluate_actions_sequence(screens[:, 3:], rams[:, 3:], actions[:, 3:], None)
        self.assertFalse(torch.allclose(v_full[:, 3:], v_cold, atol=1e-4))

    def test_trainer_carries_hidden_state_and_trains_on_whole_rollouts(self):
        src = (ROOT / "tools" / "train_deep_ppo.py").read_text(encoding="utf-8")
        self.assertIn("initial_hx = hx.detach().clone()", src)       # rollout k+1 starts from rollout k's state
        self.assertIn("hx = next_hx", src)
        self.assertIn("next_hx[:, done_mask, :] = 0.0", src)          # and resets it at episode ends
        cfg = load_config("long_v4")
        self.assertEqual(cfg.rollout_steps, 256)                      # BPTT sequence = whole rollout (256 steps)


# ------------------------------------------------------------------------------------ 3. auxiliary heads
class AuxTargetTests(unittest.TestCase):
    def test_map_hp_and_battle_targets_and_masks(self):
        B, T, k = 2, 10, 3
        rams = make_rams(B, T)
        rams[0, 5:, aux_tasks.IDX_MAP_GROUP] = 24 / 32.0              # env 0 changes map at step 5
        rams[0, 5:, aux_tasks.IDX_MAP_NUMBER] = 4 / 32.0
        rams[1, 4, aux_tasks.IDX_LEAD_HP] = 0.5                       # env 1 lead HP drops to 50 % at step 4
        rams[1, 6:8, aux_tasks.IDX_BATTLE] = 1.0                      # a battle at steps 6-7
        dones = torch.zeros(B, T, dtype=torch.bool)
        dones[1, 8] = True                                            # env 1 episode ends after step 8
        tg = aux_tasks.make_targets(rams, dones, k)
        self.assertEqual(int(tg["group"][0, 2]), 24)                  # step 2 + 3 = 5: the new map
        self.assertEqual(int(tg["number"][0, 2]), 4)
        self.assertEqual(int(tg["group"][0, 1]), 26)
        self.assertAlmostEqual(float(tg["hp"][1, 1]), -0.5 * aux_tasks.HP_TARGET_SCALE, places=5)  # step 1 -> 4
        self.assertAlmostEqual(float(tg["hp"][1, 4]), 0.5 * aux_tasks.HP_TARGET_SCALE, places=5)   # step 4 -> 7: back to full
        self.assertEqual(float(tg["battle"][1, 4]), 1.0)              # battle in (4, 7]
        self.assertEqual(float(tg["battle"][1, 1]), 0.0)
        self.assertFalse(bool(tg["battle_valid"][1, 6]))              # already in a battle: masked
        self.assertFalse(bool(tg["valid"][:, T - k:].any()))          # nothing beyond the rollout end
        self.assertFalse(bool(tg["valid"][1, 6]))                     # window 6..8 contains the episode end
        self.assertFalse(bool(tg["valid"][1, 8]))
        self.assertTrue(bool(tg["valid"][1, 3]))                      # window 3..5 is clean

    def test_head_output_shapes_split_and_loss(self):
        policy = small_policy()
        B, T = 2, 12
        screens = torch.rand(B, T, 3, 144, 160)
        rams = make_rams(B, T)
        out = policy.evaluate_actions_sequence(screens, rams, torch.zeros(B, T, dtype=torch.long), None, return_aux=True)
        self.assertEqual(len(out), 4)
        raw = out[3]
        self.assertEqual(tuple(raw.shape), (B, T, aux_tasks.AUX_OUT_DIM))
        parts = aux_tasks.split_outputs(raw)
        self.assertEqual(tuple(parts["group"].shape), (B, T, aux_tasks.AUX_MAP_GROUPS))
        self.assertEqual(tuple(parts["number"].shape), (B, T, aux_tasks.AUX_MAP_NUMBERS))
        self.assertEqual(tuple(parts["hp"].shape), (B, T))
        self.assertEqual(tuple(parts["battle"].shape), (B, T))
        total, items = aux_tasks.aux_loss(parts, aux_tasks.make_targets(rams, None, 4))
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(items), {"map", "hp", "battle"})

    def test_aux_gradients_reach_the_shared_trunk_but_not_the_policy_heads(self):
        policy = small_policy().train()
        B, T = 2, 12
        screens = torch.rand(B, T, 3, 144, 160)
        rams = make_rams(B, T)
        rams[:, 5:, aux_tasks.IDX_MAP_NUMBER] = 9 / 32.0              # something to predict
        _, _, _, raw = policy.evaluate_actions_sequence(screens, rams, torch.zeros(B, T, dtype=torch.long), None,
                                                        return_aux=True)
        total, _ = aux_tasks.aux_loss(aux_tasks.split_outputs(raw), aux_tasks.make_targets(rams, None, 4))
        total.backward()
        def nonzero(p):
            return p.grad is not None and float(p.grad.abs().sum()) > 0
        self.assertTrue(nonzero(policy.aux_head[0].weight))
        self.assertTrue(nonzero(policy.gru.weight_ih_l0))              # shared recurrent trunk
        self.assertTrue(nonzero(policy.fusion[0].weight))
        self.assertTrue(nonzero(policy.ram_trunk[0].weight))
        self.assertTrue(nonzero(policy.vision_trunk.conv1.weight))
        self.assertFalse(nonzero(policy.actor.weight))                 # action / value outputs are not touched
        self.assertFalse(nonzero(policy.critic.weight))

    def test_policy_logits_and_logprobs_do_not_depend_on_the_aux_head(self):
        a = small_policy(aux=True).eval()
        b = small_policy(aux=False).eval()
        b.load_state_dict({k: v for k, v in a.state_dict().items() if not k.startswith("aux_head.")})
        B, T = 1, 5
        screens, rams = torch.rand(B, T, 3, 144, 160), make_rams(B, T)
        acts = torch.randint(0, N_ACTIONS, (B, T))
        with torch.no_grad():
            la, ea, va = a.evaluate_actions_sequence(screens, rams, acts, None)
            lb, eb, vb = b.evaluate_actions_sequence(screens, rams, acts, None)
            l_aux = a.evaluate_actions_sequence(screens, rams, acts, None, return_aux=True)
            hx = None
            step_lp = []
            for i in range(T):                                            # rollout-style single steps
                logits, _, hx = a.forward_recurrent(screens[:, i], rams[:, i], hx)
                step_lp.append(torch.distributions.Categorical(logits=logits).log_prob(acts[:, i]))
        self.assertTrue(torch.allclose(la, lb) and torch.allclose(va, vb) and torch.allclose(ea, eb))
        self.assertTrue(torch.allclose(la, l_aux[0]))
        self.assertTrue(torch.allclose(la, torch.stack(step_lp, dim=1), atol=1e-4))
        with self.assertRaises(RuntimeError):
            b.evaluate_actions_sequence(screens, rams, acts, None, return_aux=True)

    def test_old_checkpoints_load_and_aux_is_optional_at_inference(self):
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d) / "old.pt", Path(d) / "new.pt"
            plain = small_policy(aux=False, default_dims=True)
            plain.save(old)
            blob = torch.load(str(old), map_location="cpu")
            self.assertNotIn("aux_heads", blob)                           # exact old format when the heads are off
            loaded = SpatialVisionPolicy.load(old, override_obs_dim=DEEP_OBS_DIM, override_aux_heads=True)
            self.assertTrue(loaded.aux_enabled)                           # trainer resume: fresh head, rest restored
            self.assertTrue(torch.equal(loaded.actor.weight, plain.actor.weight))
            self.assertTrue(torch.equal(loaded.gru.weight_ih_l0, plain.gru.weight_ih_l0))
            self.assertFalse(SpatialVisionPolicy.load(old, override_obs_dim=DEEP_OBS_DIM).aux_enabled)

            with_aux = small_policy(aux=True, default_dims=True)
            with_aux.save(new)
            self.assertTrue(torch.load(str(new), map_location="cpu")["aux_heads"])
            inference = SpatialVisionPolicy.load(new, override_obs_dim=DEEP_OBS_DIM)
            self.assertFalse(inference.aux_enabled)                       # aux keys dropped, nothing required
            self.assertTrue(torch.equal(inference.actor.weight, with_aux.actor.weight))
            again = SpatialVisionPolicy.load(new, override_obs_dim=DEEP_OBS_DIM, override_aux_heads=True)
            self.assertTrue(torch.equal(again.aux_head[0].weight, with_aux.aux_head[0].weight))

    def test_ppo_update_trains_with_aux_and_reports_losses(self):
        policy = small_policy()
        B, T = 2, 12
        torch.manual_seed(3)
        rams = make_rams(B, T)
        rams[:, 6:, aux_tasks.IDX_MAP_NUMBER] = 7 / 32.0
        batch = RolloutBatch(
            screens=torch.rand(B, T, 3, 144, 160), rams=rams, actions=torch.randint(0, N_ACTIONS, (B, T)),
            old_logps=torch.full((B, T), -np.log(N_ACTIONS), dtype=torch.float32), returns=torch.randn(B, T),
            advantages=torch.randn(B, T), initial_hx=torch.zeros(1, B, 16), dones=torch.zeros(B, T, dtype=torch.bool))
        opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
        before = policy.aux_head[2].weight.detach().clone()
        stats = ppo_update(policy, opt, batch, PPOHParams(epochs=1, num_minibatches=1, target_kl=None, aux_coef=0.05,
                                                          aux_horizon=4))
        for key in ("aux_map_loss", "aux_hp_loss", "aux_battle_loss"):
            self.assertTrue(np.isfinite(stats[key]), key)
        self.assertFalse(torch.equal(before, policy.aux_head[2].weight))
        off = ppo_update(small_policy(), torch.optim.SGD(small_policy().parameters(), lr=1e-2), batch,
                         PPOHParams(epochs=1, num_minibatches=1, target_kl=None))     # aux_coef 0: no aux loss
        self.assertTrue(np.isnan(off["aux_map_loss"]))


# ------------------------------------------------------------------------------------ 4. exploration budget
class ExploreBudgetTests(unittest.TestCase):
    def test_default_budget_behaviour_is_unchanged(self):
        b = {}
        self.assertEqual(budget_pay(b, "explore", 5.0), 5.0)
        self.assertAlmostEqual(budget_pay(b, "explore", 40.0), 25.0)       # the stock 30 cap
        self.assertEqual(budget_pay(b, "explore", 1.0), 0.0)
        self.assertEqual(budget_pay(None, "explore", 3.0), 3.0)

    def test_cap_override_and_weight_scale(self):
        b = {"cap:explore": 12.0}
        paid = sum(budget_pay(b, "explore", 0.5) for _ in range(100))
        self.assertAlmostEqual(paid, 12.0)                                 # capped at the override
        b = {"cap:explore": 12.0, "scale:explore": 0.5}
        self.assertAlmostEqual(budget_pay(b, "explore", 10.0), 5.0)        # weight applied before the cap
        self.assertAlmostEqual(budget_pay(b, "explore", 100.0), 7.0)       # only 7 of the 12 left
        self.assertEqual(budget_pay({"scale:explore": 0.5}, "battle", 4.0), 4.0)   # other budgets untouched

    def test_decay_schedule(self):
        f = t.intrinsic_weight_value
        self.assertEqual(f(0, 0.4, 800), 1.0)
        self.assertAlmostEqual(f(400, 0.4, 800), 0.7)
        self.assertAlmostEqual(f(800, 0.4, 800), 0.4)
        self.assertAlmostEqual(f(5000, 0.4, 800), 0.4)
        self.assertEqual(f(300, 1.0, 800), 1.0)                            # long_v3: end = 1 -> never changes
        self.assertEqual(f(300, 0.4, 0), 1.0)

    def test_extrinsic_outweighs_the_capped_exploration_on_a_full_path(self):
        cfg = load_config("long_v4")
        full_path = 0.08 * 319 + 140 + 60                                  # path + milestones + badge
        self.assertGreater(full_path, 10 * cfg.explore_budget)

    @unittest.skipUnless(HAVE_ROM, "needs the ROM")
    def test_worker_applies_the_override_and_the_scale(self):
        w = t.EnvWorker(0, ["new_bark_outside"], max_steps=40,
                        cfg={"explore_cap": 0.3, "heal_reward": True, "heal_budget": 2.0})
        self.addCleanup(w.close)
        w.reset()
        self.assertEqual(w.budget["cap:explore"], 0.3)
        self.assertEqual(w.budget["cap:heal"], 2.0)
        w.set_intrinsic_scale(0.5)
        self.assertEqual(w.budget["scale:explore"], 0.5)
        explore = 0.0
        for i in range(30):
            _, _, _, _, _, _, info = w.step(i % 4)
            c = info["reward_components"]
            explore += sum(c.get(k, 0.0) for k in ("curiosity", "tile", "neighbor", "momentum", "map", "warp", "interaction"))
        self.assertLessEqual(explore, 0.3 + 1e-6)
        self.assertEqual(w.budget["scale:explore"], 0.5)                   # survives the reset after the time limit


# ------------------------------------------------------------------------------------ 5a. heal reward
class HealRewardTests(unittest.TestCase):
    def _pair(self, hp_b, hp_a, **kw):
        return _gs(party_hp=hp_b, party_count=len(hp_b), party_species=[155] * len(hp_b),
                   party_levels=kw.pop("lv_b", [5] * len(hp_b)), **kw), \
               _gs(party_hp=hp_a, party_count=len(hp_a), party_species=[155] * len(hp_a),
                   party_levels=kw.pop("lv_a", [5] * len(hp_a)), **kw)

    def test_value_is_concave_and_low_hp_pays_most(self):
        self.assertEqual(hp_value(0.0), 0.0)
        self.assertEqual(hp_value(1.0), 1.0)
        near_dead = heal_gain(*self._pair([[1, 20]], [[11, 20]]))          # 5 % -> 55 %
        top_up = heal_gain(*self._pair([[10, 20]], [[20, 20]]))            # 50 % -> 100 %
        self.assertGreater(near_dead, 2 * top_up)
        self.assertGreater(heal_gain(*self._pair([[1, 20]], [[20, 20]])), 0.9)   # a full rescue is worth ~1
        self.assertAlmostEqual(heal_payment(*self._pair([[1, 20]], [[20, 20]]), scale=4.0), 4.0 * (1 - 0.0975), places=3)

    def test_only_genuine_heals_pay(self):
        self.assertEqual(heal_gain(*self._pair([[20, 20]], [[10, 20]])), 0.0)                  # damage
        self.assertEqual(heal_gain(*self._pair([[5, 20]], [[25, 25]], lv_a=[6])), 0.0)         # level-up raises HP and max HP
        self.assertEqual(heal_gain(*self._pair([[5, 20]], [[9, 25]])), 0.0)                    # max HP changed
        b, a = self._pair([[0, 20]], [[20, 20]], fainted=True)
        self.assertEqual(heal_gain(b, a), 0.0)                                                 # white-out recovery
        b, a = self._pair([[0, 20], [0, 30]], [[20, 20], [30, 30]])
        self.assertEqual(heal_gain(b, a), 0.0)                                                 # whole party was fainted
        b, a = self._pair([[5, 20]], [[20, 20]])
        a = _gs(party_hp=[[20, 20]], map_group=26, map_number=6, party_levels=[5])
        self.assertEqual(heal_gain(b, a), 0.0)                                                 # map changed (warp)
        self.assertEqual(heal_gain(*self._pair([[5, 20]], [[5, 20], [9, 9]])), 0.0)            # party size changed

    def test_party_average_and_budget(self):
        two = heal_gain(*self._pair([[0, 20], [30, 30]], [[20, 20], [30, 30]]))
        self.assertAlmostEqual(two, 0.5)                                    # one of two mons rescued
        budget = {"cap:heal": 6.0}
        paid = sum(budget_pay(budget, "heal", heal_payment(*self._pair([[1, 20]], [[20, 20]]))) for _ in range(10))
        self.assertAlmostEqual(paid, 6.0)                                   # a damage / heal loop cannot exceed the budget

    def test_progress_reward_skips_the_flat_heal_when_asked(self):
        b, a = self._pair([[2, 20]], [[20, 20]])
        flat = {}
        progress_reward(b, a, False, set(), set(), components=flat)
        none = {}
        progress_reward(b, a, False, set(), set(), components=none, pay_heal=False)
        self.assertGreater(flat.get("heal", 0.0), 0.0)
        self.assertEqual(none.get("heal", 0.0), 0.0)


# ------------------------------------------------------------------------------------ 5b. low-HP battle practice
class LowHpBattleTests(unittest.TestCase):
    def _battle(self, cur, potions):
        items, counts = ((0x12,), (potions,)) if potions else ((), ())
        return _gs(battle_mode=1, enemy_mon_species=16, map_group=24, map_number=3, party_hp=[[cur, 20]],
                   items=items, item_counts=counts)

    def test_key_is_unchanged_by_default_and_tagged_with_hp_and_potions(self):
        s = self._battle(3, 2)
        self.assertEqual(t.battle_archive_key(s), "m1_24-3_s16")
        self.assertEqual(t.battle_archive_key(s, with_hp=True), "m1_24-3_s16_hLp1")
        self.assertEqual(t.battle_archive_key(self._battle(3, 0), with_hp=True), "m1_24-3_s16_hLp0")
        self.assertEqual(t.battle_archive_key(self._battle(20, 2), with_hp=True), "m1_24-3_s16_hHp1")
        self.assertEqual(t.battle_archive_key(self._battle(14, 2), with_hp=True), "m1_24-3_s16_hMp1")

    def test_weights_and_sampling_bias(self):
        self.assertEqual(t.lowhp_battle_weight("m1_24-3_s16_hLp1", 3.0), 4.0)
        self.assertEqual(t.lowhp_battle_weight("m1_24-3_s16_hLp0", 3.0), 1.0)    # low HP but nothing to heal with
        self.assertEqual(t.lowhp_battle_weight("m1_24-3_s16", 3.0), 1.0)
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(Path(d), 32, protect_suffix=t.LOWHP_TAG)
            for i in range(3):
                a.save_fifo(f"m1_1-1_s{i}_hHp1", b"x")
            a.save_fifo("m1_1-1_s9_hLp1", b"x")
            rng = np.random.default_rng(0)
            hits = sum(a.sample_weighted(rng, lambda k: t.lowhp_battle_weight(k, 3.0)).endswith("hLp1") for _ in range(4000))
            self.assertAlmostEqual(hits / 4000, 4.0 / 7.0, delta=0.04)            # 4 / (3 + 4)

    def test_selector_uses_the_weights_only_when_asked(self):
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(Path(d))
            a.save_fifo("m1_1-1_s1_hHp1", b"x")
            a.save_fifo("m1_1-1_s2_hLp1", b"x")
            def share(w):
                sel = t.StartStateSelector(["s"], 0, archive=None, swarm_prob=0.0, rng=np.random.default_rng(1),
                                           battle_archive=a, battle_prob=1.0, battle_lowhp_weight=w)
                sel.pick()
                return sum(sel.pick().endswith("hLp1") for _ in range(2000)) / 2000
            self.assertAlmostEqual(share(0.0), 0.5, delta=0.05)
            self.assertAlmostEqual(share(3.0), 0.8, delta=0.05)

    def test_low_hp_states_are_evicted_last(self):
        import os
        import time
        with tempfile.TemporaryDirectory() as d:
            a = t.SwarmArchive(Path(d), 2, protect_suffix=t.LOWHP_TAG)
            base = time.time() - 100
            for i, name in enumerate(["m1_1-1_s1_hLp1", "m1_1-1_s2_hHp1", "m1_1-1_s3_hHp1"]):
                (Path(d) / f"{name}.state").write_bytes(b"x")
                os.utime(Path(d) / f"{name}.state", (base + i, base + i))
            a.prune()
            self.assertEqual(a.keys(), ["m1_1-1_s1_hLp1", "m1_1-1_s3_hHp1"])     # the oldest ordinary state went first


# ------------------------------------------------------------------------------------ 5c. entropy
class EntropyTests(unittest.TestCase):
    def test_v4_schedule_starts_low_and_v3_is_untouched(self):
        v4, v3 = load_config("long_v4"), load_config("long_v3")
        s4, s3 = EntropySchedule(v4.entropy_start, v4.entropy_end), EntropySchedule(v3.entropy_start, v3.entropy_end)
        self.assertAlmostEqual(s4.value(0.0), 0.01)
        self.assertAlmostEqual(s4.value(1.0), 0.001)
        self.assertAlmostEqual(s4.value(0.5), 0.0055)
        self.assertAlmostEqual(s3.value(0.0), 0.04)
        self.assertAlmostEqual(s3.value(1.0), 0.01)
        self.assertLess(s4.value(0.0), s3.value(1.0))


if __name__ == "__main__":
    unittest.main()
