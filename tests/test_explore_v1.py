"""explore_v1: no route knowledge in the reward / start, path progress measured only, event-scaled episode limit."""

import unittest

from src.train_config import load_config, preset


class ExploreV1Config(unittest.TestCase):
    def test_validates_and_removes_route_knowledge(self):
        cfg = preset("explore_v1").validate()
        self.assertFalse(cfg.path_progress)
        self.assertTrue(cfg.path_metric_only)
        self.assertEqual(cfg.path_progress_scale_or_none(), 0.0)          # tracker on, pays nothing
        self.assertFalse(cfg.retreat_penalty or cfg.stagnation_penalty)
        self.assertFalse(cfg.swarm or cfg.battle_archive or cfg.route_ckpt)
        self.assertEqual(cfg.start_states, ["new_bark_outside"])
        self.assertFalse(cfg.level_gate_by_target)
        self.assertTrue(all(v == 0 for v in cfg.dwell_level_targets))
        self.assertTrue(cfg.event_budget)

    def test_long_v9_is_unchanged(self):
        cfg = preset("long_v9").validate()
        self.assertTrue(cfg.path_progress)
        self.assertEqual(cfg.path_progress_scale_or_none(), 1.0)
        self.assertFalse(cfg.event_budget or cfg.path_metric_only)

    def test_loads_by_name_and_same_observation_size(self):
        self.assertEqual(load_config("explore_v1").name, "explore_v1")
        self.assertEqual(load_config("explore_v1").time_feature, load_config("long_v9").time_feature)


class EventBudget(unittest.TestCase):
    def test_limit_grows_with_new_events_only(self):
        try:
            from tools.train_deep_ppo import EnvWorker, resolve_start_states, worker_env_cfg
        except Exception as exc:
            raise unittest.SkipTest(f"trainer unavailable: {exc}")
        cfg = preset("explore_v1")
        env_cfg = worker_env_cfg(cfg, None, None, None)
        env_cfg.update({"seed": 0, "num_envs": 1, "stagger_episodes": False, "episode_len_jitter": 0.0, "jitter_max": 0})
        self.assertEqual(env_cfg["event_budget"], 2048)
        try:
            w = EnvWorker(0, resolve_start_states(cfg, None, None, None), 4096, env_cfg)
        except Exception as exc:
            raise unittest.SkipTest(f"emulator unavailable: {exc}")
        try:
            w.reset()
            self.assertEqual(w.ep_len, 4096)
            w.step(0)
            self.assertEqual(w.ep_len, 4096)               # no new event yet
            w.seen_events.add(999_001)                     # a newly set game event
            w.step(0)
            self.assertEqual(w.ep_len, 4096 + 2048)
            w.step(0)
            self.assertEqual(w.ep_len, 4096 + 2048)        # the same event pays once
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
