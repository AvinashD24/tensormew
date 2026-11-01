import json
import tempfile
import unittest
from pathlib import Path

from src.train_config import (
    ConfigError,
    TrainConfig,
    apply_overrides,
    dump_effective_config,
    load_config,
)


class TrainConfigTests(unittest.TestCase):
    def test_blueprint_defaults(self):
        c = load_config("blueprint_v1")
        self.assertEqual(c.ppo.gamma, 0.997)
        self.assertEqual((c.ppo.epochs, c.ppo.num_minibatches), (3, 4))
        self.assertEqual((c.entropy_start, c.entropy_end), (0.03, 0.005))
        self.assertEqual((c.sticky_prob, c.jitter_max, c.envs, c.rollout_steps), (0.1, 6, 16, 128))

    def test_legacy(self):
        c = load_config("legacy")
        self.assertEqual(c.ppo.gamma, 0.99)
        self.assertEqual(c.ppo.epochs, 4)
        self.assertEqual(c.ppo.num_minibatches, 1)
        self.assertEqual((c.sticky_prob, c.jitter_max), (0.0, 0))
        self.assertEqual(c.entropy_start, 0.02)

    def test_json_round_trip(self):
        c = apply_overrides(load_config("blueprint_v1"), {"envs": 8, "start_states": ["a", "b"], "gamma": 0.99})
        with tempfile.TemporaryDirectory() as d:
            p = dump_effective_config(c, Path(d) / "run1")
            self.assertEqual(p.name, "config.json")
            c2 = load_config(str(p))
        self.assertEqual(c.to_dict(), c2.to_dict())
        self.assertEqual(c2.ppo.gamma, 0.99)

    def test_partial_file_and_preset_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"preset": "legacy", "envs": 6, "ppo": {"epochs": 2}}))
            c = load_config(str(p))
        self.assertEqual((c.envs, c.ppo.epochs, c.ppo.gamma), (6, 2, 0.99))

    def test_overrides_skip_none(self):
        c = apply_overrides(load_config("blueprint_v1"), {"envs": None, "lr": 1e-4})
        self.assertEqual((c.envs, c.lr), (16, 1e-4))

    def test_envs_not_divisible(self):
        with self.assertRaises(ConfigError) as cm:
            apply_overrides(load_config("blueprint_v1"), {"envs": 10})
        self.assertIn("divisible", str(cm.exception))

    def test_unknown_key_and_preset(self):
        with self.assertRaises(ConfigError):
            apply_overrides(TrainConfig(), {"bogus": 1})
        with self.assertRaises(ConfigError):
            load_config("nope")

    def test_bad_values(self):
        for bad in ({"sticky_prob": 1.0}, {"jitter_max": -1}, {"lr": 0}, {"entropy_end": 0.5}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                apply_overrides(TrainConfig(), bad)


if __name__ == "__main__":
    unittest.main()
