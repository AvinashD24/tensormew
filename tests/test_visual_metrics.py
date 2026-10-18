"""Unit and integration tests for LiveMetricsPlotter and visual tracking."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app import extract_metric_series
from src.ppo import Policy, collect_rollout
from src.visual_metrics import LiveMetricsPlotter, StepRecord, UpdateRecord


class DummyEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, start=None):
        self.step_count = 0
        return np.zeros(17)

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 5
        reward = 1.0 if self.step_count == 1 else -0.01
        return np.zeros(17), reward, done, None


class VisualMetricsTests(unittest.TestCase):
    def test_step_and_update_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_png = Path(tmp) / "plot.png"
            jsonl = Path(tmp) / "metrics.jsonl"
            plotter = LiveMetricsPlotter(
                title="Test Monitor",
                enabled=True,
                headless=True,
                save_path=save_png,
                jsonl_path=jsonl,
            )

            # Step 1: positive reward
            rec1 = plotter.log_step(reward=1.0, entropy=1.60, done=False, action=2)
            self.assertEqual(rec1.step, 1)
            self.assertAlmostEqual(rec1.cumulative_return, 1.0)
            self.assertAlmostEqual(rec1.entropy, 1.60)

            # Step 2: negative reward
            rec2 = plotter.log_step(reward=-0.05, entropy=1.55, done=True, action=0)
            self.assertEqual(rec2.step, 2)
            self.assertAlmostEqual(rec2.cumulative_return, 0.95)
            self.assertEqual(len(plotter.episode_returns), 1)
            self.assertAlmostEqual(plotter.episode_returns[0], 0.95)
            self.assertEqual(plotter.current_episode, 1)

            # Log update
            up = plotter.log_update(update=0, mean_reward=0.475, weight_change=0.012)
            self.assertEqual(up.update, 0)
            self.assertAlmostEqual(up.mean_reward, 0.475)

            # Save plot to PNG
            saved = plotter.save()
            self.assertIsNotNone(saved)
            self.assertTrue(save_png.is_file())
            self.assertGreater(save_png.stat().st_size, 1000)

            # Check JSONL
            self.assertTrue(jsonl.is_file())
            lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            plotter.close()

    def test_collect_rollout_with_step_callback(self):
        env = DummyEnv()
        policy = Policy(rng=np.random.default_rng(0), obs_dim=17)
        callbacks = []

        def on_step(data):
            callbacks.append(data)

        batch = collect_rollout(env, policy, steps=8, on_step=on_step)
        self.assertEqual(len(batch["rew"]), 8)
        self.assertEqual(len(callbacks), 8)
        self.assertIn("entropy", callbacks[0])
        self.assertIn("reward", callbacks[0])
        self.assertIn("probabilities", callbacks[0])
        self.assertGreater(callbacks[0]["entropy"], 0.0)

    def test_extract_metric_series_from_decisions(self):
        rows = [
            {"actor": "script", "tool": "run_path"},
            {"actor": "rl", "reward": 1.0, "entropy": 1.609},
            {"actor": "rl", "reward": -0.01, "entropy": 1.580},
            {"actor": "rl", "reward": 10.0, "entropy": 1.550},
        ]
        series = extract_metric_series(rows)
        self.assertEqual(len(series["step"]), 3)
        self.assertEqual(series["reward"], [1.0, -0.01, 10.0])
        self.assertAlmostEqual(series["cumulative_return"][0], 1.0)
        self.assertAlmostEqual(series["cumulative_return"][1], 0.99)
        self.assertAlmostEqual(series["cumulative_return"][2], 10.99)
        self.assertEqual(series["entropy"], [1.609, 1.580, 1.550])


if __name__ == "__main__":
    unittest.main()
