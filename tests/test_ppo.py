import unittest

import numpy as np

from src.ppo import Policy, ppo_update
from src.rl_env import OBS_DIM, CrystalEnv
from src.rl_env import progress_reward


class _S:
    def __init__(self, group, number, x, y, badges=0):
        self.map_group = group
        self.map_number = number
        self.x = x
        self.y = y
        self.badges = badges
        self.tiles = {"up": 0, "down": 0, "left": 0, "right": 0}
        self.battle_mode = 0
        self.script_mode = 0
        self.script_running = 0


class RewardTests(unittest.TestCase):
    def test_new_ground_beats_a_wall(self):
        seen_tiles, seen_maps = set(), {(1, 1)}
        bump, _, _ = progress_reward(_S(1, 1, 0, 0), _S(1, 1, 0, 0), False, seen_tiles, seen_maps)
        walk, tile, _ = progress_reward(_S(1, 1, 0, 0), _S(1, 1, 1, 0), True, seen_tiles, seen_maps)
        self.assertGreater(walk, bump)
        self.assertTrue(tile)

    def test_new_map_is_worth_more_than_a_tile(self):
        seen_tiles, seen_maps = set(), {(1, 1)}
        tile_reward, _, _ = progress_reward(_S(1, 1, 0, 0), _S(1, 1, 1, 0), True, seen_tiles, seen_maps)
        seen_tiles, seen_maps = set(), set()
        map_reward, _, new_map = progress_reward(_S(1, 1, 0, 0), _S(2, 2, 0, 0), True, seen_tiles, seen_maps)
        self.assertTrue(new_map)
        self.assertGreater(map_reward, tile_reward)

    def test_reset_clears_episode_discoveries(self):
        class Machine:
            frame = 0

            def load_state(self, blob):
                return None

        class Reader:
            def parse(self, frame):
                return _S(1, 1, 0, 0)

        env = CrystalEnv(Machine(), Reader(), None, None, rng=np.random.default_rng(0))
        env.start = b"start"
        env.seen_maps.add((9, 9))
        env.seen_tiles.add((9, 9, 1, 1))
        env.reset()
        self.assertEqual(env.seen_maps, {(1, 1)})
        self.assertNotIn((9, 9, 1, 1), env.seen_tiles)

    def test_discovery_step_also_pays_for_the_unvisited_neighbor(self):
        class Machine:
            frame = 0

            def press(self, *_args, **_kwargs):
                return None

            def tick(self, *_args, **_kwargs):
                return None

            def save_state(self):
                return b"snap"

            def load_state(self, _blob):
                return None

        class Reader:
            def __init__(self):
                self.state = _S(1, 1, 5, 5)

            def parse(self, _frame):
                return self.state

        class Stepper:
            def __init__(self, reader):
                self.reader = reader

            def step(self, direction):
                delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
                state = self.reader.state
                self.reader.state = _S(
                    state.map_group,
                    state.map_number,
                    state.x + delta[0],
                    state.y + delta[1],
                    state.badges,
                )
                return type("Moved", (), {"moved": True})()

        reader = Reader()
        env = CrystalEnv(Machine(), reader, Stepper(reader), None, rng=np.random.default_rng(0))
        env.start = b"start"
        env.reset()
        _obs, reward, _done, info = env.step(3)
        self.assertTrue(info.discovered_tile)
        self.assertFalse(info.discovered_map)
        self.assertAlmostEqual(reward, -0.01 + 1.0 + 0.2)
        _obs, back, _done, back_info = env.step(2)
        self.assertFalse(back_info.discovered_tile)
        self.assertAlmostEqual(back, -0.01)

    def test_cleared_sets_pay_a_discovered_map_again(self):
        seen_tiles, seen_maps = set(), {(1, 1)}
        _first, _tile, discovered = progress_reward(
            _S(1, 1, 0, 0), _S(4, 7, 2, 3), True, seen_tiles, seen_maps
        )
        self.assertTrue(discovered)
        repeat, _tile, again = progress_reward(
            _S(4, 7, 2, 3), _S(4, 7, 3, 3), True, seen_tiles, seen_maps
        )
        self.assertFalse(again)

        class Machine:
            frame = 0

            def load_state(self, _blob):
                return None

        class Reader:
            def parse(self, _frame):
                return _S(1, 1, 0, 0)

        env = CrystalEnv(Machine(), Reader(), None, None, rng=np.random.default_rng(0))
        env.start = b"start"
        env.frontier.append(b"keep-frontier")
        env.seen_maps.add((4, 7))
        env.seen_tiles.add((4, 7, 2, 3))
        env.reset()
        self.assertEqual(env.frontier, [b"keep-frontier"])
        self.assertNotIn((4, 7), env.seen_maps)
        renewed, _tile, fresh = progress_reward(
            _S(1, 1, 0, 0), _S(4, 7, 2, 3), True, env.seen_tiles, env.seen_maps
        )
        self.assertTrue(fresh)
        self.assertGreater(renewed, repeat)


class _ScriptEnv:
    """Grid that pays the same progress reward. No maps or towns are named."""

    def __init__(self):
        self.pos = (0, 0)
        self.seen_tiles = set()
        self.seen_maps = set()
        self.steps = 0

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return np.zeros(OBS_DIM)

    def step(self, action):
        delta = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}[action]
        nxt = (self.pos[0] + delta[0], self.pos[1] + delta[1])
        moved = action != 4
        before = _S(0, 0, *self.pos)
        after = _S(0, 0, *nxt) if moved else before
        reward, _, _ = progress_reward(before, after, moved, self.seen_tiles, self.seen_maps)
        if moved:
            self.pos = nxt
        self.steps += 1
        return np.zeros(OBS_DIM), reward, self.steps >= 8, None


class TrainStepTests(unittest.TestCase):
    def test_weights_change_from_a_rollout(self):
        policy = Policy(rng=np.random.default_rng(1))
        before = policy.w.copy()
        obs = [np.zeros(OBS_DIM) for _ in range(4)]
        for row in obs:
            row[0] = 1.0
        batch = {
            "obs": obs,
            "actions": [0, 1, 0, 1],
            "logp": [-1.6, -1.6, -1.6, -1.6],
            "adv": np.array([1.0, -1.0, 1.0, -1.0]),
            "ret": np.array([1.0, 0.0, 1.0, 0.0]),
        }
        delta = ppo_update(policy, batch)
        self.assertGreater(delta, 0.0)
        self.assertFalse(np.allclose(before, policy.w))

    def test_entropy_changes_weights_when_advantages_are_equal(self):
        policy = Policy(rng=np.random.default_rng(1))
        policy.w[:] = 0.0
        policy.b[:] = 0.0
        policy.b[0] = 2.0
        obs = np.zeros(OBS_DIM)
        obs[0] = 1.0
        before = policy.w.copy()
        before_probs = policy.probs(obs).copy()
        batch = {
            "obs": [obs.copy() for _ in range(4)],
            "actions": [0, 0, 0, 0],
            "logp": [-20.0, -20.0, -20.0, -20.0],
            "adv": np.zeros(4),
            "ret": np.zeros(4),
        }
        delta = ppo_update(policy, batch)
        after_probs = policy.probs(obs)
        self.assertGreater(delta, 0.0)
        self.assertFalse(np.allclose(before, policy.w))
        self.assertLess(float(after_probs.max()), float(before_probs.max()))


if __name__ == "__main__":
    unittest.main()
