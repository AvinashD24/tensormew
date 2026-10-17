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


if __name__ == "__main__":
    unittest.main()
