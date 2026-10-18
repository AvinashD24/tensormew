import unittest
from types import SimpleNamespace

import numpy as np

from src.collision import COLL_WALL
from src.local_nav import NAV_DIM, LocalNav
from src.ppo import MlpPolicy, Policy, mask_logits
from src.rl_env import BASE_OBS_DIM, MASK_START, OBS_DIM

FLOOR = 0x00


def _state(x, y, up=FLOOR, down=FLOOR, left=FLOOR, right=FLOOR):
    return SimpleNamespace(
        map_group=24,
        map_number=3,
        x=x,
        y=y,
        tiles={"up": up, "down": down, "left": left, "right": right},
    )


class LocalNavTests(unittest.TestCase):
    def test_obs_dim_includes_nav_block(self):
        self.assertEqual(OBS_DIM, BASE_OBS_DIM + NAV_DIM)
        self.assertEqual(len(LocalNav().features(_state(5, 5))), NAV_DIM)

    def test_wall_is_flagged_blocked(self):
        nav = LocalNav()
        state = _state(5, 5, up=COLL_WALL)
        nav.update(state)
        self.assertEqual(nav.blocked_flags(state), [1.0, 0.0, 0.0, 0.0])

    def test_bump_blocks_direction_then_expires(self):
        nav = LocalNav()
        state = _state(5, 5)
        nav.update(state)
        nav.note_action(state, 0, moved=False)
        self.assertEqual(nav.blocked_flags(state)[0], 1.0)
        for _ in range(20):
            nav.note_action(state, 4, moved=False)
        self.assertEqual(nav.blocked_flags(state)[0], 0.0)

    def test_frontier_leads_around_a_wall_to_the_gap(self):
        # Row y=5 is a wall except a gap at x=6. Standing at (5,6) the wall is above us.
        nav = LocalNav()
        here = _state(5, 6, up=COLL_WALL, down=COLL_WALL, left=COLL_WALL)
        nav.update(here)
        nav.update(_state(6, 6, down=COLL_WALL, right=COLL_WALL))  # (6,5) above is open, unvisited
        direction, dist = nav.frontier(here)
        self.assertEqual((direction, dist), ("right", 2))

    def test_no_frontier_when_everything_known_is_visited(self):
        nav = LocalNav()
        state = _state(5, 5, up=COLL_WALL, down=COLL_WALL, left=COLL_WALL, right=COLL_WALL)
        nav.update(state)
        self.assertEqual(nav.frontier(state), (None, 0))

    def test_reset_forgets_everything(self):
        nav = LocalNav()
        nav.update(_state(5, 5))
        nav.note_action(_state(5, 5), 0, moved=False)
        nav.reset()
        self.assertEqual(nav.edges, {})
        self.assertEqual(nav.blocked_flags(_state(5, 5, up=FLOOR)), [0.0] * 4)


class MaskTests(unittest.TestCase):
    def _obs(self, blocked):
        obs = np.zeros(OBS_DIM)
        obs[MASK_START : MASK_START + 4] = blocked
        return obs

    def test_blocked_moves_get_zero_probability(self):
        for cls in (Policy, MlpPolicy):
            policy = cls(rng=np.random.default_rng(0))
            probs = policy.probs(self._obs([1, 0, 1, 0]))
            self.assertAlmostEqual(float(probs[0]), 0.0)
            self.assertAlmostEqual(float(probs[2]), 0.0)
            self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_policy_never_samples_a_blocked_move(self):
        policy = Policy(rng=np.random.default_rng(1))
        obs = self._obs([1, 1, 1, 0])
        actions = {policy.act(obs)[0] for _ in range(200)}
        self.assertTrue(actions <= {3, 4})

    def test_all_blocked_is_left_alone(self):
        logits = np.zeros(5)
        out = mask_logits(logits, self._obs([1, 1, 1, 1]))
        self.assertTrue(np.array_equal(out, logits))

    def test_short_legacy_observation_is_not_masked(self):
        logits = np.zeros(5)
        self.assertTrue(np.array_equal(mask_logits(logits, np.zeros(17)), logits))


if __name__ == "__main__":
    unittest.main()
