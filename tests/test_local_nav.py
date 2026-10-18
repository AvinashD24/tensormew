import unittest
from types import SimpleNamespace

import numpy as np

from src.collision import COLL_WALL
from src.local_nav import NAV_DIM, LocalNav
from src.ppo import MlpPolicy, Policy, mask_logits
from src.rl_env import (
    BASE_OBS_DIM,
    MASK_START,
    OBJECTIVE_TILE_BOOST,
    OBS_DIM,
    OBS_EVENT_FLAGS,
    observe,
    progress_reward,
    story_objective_maps,
)

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
        self.assertEqual(OBS_DIM, BASE_OBS_DIM + NAV_DIM + len(OBS_EVENT_FLAGS))
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


class _Story:
    """Minimal state with story flags and the fields progress_reward and observe read."""

    def __init__(self, events=(), map_id=(24, 3), x=5, y=5):
        self._events = set(events)
        self.map_group, self.map_number = map_id
        self.x, self.y = x, y
        self.tiles = {"up": FLOOR, "down": FLOOR, "left": FLOOR, "right": FLOOR}
        self.badges = 0
        self.battle_mode = 0
        self.party_count = 1
        self.party_levels = (5,)
        self.pokedex_caught_count = 0
        self.facing = "up"
        self.script_mode = 0
        self.script_running = 0
        self.textbox_flags = 0

    @property
    def map_id(self):
        return (self.map_group, self.map_number)

    def has_event(self, name):
        return name in self._events


class StoryTests(unittest.TestCase):
    def test_objective_maps_follow_the_story(self):
        get_egg = story_objective_maps(_Story())
        self.assertIn((26, 1), get_egg)  # Route 30 leads to Mr. Pokemon
        self.assertIn((26, 10), get_egg)
        returning = story_objective_maps(_Story({"EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}))
        self.assertIn((24, 5), returning)  # Elm's lab
        done = {"EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON", "EVENT_GAVE_MYSTERY_EGG_TO_ELM"}
        self.assertIn((10, 7), story_objective_maps(_Story(done)))  # Violet gym

    def test_new_tile_on_objective_map_pays_the_boost(self):
        before = _Story(map_id=(26, 1), x=5, y=5)
        after = _Story(map_id=(26, 1), x=5, y=4)
        boosted, _, _ = progress_reward(before, after, True, set(), {(26, 1)}, shaping=0.0, legacy_potential=False)
        before = _Story(map_id=(99, 1), x=5, y=5)
        after = _Story(map_id=(99, 1), x=5, y=4)
        plain, _, _ = progress_reward(before, after, True, set(), {(99, 1)}, shaping=0.0, legacy_potential=False)
        self.assertAlmostEqual(boosted - plain, OBJECTIVE_TILE_BOOST - 1.0)

    def test_event_vector_is_in_the_observation(self):
        nav = LocalNav()
        flags = {"EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"}
        obs = observe(_Story(flags), set(), nav)
        self.assertEqual(len(obs), OBS_DIM)
        events = obs[-len(OBS_EVENT_FLAGS) :]
        index = OBS_EVENT_FLAGS.index("EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON")
        self.assertEqual(events[index], 1.0)
        self.assertEqual(float(events.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
