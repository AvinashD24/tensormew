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


class ExtendedRewardTests(unittest.TestCase):
    def test_catching_pokemon_has_diminishing_returns_for_duplicates(self):
        seen_tiles, seen_maps = set(), {(1, 1)}
        caught_counts = {}

        # Initial party has 1 mon (Cyndaquil, 155)
        s0 = _S(1, 1, 0, 0)
        s0.party_count = 1
        s0.party_species = (155,)

        # 1st catch: Pidgey (species 16) -> novelty reward 10.0
        s1 = _S(1, 1, 0, 0)
        s1.party_count = 2
        s1.party_species = (155, 16)
        r1, _, _ = progress_reward(
            s0, s1, False, seen_tiles, seen_maps, caught_counts=caught_counts
        )
        # Should get 10.0 catch reward - 0.01 step penalty
        self.assertAlmostEqual(r1, -0.01 - 0.05 + 10.0)

        # 2nd catch: Duplicate Pidgey (species 16) -> 50% diminishing return (5.0)
        s2 = _S(1, 1, 0, 0)
        s2.party_count = 3
        s2.party_species = (155, 16, 16)
        r2, _, _ = progress_reward(
            s1, s2, False, seen_tiles, seen_maps, caught_counts=caught_counts
        )
        self.assertAlmostEqual(r2, -0.01 - 0.05 + 5.0)
        self.assertLess(r2, r1)

        # 3rd catch: Another duplicate Pidgey (species 16) -> 25% diminishing return (2.5)
        s3 = _S(1, 1, 0, 0)
        s3.party_count = 4
        s3.party_species = (155, 16, 16, 16)
        r3, _, _ = progress_reward(
            s2, s3, False, seen_tiles, seen_maps, caught_counts=caught_counts
        )
        self.assertAlmostEqual(r3, -0.01 - 0.05 + 2.5)
        self.assertLess(r3, r2)

        # Catch a new unique species: Rattata (species 19) -> Full 10.0 reward!
        s4 = _S(1, 1, 0, 0)
        s4.party_count = 5
        s4.party_species = (155, 16, 16, 16, 19)
        r4, _, _ = progress_reward(
            s3, s4, False, seen_tiles, seen_maps, caught_counts=caught_counts
        )
        self.assertAlmostEqual(r4, -0.01 - 0.05 + 10.0)
        self.assertGreater(r4, r3)

    def test_story_flag_and_waypoint_rewards(self):
        seen_tiles, seen_maps = set(), {(1, 1)}
        seen_events = set()
        seen_waypoints = set()

        # Step into Violet Gym (10, 7)
        s0 = _S(10, 5, 5, 5)
        s1 = _S(10, 7, 5, 5)
        r_gym, _, _ = progress_reward(
            s0, s1, True, seen_tiles, seen_maps, seen_waypoints=seen_waypoints
        )
        # 10.0 new map + 20.0 Violet Gym waypoint + 1.0 new tile - 0.01
        self.assertAlmostEqual(r_gym, -0.01 + 1.0 + 10.0 + 20.0)

        # Beat Falkner event flag (1213: byte 151, bit 5)
        flags = [0] * 256
        flags[1213 // 8] = 1 << (1213 % 8)
        s2 = _S(10, 7, 5, 5)
        s2.event_flags = tuple(flags)
        s2.badges = 1
        r_falkner, _, _ = progress_reward(
            s1, s2, False, seen_tiles, seen_maps, seen_events=seen_events
        )
        # EVENT_BEAT_FALKNER (50.0) + badge earned (30.0) - 0.01 - 0.05
        self.assertAlmostEqual(r_falkner, -0.01 - 0.05 + 50.0 + 30.0)

    def test_level_up_and_battle_victories(self):
        seen_tiles, seen_maps = set(), {(1, 1)}
        s0 = _S(1, 1, 0, 0)
        s0.party_levels = (5,)
        s1 = _S(1, 1, 0, 0)
        s1.party_levels = (7,)  # Gained 2 levels!
        r_level, _, _ = progress_reward(s0, s1, False, seen_tiles, seen_maps)
        # 2 levels * 5.0 = 10.0
        self.assertAlmostEqual(r_level, -0.01 - 0.05 + 10.0)

        # Battle victory: before was battle_mode=1, after is battle_mode=0
        s_battle = _S(1, 1, 0, 0)
        s_battle.battle_mode = 1
        s_victory = _S(1, 1, 0, 0)
        s_victory.battle_mode = 0
        r_battle, _, _ = progress_reward(s_battle, s_victory, False, seen_tiles, seen_maps)
        # Battle won gives +3.0 and does NOT penalize wall bump
        self.assertAlmostEqual(r_battle, -0.01 + 3.0)


class MlpPolicyTests(unittest.TestCase):
    def test_mlp_policy_act_and_learn(self):
        from src.ppo import MlpPolicy, load_policy
        import tempfile

        policy = MlpPolicy(rng=np.random.default_rng(42), obs_dim=OBS_DIM, hidden_dim=32)
        obs = np.zeros(OBS_DIM)
        obs[0] = 1.0
        action, logp = policy.act(obs)
        self.assertIn(action, range(5))
        self.assertLessEqual(logp, 0.0)

        # Test PPO update on MLP
        batch = {
            "obs": [obs.copy() for _ in range(4)],
            "actions": [0, 1, 0, 1],
            "logp": [-1.6, -1.6, -1.6, -1.6],
            "adv": np.array([1.0, -1.0, 1.0, -1.0]),
            "ret": np.array([1.0, 0.0, 1.0, 0.0]),
        }
        delta = ppo_update(policy, batch)
        self.assertGreater(delta, 0.0)

        # Test save and load via load_policy
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            policy.save(tmp.name)
            loaded = load_policy(tmp.name)
            self.assertIsInstance(loaded, MlpPolicy)
            loaded_action, _ = loaded.act(obs)
            self.assertIn(loaded_action, range(5))


if __name__ == "__main__":
    unittest.main()

