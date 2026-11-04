"""Valid-action masking (flag ``battle_action_mask``, src.battle_mask): rules on synthetic tilemaps, the never-all-False
guarantee, PPO consistency (act vs evaluate_actions log-probs, entropy of the masked distribution, ppo_update ratio 1 with
masks), flags off = unchanged presets, and a real-emulator test that every masked button is a true no-op."""

import dataclasses
import random
import unittest
from pathlib import Path

import numpy as np

from src import battle_mask as bm
from src import train_config

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ synthetic screens
def blank():
    return bytearray([0x7F] * bm.TILEMAP_SIZE)


def put(tm, r, c, v):
    tm[r * bm.TILEMAP_W + c] = v


def text_box(tm, with_text=True):
    for c in range(1, 19):
        put(tm, 12, c, bm.BOX_H)
        put(tm, 17, c, bm.BOX_H)
    put(tm, 12, 0, bm.BOX_TL), put(tm, 12, 19, bm.BOX_TR), put(tm, 17, 0, bm.BOX_BL), put(tm, 17, 19, bm.BOX_BR)
    for r in range(13, 17):
        put(tm, r, 0, bm.BOX_V), put(tm, r, 19, bm.BOX_V)
    if with_text:
        for k, ch in enumerate((0x96, 0xA8, 0xAB, 0xA3)):   # "Wild"
            put(tm, 14, 1 + k, ch)
    return tm


def main_menu(cell):
    tm = text_box(blank(), with_text=False)
    for r in range(13, 17):   # the left message box and the right menu box are separate boxes
        put(tm, r, 8, bm.BOX_V)
    put(tm, 12, 8, bm.BOX_TL)
    for k, ch in enumerate(bm.FIGHT_TILES):
        put(tm, 14, 10 + k, ch)
    r, c = {(0, 0): (14, 9), (0, 1): (14, 15), (1, 0): (16, 9), (1, 1): (16, 15)}[cell]
    put(tm, r, c, bm.ARROW)
    return tm


def move_list(index):
    tm = blank()
    put(tm, 8, 0, bm.BOX_TL), put(tm, 12, 0, bm.BOX_BL)
    for r in range(13, 17):
        put(tm, r, 0, bm.BOX_V), put(tm, r, 4, bm.BOX_V), put(tm, r, 19, bm.BOX_V)
    put(tm, 17, 0, bm.BOX_BL), put(tm, 17, 4, bm.BOX_BL), put(tm, 17, 19, bm.BOX_BR)
    put(tm, 13 + index, 5, bm.ARROW)
    return tm


def party_list(row):
    tm = blank()
    put(tm, 14, 0, bm.BOX_TL), put(tm, 17, 0, bm.BOX_BL)
    put(tm, row, 0, bm.ARROW)
    return tm


def M(*names):   # expected mask with the named actions forbidden
    m = np.ones(6, dtype=bool)
    for n in names:
        m[bm.ACTION_NAMES.index(n)] = False
    return m


class MaskRuleTests(unittest.TestCase):
    def mask(self, tm, n_moves=4):
        return bm.battle_mask_from_tilemap(bytes(tm), n_moves)

    def test_main_menu_edges_and_b(self):
        for cell, forbidden in {(0, 0): ("up", "left", "b"), (0, 1): ("up", "right", "b"),
                                (1, 0): ("down", "left", "b"), (1, 1): ("down", "right", "b")}.items():
            self.assertEqual(bm.classify(bytes(main_menu(cell)))[0], bm.CTX_MAIN)
            np.testing.assert_array_equal(self.mask(main_menu(cell)), M(*forbidden), str(cell))

    def test_move_list(self):
        for i in range(4):
            np.testing.assert_array_equal(self.mask(move_list(i), 4), M("left", "right"))
        np.testing.assert_array_equal(self.mask(move_list(0), 1), M("up", "down", "left", "right"))
        # the cursor cannot be on a move slot that does not exist: unknown screen, no mask
        np.testing.assert_array_equal(self.mask(move_list(3), 2), np.ones(6, dtype=bool))

    def test_party_list(self):
        for row in (1, 3, 5, 7, 9, 11):
            np.testing.assert_array_equal(self.mask(party_list(row)), M("left", "right"))

    def test_plain_text_box_masks_all_directions_but_not_a_or_b(self):
        m = self.mask(text_box(blank()))
        np.testing.assert_array_equal(m, M("up", "down", "left", "right"))
        self.assertTrue(m[bm.BTN_A] and m[bm.BTN_B])

    def test_unknown_screens_are_unmasked(self):
        np.testing.assert_array_equal(self.mask(blank()), np.ones(6, dtype=bool))
        np.testing.assert_array_equal(self.mask(text_box(blank(), with_text=False)), np.ones(6, dtype=bool))   # empty box: transition
        tm = text_box(blank())   # a text box with a cursor of any kind (yes/no prompt, bag list ...) is not plain text
        put(tm, 15, 5, bm.ARROW)
        np.testing.assert_array_equal(self.mask(tm), np.ones(6, dtype=bool))
        tm = text_box(blank())
        put(tm, 15, 5, bm.HOLLOW_ARROW)
        np.testing.assert_array_equal(self.mask(tm), np.ones(6, dtype=bool))
        two = main_menu((0, 0))
        put(two, 16, 9, bm.ARROW)   # two cursors: ambiguous
        np.testing.assert_array_equal(self.mask(two), np.ones(6, dtype=bool))
        np.testing.assert_array_equal(bm.battle_mask_from_tilemap(None), np.ones(6, dtype=bool))
        np.testing.assert_array_equal(bm.battle_mask_from_tilemap(b"\x00" * 10), np.ones(6, dtype=bool))

    def test_menu_box_being_drawn_inside_the_text_area_is_not_text(self):
        """The frame where the FIGHT/PKMN menu box is drawn but its cursor is not there yet must stay unmasked: the next
        frame's input is read by that menu (found by probing the real emulator)."""
        tm = main_menu((0, 0))
        tm[14 * 20 + 9] = 0x7F   # cursor not drawn yet
        np.testing.assert_array_equal(self.mask(tm), np.ones(6, dtype=bool))

    def test_outside_battle_is_all_true_and_a_failed_read_is_safe(self):
        class Emu:
            def read(self, bank, addr, n):
                raise RuntimeError("boom")

        class St:
            battle_mode = 0
            battle_mon_moves = (33, 0, 0, 0)

        np.testing.assert_array_equal(bm.battle_action_mask(Emu(), St()), np.ones(6, dtype=bool))
        St.battle_mode = 1
        np.testing.assert_array_equal(bm.battle_action_mask(Emu(), St()), np.ones(6, dtype=bool))

    def test_never_all_false_and_a_b_never_both_masked(self):
        for fn in (lambda: text_box(blank()), lambda: main_menu((0, 0)), lambda: main_menu((1, 1)),
                   lambda: move_list(0), lambda: party_list(1), blank):
            for n in (0, 1, 2, 4):
                m = self.mask(fn(), n)
                self.assertTrue(m.any())
                self.assertTrue(m[bm.BTN_A] or m[bm.BTN_B])
        # the safety net itself
        for bits in range(64):
            m = bm.sanitize(bm.unpack_mask(bits))
            self.assertTrue(m.any() and (m[bm.BTN_A] or m[bm.BTN_B]))
        np.testing.assert_array_equal(bm.mask_for("nope"), np.ones(6, dtype=bool))

    def test_pack_unpack_round_trip(self):
        for bits in range(64):
            self.assertEqual(bm.pack_mask(bm.unpack_mask(bits)), bits)
        self.assertEqual(bm.pack_mask(np.ones(6, dtype=bool)), 63)

    def test_moves_count(self):
        self.assertEqual(bm.moves_count((33, 43, 0, 0)), 2)
        self.assertEqual(bm.moves_count(()), 0)
        self.assertEqual(bm.moves_count(None), 0)


class FlagTests(unittest.TestCase):
    def test_off_everywhere_except_long_v8(self):
        for name in ("blueprint_v1", "legacy", "long_v1", "long_v2", "long_v3", "long_v4", "long_v5", "long_v6", "long_v7"):
            cfg = train_config.preset(name)
            self.assertFalse(cfg.battle_action_mask, name)
            self.assertFalse(cfg.periodic_eval, name)
        v8, v7 = train_config.preset("long_v8"), train_config.preset("long_v7")
        self.assertTrue(v8.battle_action_mask and v8.periodic_eval)
        self.assertEqual((v8.eval_every, v8.eval_episodes, v8.eval_max_steps), (50, 4, 1024))
        a, b = dataclasses.asdict(v8), dataclasses.asdict(v7)
        diff = {k for k in a if a[k] != b[k]}
        self.assertEqual(diff, {"name", "battle_action_mask", "periodic_eval"})

    def test_config_round_trip_and_validation(self):
        cfg = train_config.preset("long_v8")
        again = train_config.from_dict(cfg.to_dict())
        self.assertTrue(again.battle_action_mask and again.periodic_eval)
        with self.assertRaises(train_config.ConfigError):
            train_config.from_dict({"eval_every": 0})

    def test_worker_env_cfg_carries_the_flag(self):
        from tools.train_deep_ppo import worker_env_cfg
        self.assertTrue(worker_env_cfg(train_config.preset("long_v8"), None)["action_mask"])
        self.assertFalse(worker_env_cfg(train_config.preset("long_v6"), None)["action_mask"])


try:
    import torch
    from src.deep_policy import SpatialVisionPolicy
    from src.ppo_update import PPOHParams, RolloutBatch, ppo_update
    from src.rl_env import DEEP_OBS_DIM
    HAVE_TORCH = True
except Exception:   # pragma: no cover
    HAVE_TORCH = False


def _masks(B, T, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((B, T, 6)) > 0.45
    m[..., 4] = True   # A is always allowed
    return torch.from_numpy(m)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class PPOConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def _policy(self, gru):
        torch.manual_seed(1)
        p = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=6, use_gru=gru, gru_hidden_dim=32)
        p.eval()
        return p

    def _data(self, B, T):
        g = torch.Generator().manual_seed(3)
        screens = torch.rand((B, T, 3, 144, 160), generator=g)
        rams = torch.rand((B, T, DEEP_OBS_DIM), generator=g)
        return screens, rams

    def test_act_and_evaluate_actions_agree_with_a_mask(self):
        p = self._policy(False)
        screen = (np.random.default_rng(0).random((144, 160, 3)) * 255).astype(np.uint8)
        ram = np.random.default_rng(1).random(DEEP_OBS_DIM).astype(np.float32)
        mask = np.array([False, True, False, False, True, False])
        seen = set()
        for _ in range(40):
            a, logp, _v = p.act(screen, ram, action_mask=mask)
            seen.add(a)
            s_t = torch.from_numpy(screen.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            lp, ent, _ = p.evaluate_actions(s_t, torch.from_numpy(ram).unsqueeze(0), torch.tensor([a]), action_mask=torch.from_numpy(mask))
            self.assertAlmostEqual(float(lp), logp, places=5)
            self.assertLessEqual(float(ent), np.log(2) + 1e-4)   # only two allowed actions
        self.assertTrue(seen <= {1, 4}, seen)
        self.assertEqual(p.act(screen, ram, deterministic=True, action_mask=np.array([0, 0, 1, 0, 0, 0], bool))[0], 2)

    def test_rollout_and_sequence_evaluation_agree(self):
        B, T = 2, 3
        p = self._policy(True)
        screens, rams = self._data(B, T)
        masks = _masks(B, T)
        hx = torch.zeros(1, B, 32)
        old_lp, actions, h = [], [], hx
        with torch.no_grad():
            for t in range(T):
                logits, _v, h = p.forward_recurrent(screens[:, t], rams[:, t], h, action_mask=masks[:, t])
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                self.assertTrue(bool(masks[:, t].gather(1, a[:, None]).all()), "sampled a masked action")
                actions.append(a)
                old_lp.append(dist.log_prob(a))
            lp, ent, _ = p.evaluate_actions_sequence(screens, rams, torch.stack(actions, 1), hx, action_mask=masks)
        np.testing.assert_allclose(lp.numpy(), torch.stack(old_lp, 1).numpy(), atol=1e-5)
        # entropy is the entropy of the MASKED distribution: bounded by log(#allowed), and differs from the unmasked one
        allowed = masks.sum(-1).float()
        self.assertTrue(bool((ent <= torch.log(allowed) + 1e-4).all()))
        _lp2, ent_unmasked, _ = p.evaluate_actions_sequence(screens, rams, torch.stack(actions, 1), hx)
        self.assertTrue(bool((ent < ent_unmasked).all()))

    def test_none_mask_changes_nothing(self):
        p = self._policy(False)
        screens, rams = self._data(2, 1)
        a, b = p(screens[:, 0], rams[:, 0]), p(screens[:, 0], rams[:, 0], action_mask=None)
        self.assertTrue(torch.equal(a[0], b[0]))
        full = torch.ones(2, 6, dtype=torch.bool)
        self.assertTrue(torch.equal(a[0], p(screens[:, 0], rams[:, 0], action_mask=full)[0]))

    def _batch(self, p, B, T, with_masks):
        screens, rams = self._data(B, T)
        masks = _masks(B, T)
        hx = torch.zeros(1, B, 32)
        old_lp, actions, h = [], [], hx
        with torch.no_grad():
            for t in range(T):
                logits, _v, h = p.forward_recurrent(screens[:, t], rams[:, t], h, action_mask=masks[:, t] if with_masks else None)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                actions.append(a)
                old_lp.append(dist.log_prob(a))
        return RolloutBatch(screens=screens, rams=rams, actions=torch.stack(actions, 1), old_logps=torch.stack(old_lp, 1),
                            returns=torch.randn(B, T), advantages=torch.randn(B, T), initial_hx=hx,
                            dones=torch.zeros(B, T, dtype=torch.bool), masks=masks if with_masks else None)

    def test_ppo_update_first_ratio_is_one_with_masks_but_not_if_the_update_forgot_them(self):
        B, T = 2, 3
        p = self._policy(True)
        hp = PPOHParams(epochs=1, num_minibatches=1, target_kl=None, bn_mode="eval_refresh", vf_clip=False)
        stats = ppo_update(p, torch.optim.SGD(p.parameters(), lr=0.0), self._batch(p, B, T, True), hp)
        self.assertAlmostEqual(stats["approx_kl_first"], 0.0, places=5)
        # the same rollout, but the update is not given the masks: the ratio is no longer 1 (this is the bug masks prevent)
        batch = self._batch(p, B, T, True)
        batch.masks = None
        stats2 = ppo_update(p, torch.optim.SGD(p.parameters(), lr=0.0), batch, hp)
        self.assertGreater(abs(stats2["approx_kl_first"]), 1e-4)

    def test_ff_policy_ppo_update_with_masks(self):
        B, T = 2, 2
        p = self._policy(False)
        screens, rams = self._data(B, T)
        masks = _masks(B, T, seed=5)
        with torch.no_grad():
            logits, _v = p(screens.reshape(B * T, 3, 144, 160), rams.reshape(B * T, -1), action_mask=masks.reshape(B * T, 6))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            lp = dist.log_prob(a).reshape(B, T)
        batch = RolloutBatch(screens=screens, rams=rams, actions=a.reshape(B, T), old_logps=lp, returns=torch.randn(B, T),
                             advantages=torch.randn(B, T), masks=masks)
        hp = PPOHParams(epochs=1, num_minibatches=2, target_kl=None, bn_mode="eval_refresh", vf_clip=False)
        stats = ppo_update(p, torch.optim.SGD(p.parameters(), lr=0.0), batch, hp)
        self.assertAlmostEqual(stats["approx_kl_first"], 0.0, places=5)


# ------------------------------------------------------------------ real emulator
try:
    from src.emulator import Emulator
    from src.mover import TileStepper
    from src.ram_parser import RamParser
    from src.states import StateRegistry
    HAVE_EMU = (ROOT / "rom" / "pokemon_crystal.gbc").is_file()
except Exception:   # pragma: no cover
    HAVE_EMU = False

# WRAM that matters for "nothing happened" (menu cursors, battle structs, the tilemap); the sprite / sound areas and the text
# delay counter tick by themselves and are left out. The reference is "no button for the same number of frames".
_RAMS = [(0xC4A0, 0x168, 0), (0xC600, 0x140, 0), (0xCF71, 0x60, 0), (0xD0D0, 0x40, 1), (0xD22D, 0x20, 1)]


@unittest.skipUnless(HAVE_EMU, "ROM / PyBoy not available")
class RealEmulatorNoOpTests(unittest.TestCase):
    """A masked button must be a true no-op in the real game (pressed exactly like the worker does: 8 frames + 12 ticks)."""

    @classmethod
    def setUpClass(cls):
        cls.emu = Emulator(headless=True, speed=0, rtc_lock=True)
        cls.reader = RamParser(cls.emu)
        reg = StateRegistry()
        cls.emu.load_state(reg.load_bytes("post_rival_cherrygrove"))
        stepper, rng = TileStepper(cls.emu, cls.reader), random.Random(1)
        for _ in range(400):
            stepper.step(rng.choice(["up", "down", "left", "right"]))
            if cls.reader.parse(cls.emu.frame).battle_mode:
                break
        else:
            raise unittest.SkipTest("no wild battle found")
        for _ in range(80):   # advance the intro text until the FIGHT/PKMN menu is up
            cls._press("a")
            if bm.classify(bm.read_tilemap(cls.emu))[0] == bm.CTX_MAIN:
                break
        else:
            raise unittest.SkipTest("main menu not reached")
        cls.main_state = cls.emu.save_state()

    @classmethod
    def tearDownClass(cls):
        cls.emu.close()

    @classmethod
    def _press(cls, name):
        cls.emu.press(name, frames=8)
        cls.emu.tick(12, render=False)

    @classmethod
    def _snap(cls):
        out = [cls.emu._pb.screen.ndarray[:, :, :3].tobytes()]
        for addr, n, bank in _RAMS:
            raw = bytearray(cls.emu.read(bank, addr, n))
            if addr == 0xCF71:
                raw[0xCFB2 - 0xCF71] = 0   # wTextDelayFrames
            out.append(bytes(raw))
        return tuple(out)

    def _noop_report(self, base):
        """For the screen saved in ``base``: (mask, {button: True when pressing it changed nothing})."""
        self.emu.load_state(base)
        state = self.reader.parse(self.emu.frame)
        mask = bm.battle_action_mask(self.emu, state)
        self.emu.tick(20, render=False)
        idle = self._snap()
        out = {}
        for i, name in enumerate(bm.ACTION_NAMES):
            self.emu.load_state(base)
            self._press(name)
            out[name] = self._snap() == idle
        self.emu.load_state(base)
        return mask, out

    def _check_masked_are_noops(self, base, label):
        mask, noop = self._noop_report(base)
        for i, name in enumerate(bm.ACTION_NAMES):
            if not mask[i]:
                self.assertTrue(noop[name], f"{label}: masked '{name}' changed the game")
        return mask, noop

    def test_main_menu_every_cell(self):
        seen = set()
        for path in ([], ["right"], ["down"], ["right", "down"]):
            self.emu.load_state(self.main_state)
            for b in path:
                self._press(b)
            kind, cell = bm.classify(bm.read_tilemap(self.emu))
            self.assertEqual(kind, bm.CTX_MAIN)
            seen.add(cell)
            mask, noop = self._check_masked_are_noops(self.emu.save_state(), f"main {cell}")
            row, col = cell
            self.assertEqual(bool(mask[bm.UP]), row == 1)
            self.assertEqual(bool(mask[bm.DOWN]), row == 0)
            self.assertEqual(bool(mask[bm.LEFT]), col == 1)
            self.assertEqual(bool(mask[bm.RIGHT]), col == 0)
            self.assertFalse(mask[bm.BTN_B])
            # the unmasked directions really move the cursor (the mask is not too eager either)
            for name in ("up", "down", "left", "right"):
                if mask[bm.ACTION_NAMES.index(name)]:
                    self.assertFalse(noop[name], f"main {cell}: allowed '{name}' did nothing")
        self.assertEqual(seen, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_move_list_left_right_and_single_move(self):
        for moves, pps in (([33, 43, 45, 39], [27, 30, 40, 30]), ([33, 0, 0, 0], [27, 0, 0, 0])):
            self.emu.load_state(self.main_state)
            self.emu.write(0, 0xC62E, bytes(moves))
            self.emu.write(0, 0xC634, bytes(pps))
            self._press("a")   # FIGHT
            n = sum(1 for m in moves if m)
            for k in range(n):
                if k > 0:
                    self._press("down")
                kind, detail = bm.classify(bm.read_tilemap(self.emu), n)
                self.assertEqual((kind, detail), (bm.CTX_MOVES, (k,)))
                mask, noop = self._check_masked_are_noops(self.emu.save_state(), f"moves n={n} k={k}")
                self.assertFalse(mask[bm.LEFT] or mask[bm.RIGHT])
                self.assertEqual(bool(mask[bm.UP]), n > 1, "up/down wrap with 2+ moves, so they must NOT be masked")
                self.assertEqual(bool(mask[bm.DOWN]), n > 1)
                if n > 1:
                    self.assertFalse(noop["up"] or noop["down"])   # the wrap really moves the cursor

    def test_plain_battle_text_box_directions_are_noops(self):
        self.emu.load_state(self.main_state)
        self._press("a")
        self._press("a")   # use the first move: battle text follows
        checked = 0
        for _ in range(40):
            base = self.emu.save_state()
            kind, _ = bm.classify(bm.read_tilemap(self.emu))
            if kind == bm.CTX_TEXT:
                mask, _ = self._check_masked_are_noops(base, "text")
                self.assertFalse(mask[bm.UP] or mask[bm.DOWN] or mask[bm.LEFT] or mask[bm.RIGHT])
                self.assertTrue(mask[bm.BTN_A] and mask[bm.BTN_B])
                checked += 1
            self.emu.load_state(base)
            self._press("a")
            if self.reader.parse(self.emu.frame).battle_mode == 0:
                break
        self.assertGreater(checked, 2)


class WorkerMaskTests(unittest.TestCase):
    """The training worker publishes the mask of the CURRENT observation (packed bits) and all-True outside battle."""

    @unittest.skipUnless(HAVE_EMU, "ROM / PyBoy not available")
    def test_worker_reports_masks(self):
        from tools.train_deep_ppo import EnvWorker
        w = EnvWorker(0, ["post_rival_cherrygrove"], 400, {"action_mask": True, "seed": 3})
        try:
            w.reset()
            np.testing.assert_array_equal(w.last_mask, np.ones(6, dtype=bool))
            rng = random.Random(1)
            main_seen = False
            for _ in range(700):
                st = w.reader.parse(w.emu.frame)
                if st.battle_mode:
                    act = 4   # A
                else:
                    act = rng.choice([0, 1, 2, 3])
                *_, info = w.step(act)
                self.assertEqual(info["amask"], bm.pack_mask(w.last_mask))
                if bm.classify(bm.read_tilemap(w.emu))[0] == bm.CTX_MAIN:
                    main_seen = True
                    self.assertFalse(bm.unpack_mask(info["amask"])[bm.BTN_B])
                    break
            self.assertTrue(main_seen, "never reached a battle menu")
        finally:
            w.close()

    @unittest.skipUnless(HAVE_EMU, "ROM / PyBoy not available")
    def test_vector_env_publishes_masks_in_a_battle(self):
        from tools.train_deep_ppo import VectorEnv
        env = VectorEnv(num_envs=1, start_states=["post_rival_cherrygrove"], max_steps=600, env_cfg={"action_mask": True, "seed": 3})
        try:
            env.reset()
            np.testing.assert_array_equal(env.last_masks, np.ones((1, 6), dtype=bool))
            rng = random.Random(1)
            in_battle, masked_seen = False, False
            for _ in range(700):
                act = 4 if in_battle else rng.choice([0, 1, 2, 3])
                env.step(np.array([act]))
                info = env.last_infos[0]
                in_battle = bool(info["battle"])
                np.testing.assert_array_equal(env.last_masks[0], bm.unpack_mask(info["amask"]))
                if in_battle and not env.last_masks[0].all():
                    masked_seen = True
                    break
                if not in_battle:
                    self.assertTrue(env.last_masks[0].all(), "overworld must never be masked")
            self.assertTrue(masked_seen)
        finally:
            env.close()

    @unittest.skipUnless(HAVE_EMU, "ROM / PyBoy not available")
    def test_flag_off_worker_has_no_mask_key(self):
        from tools.train_deep_ppo import EnvWorker
        w = EnvWorker(0, ["new_bark_outside"], 50, {"seed": 3})
        try:
            w.reset()
            *_, info = w.step(0)
            self.assertNotIn("amask", info)
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
