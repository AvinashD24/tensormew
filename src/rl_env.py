"""Overworld environment. Reward comes from new tiles, maps, and badges."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ram_parser import dialogue_open
from src.step import DIRS

ACTIONS = DIRS + ("a",)
N_ACTIONS = len(ACTIONS)
# 4 neighbor tiles, xy, badges, battle, dialogue, 4 visited-neighbor flags.
OBS_DIM = 13


def progress_reward(before, after, moved: bool, seen_tiles: set, seen_maps: set) -> tuple[float, bool, bool]:
    """Return reward and whether this step discovered a tile or a map."""
    reward = -0.01
    new_tile = False
    new_map = False
    if not moved:
        return reward - 0.05, False, False
    tile = (after.map_group, after.map_number, after.x, after.y)
    if tile not in seen_tiles:
        seen_tiles.add(tile)
        reward += 1.0
        new_tile = True
    map_id = (after.map_group, after.map_number)
    if map_id not in seen_maps:
        seen_maps.add(map_id)
        reward += 10.0
        new_map = True
    if after.badges > before.badges:
        reward += 20.0
    return reward, new_tile, new_map


def observe(state, visited: set[tuple[int, int, int, int]]) -> np.ndarray:
    tiles = [float(state.tiles.get(direction, 0)) / 255.0 for direction in DIRS]
    pos = [state.x / 64.0, state.y / 64.0]
    flags = [
        state.badges / 8.0,
        1.0 if state.battle_mode else 0.0,
        1.0 if dialogue_open(state) else 0.0,
    ]
    neighbors = []
    for direction, (dx, dy) in zip(DIRS, ((0, -1), (0, 1), (-1, 0), (1, 0))):
        key = (state.map_group, state.map_number, state.x + dx, state.y + dy)
        neighbors.append(1.0 if key in visited else 0.0)
    return np.asarray(tiles + pos + flags + neighbors, dtype=np.float64)


@dataclass
class StepInfo:
    discovered_tile: bool = False
    discovered_map: bool = False
    frontier: bool = False


@dataclass
class CrystalEnv:
    machine: object
    reader: object
    stepper: object
    macros: object
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    frontier_prob: float = 0.3
    max_steps: int = 64

    def __post_init__(self):
        self.seen_tiles: set = set()
        self.seen_maps: set = set()
        self.visited: set[tuple[int, int, int, int]] = set()
        self.frontier: list[bytes] = []
        self.frontier_maps: set[tuple[int, int]] = set()
        self.start: bytes | None = None
        self.steps = 0

    def reset(self, start: bytes | None = None) -> np.ndarray:
        if start is not None:
            self.start = start
        blob = self.start
        if self.frontier and self.rng.random() < self.frontier_prob:
            blob = self.frontier[int(self.rng.integers(0, len(self.frontier)))]
        if blob is not None:
            self.machine.load_state(blob)
        self.steps = 0
        # Discovery bonuses start over each episode. The frontier list does not.
        self.seen_tiles = set()
        self.seen_maps = set()
        self.visited = set()
        state = self.reader.parse(self.machine.frame)
        self.seen_maps.add((state.map_group, state.map_number))
        tile = (state.map_group, state.map_number, state.x, state.y)
        self.seen_tiles.add(tile)
        self.visited.add(tile)
        return observe(state, self.visited)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        before = self.reader.parse(self.machine.frame)
        info = StepInfo()
        if before.battle_mode:
            self.macros.choose_move(1)
            moved = False
        elif dialogue_open(before):
            self.macros.mash_text(max_frames=80)
            moved = False
        elif action == 4:
            self.machine.press("a", frames=8)
            self.machine.tick(16, render=False)
            moved = False
        else:
            moved = self.stepper.step(ACTIONS[action]).moved
        after = self.reader.parse(self.machine.frame)
        reward, info.discovered_tile, info.discovered_map = progress_reward(
            before, after, moved, self.seen_tiles, self.seen_maps
        )
        tile = (after.map_group, after.map_number, after.x, after.y)
        self.visited.add(tile)
        map_id = (after.map_group, after.map_number)
        if (info.discovered_map or after.badges > before.badges) and map_id not in self.frontier_maps:
            self.frontier.append(self.machine.save_state())
            self.frontier_maps.add(map_id)
            info.frontier = True
        self.steps += 1
        done = self.steps >= self.max_steps
        return observe(after, self.visited), reward, done, info
