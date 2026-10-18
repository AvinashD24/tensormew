# TensorMew Live Demo Guide

This directory contains one-click executable shell scripts for launching, observing, and recording autonomous live evaluation runs of TensorMew on **Pokémon Crystal (Game Boy Color)**.

---

## Quick Start (Executables)

All scripts auto-detect your Conda Python environment and launch PyBoy with appropriate parameters:

| Script | Purpose | Emulation Speed | Window Mode |
| :--- | :--- | :--- | :--- |
| [`./scripts/run_demo_falkner.sh`](file:///Users/shrey/Desktop/swe%20projects/tensormew/tensormew/scripts/run_demo_falkner.sh) | **Falkner Gym Challenge**: Post-egg run to level Cyndaquil to Ember, trade for Onix, heal at Poké Center, and defeat Falkner. | `3x` (Visible) | Single Window |
| [`./scripts/run_demo_full_game.sh`](file:///Users/shrey/Desktop/swe%20projects/tensormew/tensormew/scripts/run_demo_full_game.sh) | **Full Story Run**: End-to-end run from the start of Route 29 all the way to earning the Zephyr Badge. | `3x` (Visible) | Single Window |
| [`./scripts/run_demo_watch_recording.sh`](file:///Users/shrey/Desktop/swe%20projects/tensormew/tensormew/scripts/run_demo_watch_recording.sh) | **Presentation & Recording Mode**: Paces battle text for spectators, saves animated GIF and `.state` checkpoints per milestone. | `2x` (Paced) | Single Window + GIF Recording |
| [`./scripts/run_demo_fast_headless.sh`](file:///Users/shrey/Desktop/swe%20projects/tensormew/tensormew/scripts/run_demo_fast_headless.sh) | **Uncapped Benchmark**: Fast headless evaluation for automated testing and performance benchmarks. | Uncapped (`0x`) | Headless (No GUI) |
| [`./scripts/run_4_instances.sh`](file:///Users/shrey/Desktop/swe%20projects/tensormew/tensormew/scripts/run_4_instances.sh) | **Multi-Instance Showcase**: Launches 4 tiled PyBoy windows in a 2x2 grid for simultaneous observation. | `3x` (Visible) | 4 Tiled Windows |

---

## Detailed Usage & Commands

### 1. Falkner Gym Demo (Recommended)
Demonstrates the agent delivering the Mystery Egg, leveling Cyndaquil in Route 31 grass to learn Ember, catching Bellsprout, trading for Rocky the Onix in Violet City, healing at the Pokémon Center, and defeating Falkner:

```bash
./scripts/run_demo_falkner.sh
```

*Customizing speed (e.g. 5x) or start state:*
```bash
./scripts/run_demo_falkner.sh 5 mystery_egg_received
```

---

### 2. Full End-to-End Story Run
Starts from New Bark Town / Route 29, navigates through Cherrygrove City, collects the Mystery Egg from Mr. Pokémon, defeats the Rival, delivers the egg to Elm, catches wild Pokémon, trades for Onix, and sweeps Violet Gym:

```bash
./scripts/run_demo_full_game.sh
```

*Direct Python invocation:*
```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python3 -m tools.play_live \
    --policy logs/deep_policy.pt \
    --state route29_entry \
    --speed 4
```

---

### 3. Presentation / Showcase Mode with GIF Recording
Designed for presentations and video demos. Slows battle dialogue (`--watch`) so observers can read attack prompts, and automatically records smooth frames into `recordings/demo_run/run.gif`:

```bash
./scripts/run_demo_watch_recording.sh mystery_egg_received recordings/demo_run
```

Recorded files:
- `recordings/demo_run/run.gif`: High-definition animated run replay.
- `recordings/demo_run/*.state`: PyBoy save states generated at each story milestone (e.g., `healed.state`, `rival_beaten.state`, `falkner_beaten.state`).

---

### 4. Headless Fast Benchmark
Evaluates the policy at max hardware emulation speed without rendering an SDL window:

```bash
./scripts/run_demo_fast_headless.sh mystery_egg_received
```

---

### 5. Multi-Instance Tiled Demo
Spawns 4 PyBoy windows positioned in a 2x2 grid on macOS displays (Top-Left, Top-Right, Bottom-Left, Bottom-Right):

```bash
./scripts/run_4_instances.sh mystery_egg_received 3
```

- Real-time logs are written to: `logs/inst1.log`, `logs/inst2.log`, `logs/inst3.log`, `logs/inst4.log`.
- To follow live progress:
  ```bash
  tail -f logs/inst1.log
  ```
- To terminate all 4 instances:
  ```bash
  pkill -f "tools.play_live"
  ```

---

## Available Save States

You can start any demo from any saved story checkpoint using `--state <NAME>` or passing it as the second argument:

| State Name | Location | Story Progress |
| :--- | :--- | :--- |
| `route29_entry` | New Bark Town (Route 29 gate) | Starter chosen, ready to depart |
| `cherrygrove_entry` | Route 29 / Cherrygrove border | Reached Cherrygrove City |
| `mystery_egg_received` | Mr. Pokémon's House | Holding Mystery Egg, ready to return to Elm |
| `rival_beaten` | Cherrygrove City (West) | Defeated Rival, returning to Elm |
| `gave_egg` | Elm's Lab | Egg given to Elm, Poké Balls equipped |
| `violet_entry` | Route 31 / Violet City Gate | Arrived at Violet City |
| `falkner_prep` | Violet City Pokémon Center | Full team healed, ready for Falkner |

---

## Key CLI Arguments for `tools.play_live`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--state` | string | `route29_entry` | Save state name or path to a `.state` file |
| `--speed` | int | `4` | PyBoy emulation speed (`0` = uncapped, `1` = 1x realtime, `3` = 3x) |
| `--headless` | flag | `False` | Run headless without opening an SDL window |
| `--watch` | flag | `False` | Pause on textboxes and force frame rendering for readability |
| `--record-dir` | path | `None` | Directory to output `run.gif` and milestone `.state` files |
| `--record-every` | int | `6` | Overworld steps between captured GIF frames |
| `--record-frames-every` | int | `0` | Capture smooth frame every N hardware frames in Violet City |
| `--policy` | path | `logs/deep_policy.pt` | Trained neural network checkpoint |
| `--post-egg-policy` | path | `None` | Secondary policy checkpoint that activates after receiving the Egg |

---

## Sound & Display Options

- **Enable Sound**: Set the environment variable `PYBOY_SOUND=1`:
  ```bash
  PYBOY_SOUND=1 ./scripts/run_demo_falkner.sh
  ```
- **Window Position**: Position window manually via `SDL_VIDEO_WINDOW_POS="x,y"`:
  ```bash
  SDL_VIDEO_WINDOW_POS="100,100" ./scripts/run_demo_falkner.sh
  ```
