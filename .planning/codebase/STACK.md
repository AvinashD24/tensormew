# Technology Stack

**Analysis Date:** 2026-11-05

## Languages

**Primary:**
- Python - Emulator wrapper, RL environment, policies, LLM supervisor, CLI tools, tests (`src/`, `tools/`, `tests/`, `app.py`, `crystal_smoke.py`, `scratch/route30_nav.py`)
- JavaScript (JSX) - React single-page web app (`tensormewweb/src/`)

**Secondary:**
- HTML/CSS - Web app shell and styles (`tensormewweb/index.html`, `tensormewweb/src/index.css`)
- Bash and PowerShell - Launch and recording scripts (`run_live.sh`, `run_4_live.sh`, `tools/run_hybrid.sh`, `tools/run_hybrid.ps1`, `scripts/*.sh`)
- Z80 assembly symbol file - Game symbols only, consumed by PyBoy (`rom/pokemon_crystal.sym`)

## Runtime

**Python environment:**
- Python version is not pinned. No `pyproject.toml`, `setup.py`, `setup.cfg`, or `.python-version` exists at the repo root.
- Code uses `from __future__ import annotations` and `X | Y` union annotations throughout, so Python 3.10+ is the practical floor. `pokemon_agent.md` describes a planned "Python 3.11+" target, but that is a design doc, not an enforced constraint.
- Scripts reference a macOS Homebrew miniconda interpreter (`/opt/homebrew/Caskroom/miniconda/base/bin/python` in `run_live.sh` and `run_4_live.sh`). Hybrid scripts fall back to `python` or the `PYTHON` env var (`tools/run_hybrid.sh`, `tools/run_hybrid.ps1`).
- Cloud training runs `python3` on a remote Linux pod (`tools/cloud_train.py`).

**Node.js:**
- Node version is not pinned. No `.nvmrc` or `engines` field in `tensormewweb/package.json`.

**Package Managers:**
- pip with `requirements.txt` (root). Lockfile: Not detected (pins are exact `==` versions, which serves a similar purpose).
- npm with `tensormewweb/package-lock.json`. Lockfile: present.

## Frameworks

**Core (Python):**
- PyBoy 2.8.1 - Game Boy / GBC emulator; the core runtime for `src/emulator.py`
- Streamlit 1.66.0 - Thought-log viewer UI in `app.py` (reads `logs/decisions.jsonl`)
- Pydantic 2.14.1 - Tool-call schemas for the LLM supervisor (`src/tools_schema.py`)
- Anthropic Python SDK 1.15.0 - LLM supervisor client (`src/llm_supervisor.py`, `tools/compare_cost.py`)

**Core (JavaScript):**
- React 19 (react 19.3.1, react-dom 19.3.1 installed) - Web UI (`tensormewweb/src/main.jsx`, `tensormewweb/src/App.jsx`)
- Vite 6 (vite 6.4.4 installed) - Dev server and build (`tensormewweb/vite.config.js`)

**Testing:**
- `unittest` (standard library) - Most tests in `tests/`, for example `tests/test_emulator.py`, `tests/test_llm_supervisor.py`. Run with `python -m unittest` or `pytest`.
- pytest - Referenced in `README.md` (`pytest tests/ -q`) and `docs/HYBRID_FALKNER_RUN.md`. Not listed in `requirements.txt`.

**Build/Dev:**
- `@vitejs/plugin-react` 4.7.0 - React Fast Refresh for Vite (`tensormewweb/vite.config.js`)
- Vite dev server configured for port 5173 with `host: true` (`tensormewweb/vite.config.js`)

## Key Dependencies

**Critical (Python, pinned in `requirements.txt`):**
- `pyboy==2.8.1` - Emulator; loads `rom/pokemon_crystal.gbc` and symbols from `rom/pokemon_crystal.sym`
- `numpy==2.5.4` - Observation vectors, linear and GRU policies (`src/ppo.py`, `src/recurrent_policy.py`, `src/deep_policy.py`), checkpoint I/O via `np.savez` / `np.load`
- `anthropic==1.15.0` - Tool-calling LLM supervisor (`src/llm_supervisor.py`)
- `pydantic==2.14.1` - Tool schemas and validation (`src/tools_schema.py`)
- `streamlit==1.66.0` - Thought-log dashboard (`app.py`)
- `pillow==12.3.0` - Frame and GIF handling (`tests/test_recording.py`, `tools/play_live.py`)
- `matplotlib==3.11.2` - Plots and visual metrics (`src/visual_metrics.py`)
- `pandas==3.0.6`, `pyarrow==26.0.0` - Pulled in by Streamlit; no direct import found in project code

**Critical (Python, NOT in `requirements.txt`):**
- `torch` (PyTorch) - Required for the deep policy (`src/deep_policy.py`, `models/deep_policy.pt`), PyTorch PPO training (`tools/train_deep_ppo.py`, `tools/cloud_train_deep.py`), and `tools/pod_checkpoint.py`. Imports are guarded in `src/deep_policy.py` (sets `TORCH_AVAILABLE`), so the rest of the project runs without it. `tests/test_deep_policy.py`, `tests/test_curriculum_and_recurrent_training.py`, and `tests/test_improved_training.py` import torch directly and will fail without it. `pokemon_agent.md` specifies PyTorch>=2.5 as the target.
- `python-dotenv` - Optional. `tools/play_live.py` calls `load_dotenv()` inside a `try/except ImportError`, so a missing package is silently ignored.
- `sqlite3` - Standard library; used by `src/nav_memory.py` and `src/run_log.py`, not a pip dependency.

**Infrastructure (Python, pinned):**
- `opentelemetry-api==1.45.1` - Transitive, likely via Streamlit
- `starlette==1.8.0`, `uvicorn==0.54.0`, `websockets==17.2`, `python-multipart==0.0.32` - Listed in `requirements.txt`; no direct import found in project code (`app.py` is Streamlit-based, not a FastAPI/Starlette app)
- `pysdl2==0.9.17`, `pysdl2-dll==2.32.10` - SDL2 bindings used by PyBoy's window mode (`PYBOY_VISUAL=1`, `tools/play_live.py` window mode)
- `httpx2==2.13.1`, `httpcore2==2.13.1`, `h11==0.16.0`, `truststore==0.10.5` - HTTP stack for the Anthropic SDK

**Critical (JavaScript, `tensormewweb/package.json`):**
- `react` ^19.0.0 and `react-dom` ^19.0.0 - UI framework
- `lucide-react` ^1.16.0 - Icon set (resolved 1.59.0 in lockfile)

**Dev (JavaScript):**
- `@types/react` ^19.0.0, `@types/react-dom` ^19.0.0 - Type declarations (the source is `.jsx`, so these are not used by the build)
- `vite` ^6.2.0 - Bundler

## Configuration

**Environment variables (names only; values are never stored in the repo):**

| Variable | Read in | Purpose | Default |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | `src/llm_supervisor.py` (via SDK), `tools/play_live.py`, `tools/compare_cost.py`, `tests/test_llm_supervisor.py` | Enables the LLM supervisor. Its presence turns the supervisor on in `tools/play_live.py`. | Unset (supervisor off) |
| `POKEMIND_MODEL` | `src/llm_supervisor.py` | Default Claude model | `claude-haiku-4-5` |
| `POKEMIND_HARD_MODEL` | `src/llm_supervisor.py` | Model for hard decisions | Same as `POKEMIND_MODEL` |
| `POKEMIND_INPUT_USD_PER_MTOK` | `src/llm_supervisor.py` | Input token price for cost logging | `1.0` |
| `POKEMIND_OUTPUT_USD_PER_MTOK` | `src/llm_supervisor.py` | Output token price for cost logging | `5.0` |
| `POKEMIND_LIVE` | `tests/test_llm_supervisor.py` | Must be `1` (with `ANTHROPIC_API_KEY`) to run the live API test | Unset (skipped) |
| `PYBOY_VISUAL` | `src/emulator.py` | `1` forces a window instead of headless | Unset |
| `PYBOY_SOUND` | `src/emulator.py` | `1` enables audio | `0` |
| `PYBOY_SPEED` | `src/emulator.py`, `tools/play_live.py` | Emulation speed multiplier (0 = uncapped) | `10` for windowed emulator; `4` for `play_live` |
| `MPLBACKEND` | `src/visual_metrics.py`; `tools/train_marathon.py` sets `TkAgg` via `setdefault` | Matplotlib backend | Unset |
| `STATE`, `SPEED`, `PYTHON` | `tools/run_hybrid.sh`, `tools/run_hybrid.ps1` | Launch script overrides | `new_bark_outside`, `4`, `python` |
| `PYTHONIOENCODING` | `tools/run_hybrid.ps1` | Set to `utf-8` for Windows output | - |

**Environment files:**
- `.env` is gitignored (`.gitignore`). `phase-1/02-dev-b-supervisor.md` instructs developers to put `ANTHROPIC_API_KEY` there. Presence is documented; contents were not read.
- `tensormewweb/.gitignore` ignores `.vercel` and `node_modules/` at the web app level; root `.gitignore` ignores `node_modules/` and `dist/`.

**Application configuration (Python):**
- `src/config.py` - Central path and constant module: `ROM_PATH` (`rom/pokemon_crystal.gbc`), `SYM_PATH`, `STATES_DIR` (`states/`), `MANIFEST_PATH` (`states/manifest.json`), `RUNS_DIR` (`runs/`), `CONFIG_DIR` (`config/`), and `ROM_SHA1` for ROM verification. The ROM hash is checked before the emulator starts (`src/emulator.py`).
- `src/llm_supervisor.py` - Model and pricing defaults (see table above)
- `src/options.py` - Game options applied on state load (`Emulator.load_state(..., apply_options=True)`)

**Build configuration (web):**
- `tensormewweb/vite.config.js` - Vite plus React plugin; dev server on port 5173
- `tensormewweb/index.html` - Vite entry HTML

**No build configuration exists for the Python project.** Install and run by hand per `README.md`.

## Data Files and Persisted Formats

- ROM: `rom/pokemon_crystal.gbc` (US 1.1), verified by SHA-1 in `src/config.py`
- Emulator save states: `states/*.state`, `recordings/**/*.state` (`.state` files are gitignored under `states/` but recordings are committed per `.gitignore`)
- State manifest: `states/manifest.json`
- Policy weights:
  - `models/deep_policy.pt` - PyTorch checkpoint (`torch.save` in `src/deep_policy.py`)
  - `models/deep_policy_baseline_351cb55.pt` - PyTorch checkpoint
  - `models/ppo_policy.npz` - NumPy archive (`np.savez` in `src/ppo.py`)
- Run logs: SQLite databases via `src/run_log.py` (`runs` and `steps` tables) and `src/nav_memory.py` (`transitions` table)
- Decision logs: JSON Lines at `logs/decisions.jsonl` (read by `app.py`; written by the supervisor when `log_path` is set)
- Knowledge base: `data/knowledge/bosses.json`

## Platform Requirements

**Development:**
- Python with pip; PyBoy needs SDL2 for windowed mode (provided by `pysdl2-dll` on pip installs)
- Node.js and npm for `tensormewweb/`
- A legally obtained `rom/pokemon_crystal.gbc` matching the pinned SHA-1 (required; the emulator refuses other ROMs)
- Optional: a CUDA GPU for training (`tools/train_deep_ppo.py`; `models/README.md` cites an RTX 4090)
- Optional: an SSH key at `~/.ssh/id_runpod` and a reachable RunPod pod for cloud training (`tools/cloud_train.py`, `tools/pod_checkpoint.py`)
- Windowed play needs a display; `run_live.sh` and `run_4_live.sh` assume macOS window positioning via `SDL_VIDEO_WINDOW_POS`

**Production:**
- Not applicable as a service. The Python agent is a local CLI and dashboard. The web app is a static Vite build (`npm run build` outputs `dist/`, which is gitignored). No deploy target is configured in the repo. `.vercel` is gitignored in `tensormewweb/`, which suggests Vercel was considered, but no `vercel.json` exists.

---

*Stack analysis: 2026-11-05*
