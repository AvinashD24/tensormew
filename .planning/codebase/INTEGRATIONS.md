# External Integrations

**Analysis Date:** 2026-11-05

## APIs & External Services

**LLM (Anthropic Messages API):**
- Used for: Optional tool-calling "supervisor" that picks a high-level action when the agent hits a trigger (dialogue, battle, stuck). Returns one tool call per request.
  - Implementation: `src/llm_supervisor.py` (`Supervisor` class, `client.messages.create(...)` with `tool_choice={"type": "any"}`, `max_tokens=256`)
  - Tool schemas: `src/tools_schema.py` (Pydantic models exported via `anthropic_tools()`); tool execution registry in `src/supervisor_stubs.py` and `src/tool_dispatch.py`
  - Default model: `claude-haiku-4-5` (overridable via `POKEMIND_MODEL` and `POKEMIND_HARD_MODEL`)
  - SDK/Client: `anthropic` Python package 1.15.0 (`requirements.txt`), imported lazily inside `Supervisor._client()`
  - Auth: `ANTHROPIC_API_KEY` env var (read implicitly by the SDK's `anthropic.Anthropic()` constructor)
  - Cost logging: each call appends a row to a JSONL log with model, input and output token counts, tool name, and an estimated USD cost computed from `POKEMIND_INPUT_USD_PER_MTOK` and `POKEMIND_OUTPUT_USD_PER_MTOK`
  - Activation:
    - `tools/play_live.py` enables the supervisor only when `ANTHROPIC_API_KEY` is set in the environment. `load_dotenv()` runs first if `python-dotenv` is installed, so a `.env` file can supply the key.
    - `docs/HYBRID_FALKNER_RUN.md` and `.planning/PROJECT.md` describe a `--live` opt-in flag for real API calls. No `--live` flag exists in `tools/` or `src/`; the gate is the environment variable.
    - `src/llm_supervisor.py` also has an `always_on` mode that calls the API on every decision instead of only on triggers.
  - Tests: `tests/test_llm_supervisor.py` mocks the client. One live test is skipped unless both `ANTHROPIC_API_KEY` and `POKEMIND_LIVE=1` are set.
  - Cost comparison tool: `tools/compare_cost.py` counts `messages.create` calls and reads `ANTHROPIC_API_KEY` to choose between a real and a fake client.

**Remote GPU Training (RunPod):**
- Used for: Multi-worker PPO training on a remote GPU pod
  - Implementation: `tools/cloud_train.py` (PPO, `tools.train_ppo --mlp`), `tools/cloud_train_deep.py` (PyTorch deep policy), `tools/pod_checkpoint.py` (pulls checkpoints home and snapshots the pod on failure)
  - Transport: `rsync` for code sync and `ssh` for remote commands. Both are invoked through `subprocess`; there is no Python SSH library.
  - Host and port: default values are hardcoded in `tools/cloud_train.py` and can be overridden with `--host` and `--port`
  - Auth: SSH private key at `~/.ssh/id_runpod` by default (path set by `--key`). The key contents are not stored in the repo.
  - Remote paths: code is synced to `/workspace/tensormew`; archived checkpoints go to `/workspace/checkpoints/<utc-timestamp>/` (`tools/pod_checkpoint.py`)
  - Host key checking is disabled (`-o StrictHostKeyChecking=no`) in the generated ssh commands in `tools/cloud_train.py`
  - Excludes on sync: `.venv`, `.git`, `__pycache__`, `.pytest_cache`, `.DS_Store`

**Web Assets (external CDNs, loaded by the browser):**
- Pokemon sprite images from pokemondb.net
  - `tensormewweb/src/components/PokedexTab.jsx` (`https://img.pokemondb.net/sprites/crystal/normal/<slug>.png`, with a fallback to `.../sprites/gold/...`)
  - `tensormewweb/src/components/FoundersTab.jsx` (four hardcoded sprite URLs)
- Google Fonts
  - `tensormewweb/src/index.css` imports Instrument Sans, Press Start 2P, and Silkscreen
- No API keys are used for these. They are unauthenticated public GETs.

**Browser APIs:**
- Web Audio API (`AudioContext`) for sound effects in `tensormewweb/src/utils/soundEffects.js`. Sound is generated in code; no audio files are fetched.

## Data Storage

**Databases:**
- SQLite (local file, standard library `sqlite3`). No database server.
  - `src/run_log.py` - `runs` and `steps` tables (`CREATE TABLE IF NOT EXISTS`)
  - `src/nav_memory.py` - `transitions` table for navigation memory
  - Connection: file path passed to the class constructor (no env var)
  - Client: `sqlite3` directly; no ORM

**File Storage:**
- Local filesystem only
  - Game ROM: `rom/pokemon_crystal.gbc` (verified by SHA-1 in `src/config.py`). Must be supplied by the user; not downloaded.
  - Save states: `states/*.state`, `recordings/**/*.state`, indexed by `states/manifest.json`
  - Model weights: `models/deep_policy.pt`, `models/deep_policy_baseline_351cb55.pt`, `models/ppo_policy.npz`
  - Run recordings and GIFs: `recordings/<name>/` (written by `src/recording.py` and `tools/play_live.py --record-dir`)
  - Decision logs: `logs/decisions.jsonl` (default path in `src/agent_loop.py`)
  - Episode logs: `runs/<episode_id>.jsonl` (`src/episode.py`)
  - Knowledge base: `data/knowledge/bosses.json`

**Caching:**
- None. No Redis, Memcached, or similar.

## Authentication & Identity

**Auth Provider:**
- Not applicable for the Python agent. It is a local single-user process.
- Anthropic API key is the only credential. It is read from `ANTHROPIC_API_KEY` in the environment.
- RunPod access uses SSH key authentication (see RunPod section above).
- The web app (`tensormewweb/`) has no login, sessions, or user accounts.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, Datadog, or similar SDK is referenced in the codebase.

**Experiment Tracking:**
- None. No Weights & Biases, MLflow, or TensorBoard references were found.

**Logs:**
- Terminal output via `print` and `src/terminal_monitor.py`. `tools/play_live.py` routes per-instance output to `logs/inst<N>.log` when launched by `run_4_live.sh`.
- Structured JSON Lines decision and episode logs (see File Storage above)
- Live status: `src/live_status.py` with `tools/live_status.py` (local display)
- `app.py` is a Streamlit dashboard that reads `logs/decisions.jsonl` and renders the last 50 decisions (`RECENT_LIMIT`). It is local and read-only.

## CI/CD & Deployment

**Hosting:**
- Not configured. No `vercel.json`, `netlify.toml`, `Procfile`, `fly.toml`, `render.yaml`, or Dockerfile exists. `.vercel` is gitignored in `tensormewweb/`, but nothing deploys it from this repo.

**CI Pipeline:**
- None. No `.github/` directory, `.gitlab-ci.yml`, or `.circleci/` config exists.

**Local orchestration:**
- `run_live.sh`, `run_4_live.sh` - Launch the emulator-based agent (macOS paths and window positions)
- `tools/run_hybrid.sh`, `tools/run_hybrid.ps1` - Record a full start-to-Falkner run to `recordings/<name>/`
- `scripts/run_*.sh` - Demo wrappers documented in `scripts/README.md`

## Environment Configuration

**Required env vars (for LLM features only):**
- `ANTHROPIC_API_KEY` - Needed only when the supervisor is enabled or when running live tests. The agent runs without it.

**Optional env vars:** See the full table in `STACK.md` (`POKEMIND_MODEL`, `POKEMIND_HARD_MODEL`, `POKEMIND_*_USD_PER_MTOK`, `PYBOY_VISUAL`, `PYBOY_SOUND`, `PYBOY_SPEED`, `MPLBACKEND`, `POKEMIND_LIVE`).

**Secrets location:**
- `ANTHROPIC_API_KEY`: process environment, or a `.env` file at the repo root (gitignored, loaded by `python-dotenv` when installed). Not committed.
- RunPod SSH key: `~/.ssh/id_runpod` on the developer machine (outside the repo).
- No secrets are stored in tracked files. `.gitignore` excludes `.env`, `*.sav`, `*.ram`, and `*.rtc`.

## Webhooks & Callbacks

**Incoming:**
- None. The Vite dev server (`tensormewweb/vite.config.js`, port 5173, `host: true`) and the Streamlit app (`app.py`) listen on local network interfaces while running, but expose no custom webhook or API endpoints.

**Outgoing:**
- None. The only outbound network calls from code are the Anthropic Messages API (opt-in), RunPod rsync and ssh (opt-in), and browser asset requests from the web app.

---

*Integration audit: 2026-11-05*
