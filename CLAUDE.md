# CLAUDE.md — research-mlx-ui

## What this project is

A web UI for Karpathy's autoresearch on Apple Silicon. An AI agent (Claude) runs autonomous ML experiments — modifying `train.py`, training for 5 minutes, keeping or reverting based on `val_bpb` — while a browser dashboard shows live progress. Built on MLX with Muon+AdamW via official `mlx.optimizers.MultiOptimizer`.

## Architecture

Three layers, strict separation:

```
React (Vite + Tailwind + Recharts)  →  FastAPI (async, WebSocket)  →  MLX training loop
         client/                              server/                    train.py + prepare.py
```

- **WebSocket push, REST control** — server pushes live data (agent logs, training ticks, experiment results). Client sends commands (start/stop/config) over REST. Never mix these.
- **No database** — state lives in `results.tsv` and git. The `projects/` system uses filesystem files (`results-{name}.tsv`, `trainpy-{name}.py`).
- **The agent subprocess** is spawned via `claude --print --dangerously-skip-permissions` and its stdout is streamed + parsed for structured events.

## Key constraints

- **`train.py` must stay under ~600 lines.** The autoresearch agent needs to read the whole file to reason about changes. Never split it.
- **`prepare.py` is frozen.** The agent must never modify it. It provides `get_dataloader()` and `evaluate_bpb()`.
- **Use official MLX optimizers.** `mlx.optimizers.Muon` + `MultiOptimizer` (since v0.27.1). Do NOT write custom optimizer implementations.
- **`mx.compile` pattern:** Use `@partial(mx.compile, inputs=state, outputs=state)` wrapping the full train step. Do NOT compile `loss_and_grad_fn` directly — it breaks the call signature.
- **`mx.get_peak_memory()`** not `mx.metal.get_peak_memory()` (deprecated).
- **Single `mx.eval()` per training step** — not per operation. MLX is lazy; premature eval kills throughput.

## Commands

```bash
# Install dependencies
uv sync
cd client && npm install --legacy-peer-deps && cd ..

# Prepare data (first time only, ~2 min)
uv run prepare.py

# Run training directly (5 min)
uv run train.py

# Start server
uv run uvicorn server.main:app --host 127.0.0.1 --port 8000

# Start frontend dev server
cd client && npx vite --port 5173

# Full stack (production)
./start.sh

# Full stack (dev with hot reload)
DEV=1 ./start.sh
```

**Always run the server from the project root**, not from `client/`. The server reads `results.tsv` and `train.py` from the working directory.

## Project system

Research is organized into projects. Each project has isolated results and train.py, stored as files at the **repo root** (not in a subdirectory):

```
results-{name}.tsv     # experiment history for this project (repo root)
trainpy-{name}.py      # train.py snapshot for this project (repo root)
.active-project        # text file with current project name (repo root)
```

- `GET /api/projects` — list all projects with stats
- `POST /api/projects` — create (with optional `forkFrom`)
- `POST /api/projects/{name}/activate` — switch project
- `DELETE /api/projects/{name}` — delete project

Legacy migration: if bare `results.tsv` exists without `.active-project`, auto-creates a "default" project.

## Server modules

| File | Purpose |
|------|---------|
| `server/main.py` | FastAPI app, routes, WebSocket, CORS |
| `server/process_manager.py` | Spawns agent, streams stdout, parses training ticks |
| `server/git_watcher.py` | Polls `results.tsv` for new experiments, broadcasts via WS |
| `server/project_manager.py` | Project CRUD, switching, forking, state persistence |
| `server/hardware.py` | Apple Silicon detection via `system_profiler` |
| `server/program_generator.py` | Generates `program.md` with prior context + adaptive strategy |
| `server/summarizer.py` | Claude API call for experiment summary |

## Frontend structure

| Component | Purpose |
|-----------|---------|
| `App.jsx` | State management (useReducer), routing, WebSocket connection |
| `SetupWizard/` | Project selector, focus areas, hints, experiment limit, start button |
| `Dashboard/` | 3-column layout: agent log, experiment chart, stats/current experiment |
| `Dashboard/ExperimentChart.jsx` | Recharts scatter plot (green=keep, gray=discard, red=crash) |
| `History/` | Experiment list with syntax-highlighted diff viewer |
| `Results/` | Stats grid, improvement percentage, Claude-generated summary |

## WebSocket message types

| Type | Emitted by | Purpose |
|------|-----------|---------|
| `session_started` | process_manager | Session began |
| `session_stopped` | process_manager | Session ended |
| `agent_log` | process_manager | Raw agent output line (classified: info/code/result/error/training) |
| `training_tick` | process_manager | Parsed training step: `{step, loss, tokens_per_sec, elapsed_seconds}` |
| `experiment_started` | process_manager | Detected `uv run train.py` in agent output |
| `experiment_done` | git_watcher | New row in results.tsv |
| `heartbeat` | process_manager | Session alive signal every 5s with elapsed time |

## Git workflow

- `main` branch has the baseline code
- Sessions create `autoresearch/{project}/{date}-{timestamp}` branches
- Agent commits kept experiments, reverts discarded ones
- `results.tsv` tracks all experiments (keeps + discards)

## Known issues / tech debt

- Server restart loses connection to running agent (agent keeps running as orphan). Session persistence (PID file + reconnection) not yet implemented.
- Vite 8 + @tailwindcss/vite peer dep conflict — install with `--legacy-peer-deps`
- Recharts bundle is 500KB+ — could code-split for production
- The `confirm()` call in project deletion should be replaced with a proper modal

## Testing

```bash
# Verify server
curl http://localhost:8000/api/health
curl http://localhost:8000/api/hardware
curl http://localhost:8000/api/projects

# Verify training works
uv run train.py  # takes 5 minutes, outputs val_bpb

# Verify frontend builds
cd client && npm run build
```

## Performance reference (M3 Max 128GB)

- Baseline: 1.362 BPB, 32,863 tok/s, 602 steps in 5 min
- Best achieved: 1.070 BPB (21% improvement) after 12 autonomous experiments
- Key wins: RoPE positional encoding, cosine LR decay, wider-shallower model (depth=4, n_embd=768)
- Compile warmup: ~0.4s (excluded from 5-min budget)
- Peak memory: ~8 GB
