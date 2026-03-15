# research-mlx-ui

**Autonomous ML research on your Mac — with a live dashboard.**

An AI agent runs experiments on your Apple Silicon Mac while you sleep. It modifies code, trains models, keeps what works, reverts what doesn't, and repeats. You watch it happen in real-time through a browser dashboard. No cloud GPUs. No NVIDIA. Just your Mac.

Built on [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — reimagined for Apple Silicon with a web UI.

---

## What does this actually do?

Imagine you're training a small language model. There are hundreds of knobs to turn: model size, learning rate, activation functions, positional encoding schemes, optimizer settings. A human researcher tries one change at a time, waits 5 minutes for training, checks the score, and decides whether to keep or revert. It's slow, repetitive work.

**This tool automates that entire loop.** An AI agent (Claude) reads a research plan, forms a hypothesis, makes one change to the training code, runs a 5-minute experiment, measures the result, and either keeps the improvement or reverts the change. Then it does it again. And again. All night if you want.

You configure a session through a setup wizard in your browser — pick what to focus on (speed, accuracy, optimizer tuning), add any specific ideas you want tried, and hit start. The dashboard shows you what's happening in real-time: the agent's reasoning, live training metrics, a chart of experiment results, and the current best score.

**In our first session, the agent ran 12 experiments and improved the model by 21% — completely autonomously.**

---

## How is this different from Karpathy's autoresearch?

[Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) is a brilliant concept: give an AI agent a training script, a metric, and a time budget, then let it run unsupervised ML research. It's elegant, minimal, and designed for NVIDIA H100 GPUs.

We took that idea and made three changes:

### 1. Runs on your Mac instead of an H100

The original requires NVIDIA CUDA and PyTorch. We replaced the entire stack with [Apple MLX](https://github.com/ml-explore/mlx) — a framework built specifically for Apple Silicon. Training runs natively on your Mac's GPU through Metal, using unified memory. No cloud, no rental costs, no data leaving your machine.

We use the official `mlx.optimizers.Muon` (the same Muon optimizer from the original) combined with `MultiOptimizer` to split parameter groups — Muon for weight matrices, AdamW for embeddings and norms.

### 2. Added a web UI

The original is CLI-only. You start a script, check back hours later, and parse a TSV file. We added a browser-based dashboard with:

- **Setup wizard** — configure research focus and session length, no CLI needed
- **Live dashboard** — real-time agent log, experiment scatter chart, training progress with ETA countdown
- **Experiment history** — every experiment with syntax-highlighted code diffs
- **Results summary** — AI-generated plain-language explanation of what the agent discovered

### 3. Research projects with cumulative learning

The original starts fresh every time. We added a project system where knowledge accumulates across sessions. The agent receives context about what worked and what failed in prior sessions, with adaptive strategy instructions: exploit proven improvements first, explore new directions when plateaued, never retry known failures.

---

## Quick start

### Prerequisites

- Apple Silicon Mac (M1 or newer)
- [uv](https://astral.sh/uv) (Python package manager)
- [Node.js](https://nodejs.org) 20+
- An [Anthropic API key](https://console.anthropic.com) (for the AI agent and result summaries)

### Install and run

```bash
# Clone
git clone https://github.com/makgunay/research-mlx-ui.git
cd research-mlx-ui

# One command does everything
./start.sh
```

This installs dependencies, downloads the training dataset (~800MB, first run only), builds the frontend, and starts the server.

Open **http://localhost:8000** in your browser.

For development (with hot reload):

```bash
DEV=1 ./start.sh
# Frontend at http://localhost:5173 (proxies API to :8000)
```

### Your first session

1. Open the dashboard in your browser
2. Create a project (or use the auto-created "default" project)
3. Select research focus areas (or leave blank for open exploration)
4. Optionally add hints: *"try cosine learning rate decay"* or *"explore RoPE positional encoding"*
5. Set session length (15 experiments is a good starting point — about 2 hours)
6. Click **Start Research**

The agent will begin running experiments. Each one takes about 6 minutes (5 min training + 1 min evaluation + agent thinking time). Watch the dashboard — you'll see the agent's reasoning in the log, training metrics updating in real-time, and experiment results appearing on the chart.

Green dots are improvements that were kept. Gray dots are changes that didn't help and were reverted.

---

## How it works

### The three-layer stack

```
Browser (React + Tailwind + Recharts)
    |
    | WebSocket (live data) + REST (control)
    |
FastAPI Server (Python, async)
    |
    | subprocess + file watching
    |
MLX Training Loop (train.py + prepare.py)
```

**The browser** shows the setup wizard, live dashboard, experiment history, and results. It connects to the server via WebSocket for real-time updates and REST for control operations (start/stop sessions, create projects).

**The server** manages the AI agent subprocess, parses its output into structured events (training steps, experiment results, hypotheses), watches `results.tsv` for completed experiments, and broadcasts everything to connected browsers.

**The training loop** is a GPT language model trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with a 5-minute wall-clock budget. The metric is `val_bpb` (validation bits per byte) — lower is better. The agent modifies `train.py` to improve this metric.

### The experiment loop

For each experiment, the agent:

1. Reads prior results and forms a hypothesis
2. Makes **one** clean change to `train.py`
3. Runs `uv run train.py` (5-minute training budget)
4. Reads the resulting `val_bpb`
5. If improved: `git commit` and log as "keep"
6. If not improved: `git checkout -- train.py` and log as "discard"
7. Appends the result to `results.tsv`
8. Repeats

All state lives in git and `results.tsv`. No database, no complex infrastructure.

### Research projects

Research is organized into projects, each with its own experiment history and training code:

- **Create** a project to start a new research direction
- **Fork** from an existing project to inherit its best training code
- **Switch** between projects instantly
- **Continue** a project across multiple sessions with cumulative context

The agent receives prior results as context — what worked, what failed, and adaptive strategy instructions that shift from exploitation to exploration as the session progresses.

---

## What the agent discovered (M3 Max, 128GB)

In our first 12-experiment session, starting from a baseline of **1.362 BPB**:

| Experiment | BPB | Change | Result |
|-----------|-----|--------|--------|
| Baseline | 1.362 | depth=6, n_embd=512, flat LR | Starting point |
| Cosine LR | 1.341 | Warmup + cosine decay | **Kept** |
| SwiGLU | 1.342 | SwiGLU activation | Discarded (throughput wash) |
| Wider model | 1.319 | depth=4, n_embd=768 | **Kept** |
| Schedule fix | 1.313 | Adjusted cosine steps | **Kept** |
| High Muon LR | 1.337 | muon_lr=0.03 | Discarded (overshoots) |
| High AdamW LR | 1.271 | adamw_lr=1e-3 | **Kept** |
| More heads | 1.273 | n_head=12 | Discarded (no gain) |
| **RoPE** | **1.076** | RoPE positional encoding | **Kept** (biggest win) |
| Deeper+RoPE | 1.137 | depth=6 with RoPE | Discarded (too slow) |
| Bigger batch | 1.150 | batch_size=64 | Discarded (fewer steps) |
| Less decay | **1.070** | weight_decay=0.01 | **Kept** (new best) |

**Final: 1.070 BPB — 21% improvement from baseline.** The agent discovered that on Apple Silicon with a fixed time budget, throughput matters more than model size (wider-shallower beats deeper), and RoPE positional encoding was the single biggest architectural improvement.

---

## Technical details

### Model

- GPT architecture (causal self-attention + MLP blocks)
- RMSNorm (pre-norm), weight tying (lm_head shares embedding weights)
- Default: 4 layers, 768 embedding dim, 8 attention heads (~23M params)
- Configurable by the agent — that's the whole point

### Optimizer

Uses MLX's official `MultiOptimizer` to combine:
- **Muon** (MomentUm Orthogonalized by Newton-schulz) for 2D weight matrices in Linear layers
- **AdamW** for embeddings, RMSNorm parameters, and the language model head

Both use cosine learning rate decay with linear warmup.

### Dataset

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — 2.1M synthetically generated short stories. Lower entropy than web text, which means small models can score meaningfully well and the agent's improvements are clearly visible. Custom BPE tokenizer with 4,096 vocabulary.

### Hardware requirements

| Chip | RAM | Expected BPB | Throughput |
|------|-----|-------------|------------|
| M1 / M2 base | 8 GB | ~1.8-2.0 | ~8K tok/s |
| M2/M3 Pro | 16-18 GB | ~1.4-1.6 | ~15K tok/s |
| M3/M4 Pro Max | 32-36 GB | ~1.2-1.4 | ~25K tok/s |
| M3/M4 Max | 64-128 GB | ~1.0-1.2 | ~33K tok/s |
| M4 Max (best known) | 128 GB | ~0.95-1.0 | ~45K tok/s |

The system auto-detects your hardware and adjusts defaults accordingly.

---

## Project structure

```
research-mlx-ui/
├── prepare.py              # Data prep: TinyStories download, BPE tokenizer, dataloader
├── train.py                # GPT model + Muon/AdamW + training loop (agent modifies this)
├── start.sh                # One-command startup
├── server/                 # FastAPI backend
│   ├── main.py             # Routes, WebSocket, CORS
│   ├── process_manager.py  # Agent subprocess, output parsing, heartbeat
│   ├── git_watcher.py      # Polls results.tsv for new experiments
│   ├── project_manager.py  # Project CRUD, switching, forking
│   ├── hardware.py         # Apple Silicon detection
│   ├── program_generator.py # Generates program.md with cumulative context
│   └── summarizer.py       # Claude API for result summaries
├── client/                 # React frontend
│   └── src/
│       ├── App.jsx         # State management, routing
│       ├── components/     # SetupWizard, Dashboard, History, Results
│       └── hooks/          # useWebSocket, useExperiments
├── docs/
│   └── HANDOFF.md          # Full technical specification
└── plans/
    └── implementation-plan.md
```

---

## Credits

- **[Andrej Karpathy](https://github.com/karpathy)** — [autoresearch](https://github.com/karpathy/autoresearch) is the foundational concept this project builds on. The experiment loop design, the keep-or-revert git workflow, the `val_bpb` metric, the 5-minute training budget, and the `program.md` research directive are all his ideas. We added MLX support and a UI, but the core insight — that AI agents can do useful ML research autonomously — is his.
- **[Apple MLX team](https://github.com/ml-explore/mlx)** — the MLX framework and the official Muon optimizer implementation.
- **[Keller Jordan](https://kellerjordan.github.io/posts/muon/)** — the Muon optimizer.
- **[scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx)** — reference implementation for MLX training patterns.

## License

MIT
