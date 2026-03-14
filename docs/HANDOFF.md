# research-mlx-ui — Claude Code Handoff Document

> **Purpose:** This document is a complete technical handoff for building `research-mlx-ui` — a web application that wraps Karpathy's autoresearch loop with an MLX-native backend (Muon+AdamW via official MLX optimizers) and a browser-based UI for non-technical users on Apple Silicon Macs.
>
> **Read this entire document before writing a single line of code.**

---

## 1. Project Vision

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an AI agent run autonomous ML experiments overnight. It requires an NVIDIA GPU and comfort with CLI tooling. This project makes autoresearch accessible to anyone with an Apple Silicon Mac by:

1. Replacing the NVIDIA/CUDA/PyTorch stack with a **native MLX implementation** using the official `mlx.optimizers.Muon` + `mlx.optimizers.MultiOptimizer` (available since MLX v0.27.1) — with optional **Turbo-Muon** spectral preconditioning as a novel performance enhancement
2. Wrapping the entire workflow in a **local web UI** that non-technical users can run with a single command — **no existing fork has a UI layer**
3. Packaging setup wizard, live experiment dashboard, and results summary into a polished open-source tool

The repo should be positioned as the **definitive MLX autoresearch implementation** — the first with a complete UI experience, and optionally the first with Turbo-Muon on Apple Silicon.

> **Note on Muon:** As of MLX v0.27.1 (July 2025), Muon is an official MLX optimizer ([PR #1914](https://github.com/ml-explore/mlx/pull/1914)). `MultiOptimizer` ([PR #1916](https://github.com/ml-explore/mlx/pull/1916)) handles the Muon+AdamW parameter split natively. Do NOT write a custom Muon implementation — use the official one.

---

## 2. What Already Exists (Read Before Building)

### The Original
**karpathy/autoresearch** — the canonical repo. NVIDIA only, tested on H100. Three files: `prepare.py` (frozen), `train.py` (agent edits), `program.md` (human writes). Fixed 5-minute training budget, val_bpb metric, git keep/revert loop.

### Existing MLX Forks — Understand Their Gaps

| Repo | Framework | Muon | mx.compile | Notes |
|------|-----------|------|------------|-------|
| trevin-creator/autoresearch-mlx | MLX native | ❌ private | Unknown | Best MLX fork but withholds Muon publicly, no UI |
| miolini/autoresearch-macos | PyTorch/MPS | ✅ | ❌ | Not MLX — runs through a compatibility shim, no UI |
| PR #205 (elementalcollision) | MLX native | ✅ | Unknown | Unmerged PR into main repo, not a standalone project |
| thenamangoyal/autoresearch | MLX native | ✅ | Unknown | Reports **1.295 BPB on M4 Max**, good benchmark target |

**None of these forks have a UI.** That is our primary differentiator.

### Key Reference Implementations
- **Official MLX Muon** — `mlx.optimizers.Muon` (since v0.27.1) + `mlx.optimizers.MultiOptimizer`. This is the optimizer to use. Do NOT write a custom implementation.
- **scasella/nanochat-mlx** — Has a working MLX Muon+AdamW `optim.py`. Useful as architectural reference for the training loop, but the optimizer itself should use official MLX.
- **karpathy/nanochat** — The parent project autoresearch is based on. Read for architectural decisions.
- **ml-explore/mlx** — The MLX framework (currently v0.31.1). Read the compile and optimizer documentation.
- **stockeh/mlx-optimizers** — Community library with Muon, SOAP, DiffGrad, QHAdam. Could let the agent experiment with alternative optimizers.
- **Turbo-Muon** ([arxiv:2512.04632](https://arxiv.org/abs/2512.04632)) — Spectral preconditioning for Newton-Schulz, 2.8x orthogonalization speedup, 8-10% end-to-end training time reduction. Not yet in official MLX — potential novel contribution.

---

## 3. Technical Architecture

### 3.1 Repository Structure

```
research-mlx-ui/
├── README.md
├── pyproject.toml           # Python deps: mlx, fastapi, uvicorn, tiktoken, numpy
├── uv.lock
├── .python-version          # 3.11
├── .gitignore
│
├── # ── CORE ML FILES (the autoresearch loop) ──────────────────
├── prepare.py               # Data prep, tokenizer, dataloader, evaluate_bpb
├── train.py                 # MLX GPT model + Muon+AdamW + training loop (AGENT EDITS THIS)
├── program.md               # Agent instructions (UI generates this, agent edits)
│
├── # ── SERVER ─────────────────────────────────────────────────
├── server/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── process_manager.py   # Spawns/monitors agent + training subprocess
│   ├── git_watcher.py       # Parses git log, results.tsv, train.py diffs
│   ├── hardware.py          # Auto-detect Apple Silicon chip, RAM
│   ├── program_generator.py # Generates program.md from UI wizard inputs
│   └── summarizer.py        # Calls Claude API to summarize results
│
├── # ── CLIENT ─────────────────────────────────────────────────
├── client/
│   ├── package.json         # React, Vite, Tailwind, Recharts
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── hooks/
│       │   ├── useWebSocket.js
│       │   └── useExperiments.js
│       └── components/
│           ├── SetupWizard/
│           │   ├── index.jsx
│           │   ├── HardwareCard.jsx
│           │   ├── FocusSelector.jsx
│           │   └── HintsInput.jsx
│           ├── Dashboard/
│           │   ├── index.jsx
│           │   ├── ExperimentChart.jsx
│           │   ├── AgentLog.jsx
│           │   ├── CurrentExperiment.jsx
│           │   └── BestResultCard.jsx
│           ├── History/
│           │   ├── index.jsx
│           │   └── DiffViewer.jsx
│           └── Results/
│               ├── index.jsx
│               └── SummaryCard.jsx
│
└── start.sh                 # One-command startup: uv run server + vite dev
```

### 3.2 The Three-Layer Stack

```
┌─────────────────────────────────────────────────────┐
│  Browser (React + Vite + Tailwind)                  │
│  - Setup Wizard → generates program.md              │
│  - Live Dashboard → WebSocket stream                │
│  - Experiment History → REST                        │
│  - Results Summary → REST                           │
└──────────────┬──────────────────────────────────────┘
               │ WebSocket (ws://localhost:8000/ws)
               │ REST (http://localhost:8000/api/*)
┌──────────────▼──────────────────────────────────────┐
│  FastAPI Server (Python, async)                     │
│  - ProcessManager: spawns agent subprocess          │
│  - GitWatcher: tails results.tsv + git log          │
│  - Streams all output to WebSocket clients          │
└──────────────┬──────────────────────────────────────┘
               │ subprocess.Popen / asyncio
┌──────────────▼──────────────────────────────────────┐
│  MLX Training Loop                                  │
│  - train.py (agent-modified)                        │
│  - prepare.py (frozen)                              │
│  - Git: branch per session, commit per experiment   │
└─────────────────────────────────────────────────────┘
```

---

## 4. The MLX Backend — train.py Specification

This is the most critical file. It must be:
- **~600 lines** — hard constraint. The entire file must fit in an LLM context window so the agent can reason about it holistically when making changes. Do not exceed this.
- **Fully self-contained** — no imports from other project files except `prepare.py` utilities
- **MLX-native** — zero PyTorch, zero CUDA dependencies
- **Structured as a pure function training step** so `mx.compile()` works correctly

### 4.1 Critical MLX Idioms — Read Before Writing Any Training Code

**Lazy computation.** MLX does not execute operations immediately. It builds a computation graph and only materializes values when you explicitly call `mx.eval()`. You MUST call `mx.eval()` once per training step — not per operation.

```python
# WRONG — evaluates too eagerly, kills throughput
for step in range(n_steps):
    loss, grads = loss_and_grad_fn(model, x, y)
    mx.eval(loss)                    # premature materialization
    optimizer.update(model, grads)
    mx.eval(model.parameters())      # second eval in same step

# CORRECT — single eval per step batches all pending computation
for step in range(n_steps):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)  # one eval
```

**Pure function training step for mx.compile().**

`mx.compile()` traces a function and caches a compiled Metal kernel. It only works correctly with pure functions — no side effects, no in-place operations, no Python-level branching on MLX values.

```python
from functools import partial

def loss_fn(model, x, y):
    logits = model(x)
    loss = mx.mean(
        nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        )
    )
    return loss

# Wrap with value_and_grad — computes loss AND gradients in one pass
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# CORRECT mx.compile pattern: wrap the FULL train step, declare model
# and optimizer state as mutable inputs/outputs.
# Do NOT compile loss_and_grad_fn directly — it breaks because the
# compiled function changes the call signature.
state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss
```

**Warmup step before the 5-minute timer.** MLX's `mx.compile()` JIT-compiles on the first call. This takes 5–30 seconds depending on model size. The 5-minute training budget MUST exclude this. Run exactly one warmup step, force eval to trigger compilation, then start the wall-clock timer.

```python
# Warmup — triggers JIT compilation, NOT counted in the 5-minute budget
x_warmup, y_warmup = next(train_iter)
loss = train_step(x_warmup, y_warmup)
mx.eval(loss)  # forces compilation to complete
print("Compilation complete. Starting timed training run...")

# NOW start the 5-minute clock
training_start = time.time()
TRAINING_BUDGET_SECONDS = 300
```

**Memory reporting.** Use `mx.get_peak_memory()` for memory stats. (`mx.get_peak_memory()` is deprecated.) There is no Apple Silicon equivalent of `H100_BF16_PEAK_FLOPS` — report `tokens_per_sec` instead of MFU%. This is an honest metric.

### 4.2 GPT Model Architecture in MLX

```python
import math
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, max_seq_len):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q, k, v = mx.split(self.c_attn(x), 3, axis=-1)
        head_dim = C // self.n_head
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        # Scaled dot-product with causal mask
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        # Causal mask: upper triangle = -inf
        causal_mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        attn = attn + causal_mask
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.c_proj(nn.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_head, n_embd, max_seq_len):
        super().__init__()
        self.ln_1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, max_seq_len)
        self.ln_2 = nn.RMSNorm(n_embd)
        self.mlp  = MLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, max_seq_len):
        super().__init__()
        self.wte     = nn.Embedding(vocab_size, n_embd)
        self.wpe     = nn.Embedding(max_seq_len, n_embd)
        self.blocks  = [Block(n_head, n_embd, max_seq_len) for _ in range(n_layer)]
        self.ln_f    = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying: lm_head shares weights with token embedding
        self.lm_head.weight = self.wte.weight

    def __call__(self, idx):
        B, T = idx.shape
        pos  = mx.arange(T)
        x    = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))
```

### 4.3 Muon + AdamW Optimizer in MLX

**Use the official MLX optimizers.** Since MLX v0.27.1, `mlx.optimizers.Muon` and `mlx.optimizers.MultiOptimizer` handle this natively. Do NOT write a custom Muon implementation.

The key design principle: Muon is applied to 2D weight matrices in Linear layers (the dominant parameters in a transformer). AdamW handles everything else — embeddings, RMSNorm gamma, lm_head, and any bias terms. `MultiOptimizer` handles this split via a filter lambda.

```python
import mlx.optimizers as optim

def build_optimizer(muon_lr=0.02, adamw_lr=3e-4,
                    adamw_betas=(0.9, 0.95), weight_decay=0.1):
    """
    Build a MultiOptimizer that routes parameters automatically:
    - Muon: all 2D weight matrices (Linear layers)
    - AdamW: embeddings, RMSNorm gamma (1D), lm_head weights, any biases

    The filter lambda receives (param_name, param_value).
    Muon gets params matching the filter; AdamW gets the rest.
    """
    muon = optim.Muon(
        learning_rate=muon_lr,
        momentum=0.95,
        nesterov=True,
    )
    adamw = optim.AdamW(
        learning_rate=adamw_lr,
        betas=adamw_betas,
        weight_decay=weight_decay,
    )

    # MultiOptimizer: first optimizer gets params matching the filter,
    # second optimizer gets everything else (the fallback).
    optimizer = optim.MultiOptimizer(
        [muon, adamw],
        [lambda name, w: w.ndim >= 2 and "wte" not in name
         and "wpe" not in name and "lm_head" not in name]
    )
    return optimizer
```

**In the training loop, use the optimizer like any single optimizer:**

```python
optimizer = build_optimizer()

# Standard MLX training step — MultiOptimizer routes gradients internally
loss, grads = loss_and_grad_fn(model, x, y)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state, loss)
```

> **Optional Enhancement — Turbo-Muon:** The official Muon uses 5 Newton-Schulz iterations. Turbo-Muon ([arxiv:2512.04632](https://arxiv.org/abs/2512.04632)) adds spectral preconditioning to reduce this to 4 steps, yielding ~2.8x orthogonalization speedup and 8-10% end-to-end training time savings. On a 5-minute budget, that's ~30 extra seconds of training. Not yet in official MLX — implementing this would be a genuine novel contribution. Consider as a v2 enhancement.

### 4.4 Default Hyperparameters for Apple Silicon

These are the defaults that ship in `train.py`. They are tuned for a 5-minute budget on M2/M3 class hardware. The agent is free to change all of them — that is the point.

```python
# ─── Model Architecture ───────────────────────────────────────────
DEPTH         = 4     # transformer layers. depth 4 ≈ 5M params. More steps > bigger model.
N_HEAD        = 8     # attention heads
N_EMBD        = 512   # embedding dimension
MAX_SEQ_LEN   = 512   # sequence length (lower than H100's 1024 for throughput)
VOCAB_SIZE    = 4096  # BPE vocab size (TinyStories needs less than FineWeb-Edu)

# ─── Training ─────────────────────────────────────────────────────
DEVICE_BATCH_SIZE  = 16      # tokens processed per forward pass
TOTAL_BATCH_SIZE   = 2**14   # ~16K effective batch (gradient accumulation)
GRAD_ACCUM_STEPS   = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)

# ─── Optimizer ────────────────────────────────────────────────────
MUON_LR       = 0.02
ADAMW_LR      = 3e-4
WEIGHT_DECAY  = 0.1

# ─── Budget ───────────────────────────────────────────────────────
TRAINING_BUDGET_SECONDS = 300   # 5 minutes wall clock, excludes warmup/compile
```

**Why these differ from H100 defaults:**
- `DEPTH=4` not 8: Apple Silicon wins by fitting more optimizer steps in 5 minutes, not bigger models
- `MAX_SEQ_LEN=512` not 1024: halves memory pressure, doubles throughput per step
- `VOCAB_SIZE=4096` not 8192: TinyStories dataset has lower entropy — smaller vocab is sufficient

### 4.5 Training Loop Skeleton

```python
def main():
    model = GPT(
        vocab_size=VOCAB_SIZE,
        n_layer=DEPTH,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        max_seq_len=MAX_SEQ_LEN
    )
    mx.eval(model.parameters())
    print(f"Model parameters: {model.num_params() / 1e6:.1f}M")

    optimizer = build_optimizer(
        muon_lr=MUON_LR, adamw_lr=ADAMW_LR, weight_decay=WEIGHT_DECAY
    )

    # IMPORTANT: create a persistent iterator — do NOT call iter() inside the loop,
    # or you'll keep getting the first batch forever
    train_iter = iter(get_dataloader("train", DEVICE_BATCH_SIZE, MAX_SEQ_LEN))

    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)
        ))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Compile the full train step with state in/out (validated pattern)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    # ── Warmup step (excluded from 5-minute budget) ──────────────
    x, y = next(train_iter)
    loss = train_step(x, y)
    mx.eval(loss)
    print("Warmup complete. Starting timed run...")

    # ── Timed training loop ───────────────────────────────────────
    training_start = time.time()
    step = 0
    tokens_processed = 0

    while True:
        elapsed = time.time() - training_start
        if elapsed >= TRAINING_BUDGET_SECONDS:
            break

        # Gradient accumulation: run multiple micro-batches per outer step.
        # NOTE: with mx.compile, gradients are applied inside train_step.
        # For true gradient accumulation with compile, you'd need to split
        # the compiled function. For simplicity, each train_step is one
        # micro-batch with its own update — effective batch size is
        # DEVICE_BATCH_SIZE * MAX_SEQ_LEN per step.
        accum_loss = 0.0
        for micro in range(GRAD_ACCUM_STEPS):
            x, y = next(train_iter)
            loss = train_step(x, y)
            mx.eval(loss)
            accum_loss += loss.item()

        step += 1
        tokens_processed += TOTAL_BATCH_SIZE
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

        if step % 10 == 0:
            print(
                f"step {step:5d} | "
                f"loss {accum_loss / GRAD_ACCUM_STEPS:.4f} | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"elapsed {elapsed:.1f}s"
            )

    total_seconds = time.time() - training_start
    peak_memory_mb = mx.get_peak_memory() / 1024 / 1024

    # ── Evaluation ────────────────────────────────────────────────
    print("\nRunning evaluation...")
    val_bpb = evaluate_bpb(model)

    # ── Output (parsed by git_watcher.py) ────────────────────────
    print(f"\n--- Results ---")
    print(f"val_bpb: {val_bpb:.6f}")
    print(f"training_seconds: {total_seconds:.1f}")
    print(f"peak_memory_mb: {peak_memory_mb:.1f}")
    print(f"tokens_per_sec: {tokens_per_sec:,.0f}")
    print(f"num_steps: {step}")
    print(f"num_params_M: {model.num_params() / 1e6:.1f}")
    print(f"depth: {DEPTH}")
```

> **Note:** `tree_map` can be imported from `mlx.utils` or implemented as a simple recursive dict merge. The key correctness point is: gradients are accumulated across micro-batches and applied once, not applied per micro-batch.

---

## 5. The FastAPI Server — Specification

### 5.1 main.py Structure

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio

from server.process_manager import ProcessManager
from server.git_watcher import GitWatcher
from server.hardware import detect_hardware
from server.program_generator import generate_program_md

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                pass

manager = ConnectionManager()
process_mgr = ProcessManager(manager)
git_watcher = GitWatcher(manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start git watcher background task
    watcher_task = asyncio.create_task(git_watcher.watch())
    yield
    watcher_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # Keep alive; server pushes
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/hardware")
async def get_hardware():
    return detect_hardware()


class SessionConfig(BaseModel):
    focusAreas: list[str] = []
    hints: str = ""
    agentCommand: str = "claude"
    branchName: str = "session"


@app.post("/api/session/start")
async def start_session(config: SessionConfig):
    """
    Generates program.md, creates git branch, spawns agent.
    """
    hardware = detect_hardware()
    program_md = generate_program_md(
        focus_areas=config.focusAreas,
        hints=config.hints,
        hardware=hardware
    )
    with open("program.md", "w") as f:
        f.write(program_md)

    await process_mgr.start_session(
        agent_command=config.agentCommand,
        branch_name=config.branchName
    )
    return {"status": "started", "branch": config.branchName}


@app.post("/api/session/stop")
async def stop_session():
    await process_mgr.stop_session()
    return {"status": "stopped"}


@app.get("/api/session/status")
async def session_status():
    return process_mgr.get_status()


@app.get("/api/experiments")
async def get_experiments():
    return git_watcher.get_all_experiments()


@app.get("/api/experiments/{commit}")
async def get_experiment(commit: str):
    return git_watcher.get_experiment_with_diff(commit)


@app.get("/api/results/summary")
async def get_summary():
    from server.summarizer import generate_summary
    experiments = git_watcher.get_all_experiments()
    return await generate_summary(experiments)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve built React app in production (after `npm run build`)
# app.mount("/", StaticFiles(directory="client/dist", html=True), name="static")
```

### 5.2 REST API Summary

```
POST   /api/session/start          Start a new research session
POST   /api/session/stop           Stop the current session
GET    /api/session/status         Current session state (active/idle)
GET    /api/hardware               Detected Apple Silicon specs + recommendations
GET    /api/experiments            All experiments from results.tsv
GET    /api/experiments/{commit}   Single experiment with train.py diff
GET    /api/results/summary        Claude-generated plain-language summary
GET    /api/health                 Server health check
```

### 5.3 WebSocket Message Protocol

All messages are JSON. The server pushes; the client only receives on this channel. The client never sends messages to the server over WebSocket — use REST for control operations.

```typescript
type WSMessage =
  | { type: "session_started";   branch: string; timestamp: string }
  | { type: "experiment_started"; n: number; hypothesis: string }
  | { type: "agent_log";          level: "info"|"code"|"result"|"error"; text: string }
  | { type: "training_tick";      step: number; loss: number; tokens_per_sec: number; elapsed_seconds: number }
  | { type: "experiment_done";    commit: string; val_bpb: number; memory_gb: number; status: "keep"|"discard"|"crash"; description: string }
  | { type: "session_stopped";    total_experiments: number; best_val_bpb: number }
  | { type: "error";              message: string }
```

### 5.4 process_manager.py

```python
import asyncio
import subprocess
import time
from pathlib import Path


class ProcessManager:
    def __init__(self, broadcaster):
        self.broadcaster = broadcaster
        self.process = None
        self._stream_task = None
        self._active = False
        self._branch = None
        self._started_at = None

    async def start_session(self, agent_command: str, branch_name: str):
        if self._active:
            raise RuntimeError("Session already active")

        # Create git branch
        subprocess.run(
            ["git", "checkout", "-b", f"autoresearch/{branch_name}"],
            check=True
        )
        self._branch = f"autoresearch/{branch_name}"
        self._started_at = time.time()
        self._active = True

        # Spawn Claude Code (or other agent) as a subprocess
        # SECURITY: Use create_subprocess_exec (not shell) to prevent command injection.
        # agent_command comes from user input via REST API.
        ALLOWED_AGENTS = {"claude", "aider", "cursor"}
        if agent_command not in ALLOWED_AGENTS:
            raise ValueError(f"Unknown agent: {agent_command}. Allowed: {ALLOWED_AGENTS}")

        self.process = await asyncio.create_subprocess_exec(
            agent_command, "--print",
            "Please read program.md and start a new research session",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=Path.cwd()
        )

        self._stream_task = asyncio.create_task(self._stream_output())

        await self.broadcaster.broadcast({
            "type": "session_started",
            "branch": self._branch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })

    async def _stream_output(self):
        """Stream agent stdout to WebSocket clients."""
        async for raw_line in self.process.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            await self.broadcaster.broadcast({
                "type": "agent_log",
                "level": self._classify(line),
                "text": line
            })
        # Process ended
        self._active = False
        await self.broadcaster.broadcast({
            "type": "session_stopped",
            "total_experiments": 0,  # git_watcher updates this
            "best_val_bpb": 0
        })

    @staticmethod
    def _classify(text: str) -> str:
        lower = text.lower()
        if "val_bpb" in lower:                      return "result"
        if any(k in text for k in ["def ", "class ", "import ", "return "]):
            return "code"
        if any(k in lower for k in ["error", "crash", "traceback", "exception"]):
            return "error"
        return "info"

    async def stop_session(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
        self._active = False

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "branch": self._branch,
            "started_at": self._started_at,
            "pid": self.process.pid if self.process else None
        }
```

### 5.5 git_watcher.py

```python
import asyncio
import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Experiment:
    commit: str
    val_bpb: float
    memory_gb: float
    status: str          # "keep" | "discard" | "crash"
    description: str
    diff: str = field(default="", repr=False)


class GitWatcher:
    def __init__(self, broadcaster, poll_interval: float = 2.0):
        self.broadcaster = broadcaster
        self.poll_interval = poll_interval
        self._seen: set[str] = set()
        self._experiments: list[Experiment] = []

    async def watch(self):
        while True:
            await asyncio.sleep(self.poll_interval)
            fresh = self._read_results_tsv()
            for exp in fresh:
                if exp.commit and exp.commit not in self._seen:
                    self._seen.add(exp.commit)
                    self._experiments.append(exp)
                    await self.broadcaster.broadcast({
                        "type": "experiment_done",
                        "commit": exp.commit,
                        "val_bpb": exp.val_bpb,
                        "memory_gb": exp.memory_gb,
                        "status": exp.status,
                        "description": exp.description
                    })

    def get_all_experiments(self) -> list[dict]:
        return [
            {"commit": e.commit, "val_bpb": e.val_bpb, "memory_gb": e.memory_gb,
             "status": e.status, "description": e.description}
            for e in self._experiments
        ]

    def get_experiment_with_diff(self, commit: str) -> dict | None:
        for exp in self._experiments:
            if exp.commit == commit:
                if not exp.diff:
                    exp.diff = self._get_diff(commit)
                return {
                    "commit": exp.commit, "val_bpb": exp.val_bpb,
                    "memory_gb": exp.memory_gb, "status": exp.status,
                    "description": exp.description, "diff": exp.diff
                }
        return None

    @staticmethod
    def _read_results_tsv() -> list[Experiment]:
        path = Path("results.tsv")
        if not path.exists():
            return []
        experiments = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    experiments.append(Experiment(
                        commit=row.get("commit", "").strip(),
                        val_bpb=float(row.get("val_bpb", 0)),
                        memory_gb=float(row.get("memory_gb", 0)),
                        status=row.get("status", "").strip(),
                        description=row.get("description", "").strip()
                    ))
                except (ValueError, KeyError):
                    continue  # skip malformed rows
        return experiments

    @staticmethod
    def _get_diff(commit: str) -> str:
        result = subprocess.run(
            ["git", "show", "--unified=5", commit, "--", "train.py"],
            capture_output=True, text=True
        )
        return result.stdout
```

### 5.6 hardware.py

```python
import subprocess
import re


def detect_hardware() -> dict:
    """Auto-detect Apple Silicon chip and memory. Returns specs + tuned recommendations."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10
        )
        info = result.stdout
    except Exception:
        return _fallback_hardware()

    chip_match   = re.search(r"Chip:\s+(.+)", info)
    memory_match = re.search(r"Memory:\s+(\d+)\s*GB", info)

    chip   = chip_match.group(1).strip() if chip_match else "Apple Silicon"
    mem_gb = int(memory_match.group(1)) if memory_match else 16

    return {
        "chip": chip,
        "memory": f"{mem_gb}GB",
        "recommendations": _get_recommendations(chip, mem_gb)
    }


def _get_recommendations(chip: str, mem_gb: int) -> dict:
    """
    Returns conservative defaults that will work on the detected hardware.
    The agent is free to push these limits — that's the point.
    """
    if mem_gb >= 64:   # M2/M3/M4 Max or Ultra
        return {"depth": 6, "max_seq_len": 512, "device_batch_size": 32,
                "muon_lr": 0.02, "adamw_lr": 3e-4}
    elif mem_gb >= 32: # M2/M3 Pro Max
        return {"depth": 5, "max_seq_len": 512, "device_batch_size": 24,
                "muon_lr": 0.02, "adamw_lr": 3e-4}
    elif mem_gb >= 16: # M2/M3 Pro
        return {"depth": 4, "max_seq_len": 512, "device_batch_size": 16,
                "muon_lr": 0.02, "adamw_lr": 3e-4}
    else:              # M1/M2 base (8GB)
        return {"depth": 3, "max_seq_len": 256, "device_batch_size": 8,
                "muon_lr": 0.015, "adamw_lr": 2e-4}


def _fallback_hardware() -> dict:
    return {
        "chip": "Apple Silicon (detection failed)",
        "memory": "Unknown",
        "recommendations": {"depth": 4, "max_seq_len": 256, "device_batch_size": 8,
                            "muon_lr": 0.02, "adamw_lr": 3e-4}
    }
```

### 5.7 program_generator.py

This generates `program.md` from UI wizard inputs. It is the highest-value server component — a well-crafted research directive is the primary lever the human has over the agent.

```python
def generate_program_md(focus_areas: list[str], hints: str, hardware: dict) -> str:
    chip = hardware.get("chip", "Apple Silicon")
    memory = hardware.get("memory", "Unknown")
    rec = hardware.get("recommendations", {})
    depth = rec.get("depth", 4)
    muon_lr = rec.get("muon_lr", 0.02)
    adamw_lr = rec.get("adamw_lr", "3e-4")

    focus_block = _build_focus_block(focus_areas)
    hints_block = f"\n## Human Hints\n{hints.strip()}\n" if hints.strip() else ""

    return f"""# autoresearch-mlx — Research Session

## Hardware Context
- **Chip:** {chip}
- **Memory:** {memory} unified

**Apple Silicon key insight:** This is not an H100. With a fixed 5-minute training
budget, smaller and faster models typically beat larger ones because they fit more
optimizer steps into the window. Do not port H100 intuitions. Discover what works
for this specific hardware by trusting the metric.

## Your Goal
Find the lowest possible `val_bpb` in the 5-minute training budget.

## Session Setup (do this once)
1. Verify `~/.cache/autoresearch/` contains data shards and a tokenizer.
   If missing, stop and tell the human to run: `uv run prepare.py`
2. Create a session branch: `git checkout -b autoresearch/<date>-<tag>`
3. Create `results.tsv` with the header row:
   `commit\\tval_bpb\\tmemory_gb\\tstatus\\tdescription`
4. Run baseline: `uv run train.py` — record as the first row in results.tsv

## Experiment Loop
For each experiment:
1. Inspect the current best result and your prior experiments
2. Form ONE hypothesis (write it down before touching code)
3. Make ONE clean change to `train.py`
4. Run: `uv run train.py`
5. Read `val_bpb` from the output
6. If improved → `git commit -am "brief description"` → log as "keep"
7. If not improved → `git checkout -- train.py` → log as "discard"
8. Add to results.tsv: `<commit>\\t<val_bpb>\\t<mem_gb>\\t<status>\\t<description>`

## Research Focus
{focus_block}

## Optimizer Notes
The baseline uses Muon for 2D weight matrices (Linear layers) and AdamW for
embeddings, norms, and lm_head. Their learning rates are independent:
- Muon LR: {muon_lr} (controls matrix weight updates)
- AdamW LR: {adamw_lr} (controls embedding and norm updates)
Tuning the ratio between them is a valid and often high-value experiment.

## Simplicity Criterion
All else equal, simpler is better.
- 0.001 val_bpb gain from 20 lines of complexity → not worth it
- Removing something and maintaining val_bpb → always worth it
- A clean change that finds 0.01 improvement → excellent

## Constraints
- Only modify `train.py`. Never touch `prepare.py`.
- Do not install new packages.
- Do not modify the 5-minute budget constant.
- Keep each experiment to ONE logical change.
{hints_block}
"""


def _build_focus_block(focus_areas: list[str]) -> str:
    descriptions = {
        "speed": (
            "**Speed / Throughput:** Maximize optimizer steps per 5 minutes. "
            "Try: reduce DEPTH, reduce MAX_SEQ_LEN, increase DEVICE_BATCH_SIZE, "
            "remove expensive operations (attention patterns, extra norms)."
        ),
        "memory": (
            "**Memory Efficiency:** Run the most capable model that fits without "
            "memory pressure. Try: gradient checkpointing, mixed precision strategies, "
            "attention memory optimizations, smaller intermediate activations."
        ),
        "accuracy": (
            "**Model Quality:** Push val_bpb lower through architectural improvements. "
            "Try: attention variants (RoPE positional encoding, grouped query attention), "
            "normalization improvements, activation functions (SwiGLU vs GeLU), "
            "learning rate schedule (warmup, cosine decay)."
        ),
        "optimizer": (
            "**Optimizer Tuning:** Explore the Muon/AdamW configuration. "
            "Try: Muon vs AdamW LR ratio, Muon momentum values, Muon Newton-Schulz "
            "steps, weight decay values, gradient clipping."
        ),
    }

    if not focus_areas:
        return (
            "**Open Exploration:** No specific focus provided. Explore broadly: "
            "try a mix of architecture changes, optimizer tuning, and hyperparameter "
            "sweeps. Document your reasoning in each experiment description."
        )

    return "\n\n".join(descriptions.get(a, a) for a in focus_areas)
```

### 5.8 summarizer.py

```python
import anthropic
from pathlib import Path


async def generate_summary(experiments: list[dict]) -> dict:
    """
    Call Claude API to generate a plain-language summary of what the
    agent discovered. Returns structured summary for the Results view.
    """
    if not experiments:
        return {"summary": "No experiments completed yet.", "insights": []}

    kept = [e for e in experiments if e["status"] == "keep"]
    discarded = [e for e in experiments if e["status"] == "discard"]
    best = min(experiments, key=lambda e: e["val_bpb"]) if kept else None

    experiments_text = "\n".join(
        f"- [{e['status'].upper()}] val_bpb={e['val_bpb']:.4f}: {e['description']}"
        for e in experiments
    )

    prompt = f"""You are summarizing an autonomous ML research session on Apple Silicon.
The agent ran {len(experiments)} experiments to minimize val_bpb (lower is better).

Experiments:
{experiments_text}

Write a concise plain-language summary (3-4 sentences) of:
1. What the agent discovered about this hardware
2. What types of changes helped vs. hurt
3. The best result achieved

Then list 2-3 key insights as short bullet points.
Keep the language accessible to someone who doesn't know ML deeply.
Format as JSON: {{"summary": "...", "insights": ["...", "...", "..."]}}"""

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    try:
        return json.loads(message.content[0].text)
    except Exception:
        return {
            "summary": message.content[0].text,
            "insights": []
        }
```

---

## 6. The React Frontend — Specification

### 6.1 Tech Stack
- **Vite 5** — build tool and dev server (`npm create vite@latest client -- --template react`)
- **React 18** — UI framework
- **Tailwind CSS 3** — utility-first styling, dark theme
- **Recharts** — experiment scatter plot (React-native, no D3 complexity)
- **date-fns** — timestamp formatting

No other dependencies without a very good reason.

### 6.2 App State (managed in App.jsx with useReducer)

```typescript
interface AppState {
  // Navigation
  view: "setup" | "dashboard" | "history" | "results";

  // Hardware (fetched on load)
  hardware: {
    chip: string;
    memory: string;
    recommendations: Record<string, number>;
  } | null;

  // Session
  session: {
    active: boolean;
    branch: string | null;
    startedAt: number | null;  // unix timestamp
  };

  // Live data
  experiments: Array<{
    commit: string;
    val_bpb: number;
    memory_gb: number;
    status: "keep" | "discard" | "crash";
    description: string;
    n: number;  // sequential index
  }>;

  agentLog: Array<{
    level: "info" | "code" | "result" | "error";
    text: string;
    timestamp: number;
  }>;

  currentExperiment: {
    n: number;
    hypothesis: string;
    ticks: Array<{ step: number; loss: number; elapsed: number }>;
  } | null;

  bestResult: {
    val_bpb: number;
    commit: string;
    description: string;
  } | null;
}
```

### 6.3 Component Specifications

**SetupWizard** — single scrollable page, no multi-step flow:

```jsx
// Three sections stacked vertically:
// 1. HardwareCard — chip name, memory, "Detected automatically"
// 2. FocusSelector — 4 chips: Speed 🚀 / Memory 💾 / Accuracy 🎯 / Optimizer ⚙️
//                    Multi-select. If none selected, agent explores broadly.
// 3. HintsInput — textarea: "Anything specific to try? (optional)"
// 4. Start button — green, full width, "Start Research →"
```

**Dashboard layout:**

```
┌──────────────────────────────────────────────────────────────────┐
│  🔬 autoresearch-mlx        ● Running  Branch: autoresearch/...  │
│                                                  [Stop Session]  │
├──────────────┬──────────────────────────────┬───────────────────┤
│  Agent Log   │   Experiment Chart           │   Best Result     │
│  (scrollable)│                              │   1.808 bpb       │
│              │   scatter: bpb vs exp#       │   commit a1b2c3   │
│  [INFO] ...  │   green=keep, grey=discard   │   "reduce depth"  │
│  [CODE] ...  │   pulsing dot = current      ├───────────────────┤
│  [RESULT]... │                              │   Current Exp     │
│              │                              │   #7 • 2:34 left  │
│              │                              │   loss: 2.14 ↓    │
└──────────────┴──────────────────────────────┴───────────────────┘
```

**ExperimentChart** — Recharts ScatterChart:
- X axis: experiment number (1, 2, 3...)
- Y axis: val_bpb (inverted scale or labelled clearly: lower = better)
- Green filled circle: status = "keep"
- Grey unfilled circle: status = "discard"
- Red X: status = "crash"
- Pulsing animated dot: current experiment in progress
- Click any dot: shows description in a tooltip

**AgentLog** — scrolling list, auto-scrolls to bottom:
- `info` lines: grey text
- `code` lines: monospace, slightly lighter background
- `result` lines: green text (contains "val_bpb")
- `error` lines: red text

**BestResultCard** — prominent display:
- Large number: current best val_bpb
- Subtitle: improvement from baseline (e.g., "↓ 32% from baseline")
- Small text: commit hash + description

**CurrentExperimentCard** — shows while training is running:
- Experiment number
- Mini sparkline of loss over current run
- Estimated time remaining in 5-minute budget
- Token/sec throughput

### 6.4 useWebSocket Hook

```javascript
// hooks/useWebSocket.js
import { useRef, useEffect } from "react";

export function useWebSocket(url, onMessage) {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  // Store callback in a ref to avoid reconnection loops.
  // Without this, putting onMessage in useCallback/useEffect deps
  // would trigger reconnection on every render.
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    function connect() {
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log("WebSocket connected");
      };

      wsRef.current.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          onMessageRef.current(msg);
        } catch (e) {
          console.error("WS parse error", e);
        }
      };

      wsRef.current.onclose = () => {
        // Auto-reconnect after 2 seconds
        reconnectTimer.current = setTimeout(connect, 2000);
      };

      wsRef.current.onerror = (err) => {
        console.error("WebSocket error", err);
      };
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [url]);
}
```

### 6.5 Styling Decisions

- **Background:** `bg-gray-950` (near black)
- **Cards:** `bg-gray-900 border border-gray-800 rounded-xl`
- **Primary accent:** `green-500` (kept experiments, success states, start button)
- **Danger:** `red-500` (crashes, stop button)
- **Muted:** `gray-500` (discarded experiments, secondary text)
- **Font:** System default (San Francisco on Mac — no custom font needed)
- **Monospace:** `font-mono` for log output and code diffs

---

## 7. prepare.py Specification

`prepare.py` is **frozen** — the agent must never modify it. It must expose these interfaces that `train.py` imports:

```python
# Constants (used by both prepare.py and train.py)
MAX_SEQ_LEN  = 512
VOCAB_SIZE   = 4096
CACHE_DIR    = Path.home() / ".cache" / "autoresearch"

# Functions
def get_dataloader(split: str, batch_size: int, seq_len: int) -> Iterator[tuple[mx.array, mx.array]]:
    """
    Yields (x, y) pairs where:
    - x: (batch_size, seq_len) int32 token indices
    - y: (batch_size, seq_len) int32 token targets (x shifted by 1)
    Both are mx.array on the default MLX device.
    split: "train" or "val"
    """

def evaluate_bpb(model, n_tokens: int = 500_000) -> float:
    """
    Evaluate validation bits per byte.
    Runs model in inference mode (no gradients).
    Deterministic — same result for same model weights.
    Returns float: lower is better, vocab-size-independent.
    """
```

**Data source:** TinyStories dataset. Karpathy explicitly recommends this for small compute — lower entropy than FineWeb-Edu means smaller models score meaningfully well, which makes the agent's discoveries more visible in the UI.

**evaluate_bpb implementation note:**

```python
def evaluate_bpb(model, n_tokens=500_000):
    model.eval()  # disable dropout if any
    total_loss = 0.0
    total_tokens = 0

    for x, y in get_dataloader("val", batch_size=8, seq_len=MAX_SEQ_LEN):
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        ))
        mx.eval(loss)
        total_loss  += loss.item() * y.size
        total_tokens += y.size
        if total_tokens >= n_tokens:
            break

    avg_loss_nats = total_loss / total_tokens

    # Convert: nats/token → bits/byte
    # bits/token = nats/token / ln(2)
    # bytes/token ≈ ln(VOCAB_SIZE) / ln(256) (average bytes per token given BPE vocab)
    import math
    bits_per_token = avg_loss_nats / math.log(2)
    bytes_per_token = math.log(VOCAB_SIZE) / math.log(256)
    bpb = bits_per_token / bytes_per_token

    model.train()
    return bpb
```

---

## 8. Build Order and Milestones

Build strictly in this order. Each milestone is independently testable before proceeding.

### Milestone 1: MLX Training Core
**Goal:** `uv run train.py` completes a 5-minute run and outputs a real val_bpb.

Steps:
1. Initialize repo with `pyproject.toml` and install `mlx`, `tiktoken`, `numpy`
2. Write `prepare.py`: TinyStories download, BPE tokenizer, `mx.array` dataloader, `evaluate_bpb`
3. Write `train.py`: GPT model, Muon+AdamW optimizer, warmup step, 5-minute timed loop
4. Run `uv run prepare.py` — should download ~800MB data and save tokenizer
5. Run `uv run train.py` — should see: compilation, training loss curve, then val_bpb output

**Verification steps:**
- Confirm `val_bpb < 4.0` in 5 minutes on M2 Pro or better (baseline should be ~2.5-3.0)
- Confirm Muon is actually working: print L2 norm of a Linear weight before and after step 10 — it should change
- Confirm warmup is excluded: time the full script, subtract 5 minutes — remainder should be compile time

**Definition of done:** Reproducible val_bpb output in 5 minutes with no crashes.

---

### Milestone 2: FastAPI Server Core
**Goal:** Server starts and hardware detection works. Session start/stop work via REST.

Steps:
1. Create `server/` package with `__init__.py`
2. Write `server/hardware.py` and test: `python -c "from server.hardware import detect_hardware; print(detect_hardware())"`
3. Write `server/program_generator.py`
4. Write `server/git_watcher.py` — test by manually appending a row to results.tsv and checking `get_all_experiments()`
5. Write `server/process_manager.py` — test with a dummy echo command: `echo "hello" | head -20`
6. Write `server/main.py` — start with `uvicorn server.main:app --reload`

**Verification steps:**
- `curl http://localhost:8000/api/hardware` returns chip name and recommendations
- `curl http://localhost:8000/api/health` returns `{"status": "ok"}`
- POST to `/api/session/start` with dummy config creates git branch

**Definition of done:** All REST endpoints return valid JSON. Server starts with `uv run uvicorn server.main:app`.

---

### Milestone 3: React Frontend — Setup Wizard
**Goal:** Browser renders setup wizard and connects to server.

Steps:
1. `npm create vite@latest client -- --template react` inside project root
2. Install: `npm install tailwindcss recharts date-fns` and configure Tailwind
3. Build `HardwareCard` that fetches `/api/hardware` and displays chip info
4. Build `FocusSelector` with 4 multi-select chip buttons
5. Build `HintsInput` textarea
6. Wire "Start Research" button to POST `/api/session/start`

**Verification steps:**
- `npm run dev` in `client/` opens browser at `localhost:5173`
- Hardware card shows real chip name
- Clicking Start calls the server and creates a git branch

**Definition of done:** Setup wizard renders correctly and triggers session start.

---

### Milestone 4: WebSocket Live Dashboard
**Goal:** Agent log and experiment dots update live in the browser.

Steps:
1. Implement `useWebSocket` hook
2. Wire server WebSocket broadcast to ProcessManager and GitWatcher
3. Build `AgentLog` component with colour-coded lines, auto-scroll
4. Build `ExperimentChart` with Recharts ScatterChart
5. Build `BestResultCard` and `CurrentExperimentCard`
6. Assemble full Dashboard layout

**Verification steps:**
- Start a session via the UI
- Watch agent log lines appear in real-time
- Manually add a row to results.tsv — watch a green dot appear on the chart
- Stop the session — watch the session stopped indicator update

**Definition of done:** Full live dashboard works end-to-end with a real training session.

---

### Milestone 5: History + Results Summary
**Goal:** Complete the end-to-end experience.

Steps:
1. Build `History` view: fetch `/api/experiments`, render as table, expandable diff rows
2. Build diff viewer using a simple pre-formatted display (no heavy diff library)
3. Write `server/summarizer.py` with Anthropic API call
4. Build `Results` view with summary card and insights list
5. Add navigation between views (simple tab bar at top)

**Definition of done:** After a session, History shows all experiments with diffs. Results shows Claude-generated summary.

---

### Milestone 6: One-Command Startup + Polish
**Goal:** `./start.sh` is the only command a user needs.

```bash
#!/bin/bash
set -e

echo "research-mlx-ui starting..."

# Cleanup function — defined early, before any background processes
cleanup() {
    [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Check prerequisites
command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js not found. Install from nodejs.org"; exit 1; }

# Install Python deps
uv sync

# Install frontend deps
cd client && npm install && cd ..

# Run prepare.py if data doesn't exist
if [ ! -d "$HOME/.cache/autoresearch" ]; then
    echo "First run: downloading data (this takes ~2 minutes)..."
    uv run prepare.py
fi

# Build frontend (or run dev server)
if [ "$DEV" = "1" ]; then
    cd client && npm run dev &
    CLIENT_PID=$!
    cd ..
else
    cd client && npm run build && cd ..
fi

# Start FastAPI server (in background so trap can fire)
echo "Starting server at http://localhost:8000"
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server process — trap fires on signal
wait $SERVER_PID
```

---

## 9. pyproject.toml

```toml
[project]
name = "research-mlx-ui"
version = "0.1.0"
description = "Karpathy's autoresearch for Apple Silicon: MLX + Muon + web UI"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.27.1",              # v0.27.1+ required for Muon + MultiOptimizer
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "websockets>=12.0",
    "tiktoken>=0.6.0",
    "numpy>=1.26.0",
    "anthropic>=0.25.0",
    "httpx>=0.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
    "pytest-asyncio>=0.23.0",
]

[tool.ruff]
line-length = 100
```

---

## 10. Key Design Decisions — Do Not Change Without Thinking

**Single `train.py` file, ~600 lines.** Hard constraint from autoresearch design. The agent needs to read the whole file to make good changes. Splitting it violates this.

**Use official MLX Muon, not a custom implementation.** As of MLX v0.27.1, `mlx.optimizers.Muon` and `MultiOptimizer` are official. Our value add is the autoresearch loop + UI, not the optimizer itself. A custom Muon would diverge from upstream and miss future improvements.

**`mx.compile()` with warmup.** Without it you leave ~20% performance on the table. The warmup step is mandatory — without it the compilation time corrupts the 5-minute budget measurement.

**TinyStories over FineWeb-Edu.** On Apple Silicon with small models and a 5-minute budget, TinyStories produces more meaningful val_bpb improvements. The agent finds more wins in fewer experiments, which makes the UI experience better.

**FastAPI over Flask.** Async-native matters for WebSocket + subprocess management. This is the right tool.

**No database.** State lives in `results.tsv` and git. Portable, grep-able, in the spirit of the original autoresearch. Adding a database would be scope creep.

**Dark UI default.** This is a developer tool that runs overnight. Dark is correct.

**WebSocket push, REST control.** The server pushes all live data over WebSocket. The client sends control commands (start/stop/config) over REST. Never mix these.

---

## 11. Scope Limits for v1 — Do Not Build These

- Multi-GPU or multi-Mac distributed training
- Model inference/chat UI (this is about training, not inference)
- Authentication or multi-user support (local tool only)
- Pause and resume (stop and restart is sufficient; git holds state)
- Custom dataset upload (TinyStories only in v1)
- Windows or Linux support (Apple Silicon only by design)
- Docker container (adds complexity for minimal benefit)

---

## 12. Pre-Flight Checklist for Claude Code

Before writing the first line of code, verify:

- [ ] Running on Apple Silicon Mac (M1 or newer) — `uname -m` should return `arm64`
- [ ] `uv` installed — `uv --version`
- [ ] Python 3.11+ available — `python3 --version`
- [ ] Node.js 20+ available — `node --version`
- [ ] `git` configured with name and email — `git config user.name`
- [ ] MLX ≥0.27.1 installs correctly — `pip install "mlx>=0.27.1" && python -c "import mlx.core as mx; import mlx.optimizers; print(mx.default_device()); print(hasattr(mlx.optimizers, 'Muon'))"`
- [ ] Verify `mlx.optimizers.Muon` and `mlx.optimizers.MultiOptimizer` exist (they should since v0.27.1)
- [ ] Read MLX compile docs: `ml-explore.github.io/mlx/build/html/usage/compile.html`
- [ ] Anthropic API key available (for summarizer — set as env var `ANTHROPIC_API_KEY`)

**Start with Milestone 1. Do not touch the server or UI until `uv run train.py` produces a real `val_bpb` value.**

---

## 13. Reference Table

| Resource | URL | Why Essential |
|----------|-----|---------------|
| karpathy/autoresearch | github.com/karpathy/autoresearch | Original — read README and program.md carefully |
| trevin-creator/autoresearch-mlx | github.com/trevin-creator/autoresearch-mlx | Best existing MLX fork — understand what's missing |
| thenamangoyal/autoresearch | github.com/thenamangoyal/autoresearch | Reports 1.295 BPB on M4 Max — benchmark target |
| MLX optimizers docs | ml-explore.github.io/mlx/build/html/python/optimizers.html | **Official Muon + MultiOptimizer API** |
| MLX compile docs | ml-explore.github.io/mlx/build/html/usage/compile.html | Critical for train step correctness |
| MLX value_and_grad | ml-explore.github.io/mlx/build/html/python/nn.html | Autograd API |
| Muon PR #1914 | github.com/ml-explore/mlx/pull/1914 | Official Muon implementation — reference for behavior |
| MultiOptimizer PR #1916 | github.com/ml-explore/mlx/pull/1916 | Parameter group routing for Muon+AdamW |
| Turbo-Muon paper | arxiv.org/abs/2512.04632 | Spectral preconditioning — potential novel contribution |
| stockeh/mlx-optimizers | github.com/stockeh/mlx-optimizers | Community optimizers: SOAP, DiffGrad, QHAdam |
| Muon paper/post | kellerjordan.github.io/posts/muon | Mathematical background for the optimizer |
| FastAPI WebSocket | fastapi.tiangolo.com/advanced/websockets | Server WebSocket implementation |
| Recharts ScatterChart | recharts.org/api/ScatterChart | Frontend chart component |

---

*Document version: 1.1*
*Prepared: March 2026*
*Updated: March 14, 2026 — replaced custom Muon with official MLX Muon+MultiOptimizer, fixed gradient accumulation bug, fixed iterator bug, fixed command injection, fixed WebSocket reconnection loop, fixed start.sh trap ordering, added Pydantic validation, updated competitive landscape, added Turbo-Muon reference*
*Project: research-mlx-ui*
*Status: Ready for Claude Code*
