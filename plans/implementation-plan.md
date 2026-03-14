# research-mlx-ui — Implementation Plan

> Based on HANDOFF.md v1.1 (March 14, 2026)

---

## Phase 0: Pre-flight & API Validation (Do First)

**Goal:** Confirm the environment works and validate key assumptions before writing any project code.

### 0.1 Environment checks ✅ PASSED (March 14, 2026)
- [x] `uname -m` → `arm64` ✅
- [x] `uv --version` → 0.10.10 ✅
- [x] `python3 --version` → 3.12.2 ✅
- [x] `node --version` → v25.8.1 ✅
- [x] `git config user.name` → configured ✅
- [x] **Hardware: Apple M3 Max, 128 GB** — top tier, use depth=6+ defaults

### 0.2 MLX API validation (critical — blocks everything)
Before building anything, validate that official MLX Muon + MultiOptimizer work as the handoff assumes:

```python
# Run this as a standalone script to verify API
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# 1. Confirm Muon exists
muon = optim.Muon(learning_rate=0.02)

# 2. Confirm MultiOptimizer exists and accepts filter lambdas
adamw = optim.AdamW(learning_rate=3e-4)
multi = optim.MultiOptimizer(
    [muon, adamw],
    [lambda name, w: w.ndim >= 2]
)

# 3. Test with a tiny model
model = nn.Linear(8, 4)
x = mx.random.normal((2, 8))
y = mx.random.normal((2, 4))

def loss_fn(model, x, y):
    return mx.mean((model(x) - y) ** 2)

loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, x, y)
multi.update(model, grads)
mx.eval(model.parameters(), multi.state)
print(f"Loss: {loss.item():.4f} — MultiOptimizer works!")
```

**If this fails:** We need to read the actual MLX source to understand the correct MultiOptimizer API before proceeding. This is the single biggest risk.

### 0.3 Install dependencies ✅ PASSED
```bash
uv sync  # MLX 0.31.1 installed
```

### 0.4 API Validation Results ✅ PASSED

**Findings from validation:**
1. `mlx.optimizers.Muon` + `MultiOptimizer` work exactly as expected
2. **`mx.compile` pattern must be different than handoff assumed:**
   - WRONG: `mx.compile(loss_and_grad_fn, inputs=model.trainable_parameters())`
   - CORRECT: `@partial(mx.compile, inputs=state, outputs=state)` wrapping the full train step
3. `mx.metal.get_peak_memory()` is deprecated → use `mx.get_peak_memory()` instead
4. Loss went from 5.2 → 0.046 in 100 compiled steps (1.4s) — optimizer works

**HANDOFF.md updated with these corrections.**

**Phase 0 complete. Proceeding to Phase 1.**

---

## Phase 1: Milestone 1 — MLX Training Core

**Goal:** `uv run train.py` completes a 5-minute run and outputs a real val_bpb.

This is the hardest phase. Two files, strict ordering: prepare.py first, then train.py.

### 1.1 Research before coding
- [ ] Study karpathy/autoresearch `prepare.py` to understand TinyStories download + tokenization approach
- [ ] Study scasella/nanochat-mlx for training loop patterns
- [ ] Read MLX compile docs to understand `mx.compile()` constraints
- [ ] Decide: use tiktoken with a pre-existing encoding, or train a custom BPE? (Handoff says tiktoken + 4096 vocab — need to verify tiktoken supports custom vocab training or if we use a pre-trained encoding)

### 1.2 Write `prepare.py`
Order of implementation within the file:
1. **Constants:** `VOCAB_SIZE=4096`, `MAX_SEQ_LEN=512`, `CACHE_DIR`
2. **Data download:** TinyStories from HuggingFace, save raw text
3. **Tokenizer:** Train or load BPE tokenizer (tiktoken), save to cache
4. **Tokenization:** Process raw text → token shards as `.npy` files
5. **`get_dataloader()`:** Yields `(x, y)` mx.array pairs from shards, infinite iterator with shuffling
6. **`evaluate_bpb()`:** Validation loss → bits per byte conversion

**Validate:** `uv run prepare.py` downloads data, creates `~/.cache/autoresearch/` with shards + tokenizer.

### 1.3 Write `train.py`
Order of implementation within the file (~600 lines):
1. **Imports + hyperparameter constants** (~30 lines)
2. **GPT model classes:** CausalSelfAttention, MLP, Block, GPT (~80 lines)
3. **`build_optimizer()`** using official MultiOptimizer (~20 lines)
4. **`main()` function:** model init, warmup, timed training loop, eval, output (~100 lines)
5. **`if __name__ == "__main__": main()`**

Key correctness points:
- Persistent iterator (not `next(iter(...))` in loop)
- Gradient accumulation: accumulate then apply once
- Single `mx.eval()` per outer step
- Warmup excluded from 5-minute timer
- Output format parseable by git_watcher.py

### 1.4 Validate
- [ ] `uv run train.py` runs without crash
- [ ] Training loss decreases over steps
- [ ] `val_bpb < 4.0` after 5 minutes
- [ ] Peak memory reasonable for hardware
- [ ] Warmup time excluded from budget (total script time ≈ 5min + compile time)

**Estimated time: This is the bulk of the work. Gate: must produce real val_bpb before Phase 2.**

---

## Phase 2: Milestone 2 — FastAPI Server

**Goal:** Server starts, hardware detection works, session start/stop via REST.

### 2.1 Build server files (in order)
1. `server/hardware.py` — system_profiler parsing, recommendations
2. `server/program_generator.py` — program.md template from wizard inputs
3. `server/git_watcher.py` — results.tsv polling, experiment dataclass
4. `server/process_manager.py` — subprocess exec with allowlist, stream output
5. `server/main.py` — FastAPI app, routes, WebSocket, CORS, Pydantic models

### 2.2 Validate
```bash
uv run uvicorn server.main:app --reload
curl http://localhost:8000/api/health        # → {"status": "ok"}
curl http://localhost:8000/api/hardware      # → chip, memory, recommendations
curl -X POST http://localhost:8000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"focusAreas": ["speed"], "hints": "", "branchName": "test"}'
```

**Estimated time: Moderate. Mostly transcribing from handoff with minor adjustments.**

---

## Phase 3: Milestones 3+4 — React Frontend

**Goal:** Setup wizard renders, dashboard shows live data.

### 3.1 Scaffold React app
```bash
cd client
npm create vite@latest . -- --template react
npm install tailwindcss @tailwindcss/vite recharts date-fns
```

### 3.2 Build Setup Wizard (Milestone 3)
Use `/frontend-design` skill for high-quality dark UI.
1. `App.jsx` — useReducer state management, view routing
2. `SetupWizard/HardwareCard.jsx` — fetch /api/hardware, display chip info
3. `SetupWizard/FocusSelector.jsx` — 4 multi-select chips
4. `SetupWizard/HintsInput.jsx` — optional textarea
5. `SetupWizard/index.jsx` — compose + start button

### 3.3 Build Dashboard (Milestone 4)
1. `hooks/useWebSocket.js` — ref-based callback pattern
2. `Dashboard/AgentLog.jsx` — color-coded, auto-scroll
3. `Dashboard/ExperimentChart.jsx` — Recharts scatter plot
4. `Dashboard/BestResultCard.jsx` — prominent metric display
5. `Dashboard/CurrentExperiment.jsx` — live training stats
6. `Dashboard/index.jsx` — 3-column layout

### 3.4 Validate
- Setup wizard renders at localhost:5173
- Hardware card shows real chip
- Start button triggers session
- Agent log streams in real-time
- Experiment dots appear on chart
- Stop button works

**Estimated time: Moderate. Use `/frontend-design` for quality.**

---

## Phase 4: Milestone 5 — History + Results Summary

**Goal:** Complete the end-to-end experience.

### 4.1 Build remaining views
1. `History/index.jsx` — experiment table with expandable diffs
2. `History/DiffViewer.jsx` — pre-formatted diff display
3. `Results/SummaryCard.jsx` — Claude-generated summary
4. `Results/index.jsx` — summary + insights list
5. Navigation tab bar in App.jsx

### 4.2 Build summarizer
Use `/claude-api` skill for correct Anthropic SDK usage.
1. `server/summarizer.py` — Claude API call, structured JSON output

### 4.3 Validate
- After a session, History shows all experiments
- Clicking an experiment shows the train.py diff
- Results view shows Claude-generated summary

---

## Phase 5: Milestone 6 — Polish & Startup

**Goal:** `./start.sh` is the only command needed.

### 5.1 Final items
1. Write `start.sh` with trap cleanup
2. Write README.md
3. Run `/simplify` on all changed code
4. Run `/claude-md-management:revise-claude-md` to capture learnings
5. Initial commit + push

### 5.2 End-to-end test
- Fresh clone → `./start.sh` → browser opens → start session → experiments run → results show

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| MultiOptimizer API differs from handoff | Blocks Phase 1 | Validate in Phase 0.2 before writing any code |
| TinyStories download changes/breaks | Blocks Phase 1 | Check HuggingFace availability, have fallback URLs |
| tiktoken can't train custom 4096 vocab | Blocks Phase 1 | May need sentencepiece or a pre-trained encoding |
| `mx.compile()` breaks with MultiOptimizer | Degrades perf | Can fall back to uncompiled — 20% slower but functional |
| 5-min budget too short on base M1/M2 (8GB) | Poor UX | Hardware detection adjusts defaults (depth=3, seq=256) |
| Muon merged into autoresearch upstream | Reduces differentiation | UI remains our primary differentiator |

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package manager | uv | User preference, fast, handles venv+lockfile |
| Muon implementation | Official MLX (not custom) | Merged in v0.27.1, maintained upstream |
| Primary differentiator | UI layer | No fork has a UI; Muon is no longer novel |
| Repo name | research-mlx-ui | Already created on GitHub |
| Build order | ML core → server → frontend | Handoff-prescribed; validate hardest part first |

---

*Plan version: 1.0*
*Created: March 14, 2026*
