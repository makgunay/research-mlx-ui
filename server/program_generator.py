"""Generates program.md from UI wizard inputs, with cumulative session context."""

import csv
from pathlib import Path


def generate_program_md(focus_areas: list[str], hints: str, hardware: dict,
                        max_experiments: int = 15) -> str:
    chip = hardware.get("chip", "Apple Silicon")
    memory = hardware.get("memory", "Unknown")
    rec = hardware.get("recommendations", {})
    muon_lr = rec.get("muon_lr", 0.02)
    adamw_lr = rec.get("adamw_lr", 1e-3)

    focus_block = _build_focus_block(focus_areas)
    hints_block = f"\n## Human Hints\n{hints.strip()}\n" if hints.strip() else ""
    prior_block = _build_prior_context()
    strategy_block = _build_strategy(has_prior=prior_block != "", max_experiments=max_experiments)

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
{strategy_block}

## Session Setup (do this once)
1. Verify `~/.cache/autoresearch/` contains data shards and a tokenizer.
   If missing, stop and tell the human to run: `uv run prepare.py`
2. Create a session branch: `git checkout -b autoresearch/<date>-<tag>`
3. If `results.tsv` does not exist, create it with the header row:
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
9. After {max_experiments} experiments, STOP and print "Session complete."

## Research Focus
{focus_block}
{prior_block}

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
- Stop after {max_experiments} experiments.
{hints_block}
"""


def _build_prior_context() -> str:
    """Read results.tsv and build prior context for the agent."""
    path = Path("results.tsv")
    if not path.exists():
        return ""

    experiments = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                experiments.append({
                    "val_bpb": float(row.get("val_bpb", 0)),
                    "status": row.get("status", "").strip(),
                    "description": row.get("description", "").strip(),
                    "memory_gb": float(row.get("memory_gb", 0)),
                })
            except ValueError:
                continue

    if not experiments:
        return ""

    kept = [e for e in experiments if e["status"] == "keep"]
    discarded = [e for e in experiments if e["status"] == "discard"]
    best = min(kept, key=lambda e: e["val_bpb"]) if kept else None

    lines = ["\n## Prior Session Results"]
    lines.append(f"Previous sessions ran {len(experiments)} experiments.")
    if best:
        lines.append(f"**Current best: {best['val_bpb']:.4f} BPB** — {best['description']}")
    lines.append("")

    # Kept experiments (what worked)
    if kept:
        lines.append("### What worked (kept)")
        for e in sorted(kept, key=lambda x: x["val_bpb"]):
            lines.append(f"- **{e['val_bpb']:.4f}** — {e['description']}")
        lines.append("")

    # Discarded experiments (what failed — don't retry)
    if discarded:
        lines.append("### Known failures (do NOT retry these)")
        for e in discarded:
            lines.append(f"- {e['val_bpb']:.4f} — {e['description']}")
        lines.append("")

    return "\n".join(lines)


def _build_strategy(*, has_prior: bool, max_experiments: int) -> str:
    """Generate adaptive strategy instructions."""
    if not has_prior:
        return f"""
## Research Strategy
This is a fresh session. Explore broadly for the first 5 experiments, then
focus on the most promising direction. Run at most {max_experiments} experiments."""

    return f"""
## Research Strategy
This session builds on prior work (see Prior Session Results below).
1. **First 2-3 experiments:** Apply any proven improvements not yet in the
   current train.py. Establish a strong baseline quickly.
2. **Experiments 4-10:** Exploit the most promising direction. Try variations
   of what worked best.
3. **When plateaued** (3+ experiments with <0.5% BPB improvement): pivot to
   a fundamentally different approach. Don't keep tweaking the same knob.
4. **Never retry known failures** listed below unless you have a specific
   reason to believe the outcome will differ (e.g., combined with a new change).

Run at most {max_experiments} experiments, then stop."""


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
