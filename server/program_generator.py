"""Generates program.md from UI wizard inputs."""


def generate_program_md(focus_areas: list[str], hints: str, hardware: dict) -> str:
    chip = hardware.get("chip", "Apple Silicon")
    memory = hardware.get("memory", "Unknown")
    rec = hardware.get("recommendations", {})
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
