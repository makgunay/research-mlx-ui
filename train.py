"""
train.py — MLX GPT training with Muon + AdamW on Apple Silicon.

This file is the ONLY file the autoresearch agent modifies.
Hard constraint: keep under ~600 lines so the agent can reason about it holistically.
"""

import math
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from prepare import VOCAB_SIZE, MAX_SEQ_LEN, get_dataloader, evaluate_bpb

# ─── Model Architecture ─────────────────────────────────────────────────────
DEPTH = 4          # transformer layers (tuned for M3 Max 128GB)
N_HEAD = 8         # attention heads
N_EMBD = 768       # embedding dimension

# ─── Training ───────────────────────────────────────────────────────────────
DEVICE_BATCH_SIZE = 32     # sequences per forward pass
TOTAL_BATCH_SIZE = 2**14   # ~16K effective batch (gradient accumulation)
GRAD_ACCUM_STEPS = max(1, TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN))

# ─── Optimizer ──────────────────────────────────────────────────────────────
MUON_LR = 0.02
ADAMW_LR = 1e-3
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 50
TOTAL_STEPS = 480

# ─── Budget ─────────────────────────────────────────────────────────────────
TRAINING_BUDGET_SECONDS = 300  # 5 minutes wall clock, excludes warmup/compile


# ─── GPT Model ──────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = nn.RoPE(n_embd // n_head)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        head_dim = C // self.n_head
        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        causal_mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        attn = attn + causal_mask
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.c_proj(nn.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.ln_1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd)
        self.ln_2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, max_seq_len):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = [Block(n_head, n_embd) for _ in range(n_layer)]
        self.ln_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.wte.weight

    def __call__(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))


# ─── Optimizer Setup ─────────────────────────────────────────────────────────

def build_optimizer(muon_lr=0.02, adamw_lr=3e-4,
                    adamw_betas=(0.9, 0.95), weight_decay=0.1):
    """
    MultiOptimizer: Muon for 2D Linear weights, AdamW for everything else.
    Filter excludes embeddings and lm_head from Muon.
    Uses cosine decay with linear warmup for both optimizers.
    """
    muon_schedule = optim.join_schedules(
        [optim.linear_schedule(0, muon_lr, WARMUP_STEPS),
         optim.cosine_decay(muon_lr, TOTAL_STEPS - WARMUP_STEPS, 0.1 * muon_lr)],
        [WARMUP_STEPS]
    )
    adamw_schedule = optim.join_schedules(
        [optim.linear_schedule(0, adamw_lr, WARMUP_STEPS),
         optim.cosine_decay(adamw_lr, TOTAL_STEPS - WARMUP_STEPS, 0.1 * adamw_lr)],
        [WARMUP_STEPS]
    )
    muon = optim.Muon(learning_rate=muon_schedule, momentum=0.95, nesterov=True)
    adamw = optim.AdamW(
        learning_rate=adamw_schedule, betas=adamw_betas, weight_decay=weight_decay
    )
    return optim.MultiOptimizer(
        [muon, adamw],
        [lambda name, w: w.ndim >= 2
         and "wte" not in name
         and "lm_head" not in name]
    )


# ─── Training ───────────────────────────────────────────────────────────────

def main():
    print(f"MLX device: {mx.default_device()}")
    print(f"Config: depth={DEPTH}, n_head={N_HEAD}, n_embd={N_EMBD}")
    print(f"  seq_len={MAX_SEQ_LEN}, vocab={VOCAB_SIZE}")
    print(f"  batch={DEVICE_BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}")
    print(f"  muon_lr={MUON_LR}, adamw_lr={ADAMW_LR}, wd={WEIGHT_DECAY}")
    print(f"  budget={TRAINING_BUDGET_SECONDS}s")
    print()

    # Model
    model = GPT(
        vocab_size=VOCAB_SIZE,
        n_layer=DEPTH,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        max_seq_len=MAX_SEQ_LEN,
    )
    mx.eval(model.parameters())
    n_params = model.num_params()
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Optimizer
    optimizer = build_optimizer(
        muon_lr=MUON_LR, adamw_lr=ADAMW_LR, weight_decay=WEIGHT_DECAY
    )

    # Dataloader (persistent iterator)
    train_iter = iter(get_dataloader("train", DEVICE_BATCH_SIZE, MAX_SEQ_LEN))

    # Loss function
    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)
        ))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Compile the full train step
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    # ── Warmup (excluded from 5-minute budget) ──────────────────────────
    print("Compiling (warmup step)...")
    compile_start = time.time()
    x, y = next(train_iter)
    loss = train_step(x, y)
    mx.eval(loss)
    compile_time = time.time() - compile_start
    print(f"Compilation complete in {compile_time:.1f}s. Starting timed run...")
    print()

    # ── Timed training loop ─────────────────────────────────────────────
    training_start = time.time()
    step = 0
    tokens_processed = 0
    tokens_per_sec = 0

    while True:
        elapsed = time.time() - training_start
        if elapsed >= TRAINING_BUDGET_SECONDS:
            break

        accum_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = next(train_iter)
            loss = train_step(x, y)
            mx.eval(loss)
            accum_loss += loss.item()

        step += 1
        tokens_processed += DEVICE_BATCH_SIZE * MAX_SEQ_LEN * GRAD_ACCUM_STEPS
        elapsed = time.time() - training_start
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

        if step % 10 == 0 or step <= 3:
            avg_loss = accum_loss / GRAD_ACCUM_STEPS
            print(
                f"step {step:5d} | "
                f"loss {avg_loss:.4f} | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"elapsed {elapsed:.1f}s"
            )

    total_seconds = time.time() - training_start
    peak_memory_mb = mx.get_peak_memory() / 1024 / 1024

    # ── Evaluation ──────────────────────────────────────────────────────
    print()
    print("Running evaluation...")
    eval_start = time.time()
    val_bpb = evaluate_bpb(model)
    eval_time = time.time() - eval_start
    print(f"Evaluation complete in {eval_time:.1f}s")

    # ── Output (parsed by git_watcher.py) ───────────────────────────────
    print()
    print("--- Results ---")
    print(f"val_bpb: {val_bpb:.6f}")
    print(f"training_seconds: {total_seconds:.1f}")
    print(f"peak_memory_mb: {peak_memory_mb:.1f}")
    print(f"tokens_per_sec: {tokens_per_sec:,.0f}")
    print(f"num_steps: {step}")
    print(f"num_params_M: {n_params / 1e6:.1f}")
    print(f"depth: {DEPTH}")


if __name__ == "__main__":
    main()
