"""
prepare.py — Data preparation for autoresearch-mlx.

Downloads TinyStories, trains a BPE tokenizer, tokenizes data into shards,
and provides get_dataloader() + evaluate_bpb() for train.py.

This file is FROZEN — the agent must never modify it.
"""

import math
import os
import json
import struct
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

# ─── Constants (imported by train.py) ────────────────────────────────────────

VOCAB_SIZE = 4096
MAX_SEQ_LEN = 512
CACHE_DIR = Path.home() / ".cache" / "autoresearch"
SHARD_SIZE = 1_000_000  # tokens per shard

# Special tokens
BOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


# ─── Download TinyStories ────────────────────────────────────────────────────

def _download_tinystories():
    """Download TinyStories dataset from HuggingFace."""
    data_dir = CACHE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.txt"
    val_path = data_dir / "val.txt"

    if train_path.exists() and val_path.exists():
        print("Data already downloaded.")
        return train_path, val_path

    print("Downloading TinyStories dataset...")
    from huggingface_hub import hf_hub_download

    # Download train and validation splits
    for split, out_path in [("train", train_path), ("validation", val_path)]:
        print(f"  Downloading {split} split...")
        # TinyStories is available as a dataset; download parquet and extract text
        from datasets import load_dataset
        try:
            ds = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)
        except Exception:
            # Fallback: try without trust_remote_code
            ds = load_dataset("roneneldan/TinyStories", split=split)

        with open(out_path, "w", encoding="utf-8") as f:
            for example in ds:
                text = example.get("text", "")
                if text.strip():
                    f.write(text.strip() + "\n<|endoftext|>\n")

        print(f"  Saved {split}: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return train_path, val_path


# ─── Tokenizer Training ─────────────────────────────────────────────────────

def _train_tokenizer(train_path: Path):
    """Train a BPE tokenizer with VOCAB_SIZE tokens on TinyStories."""
    tokenizer_path = CACHE_DIR / "tokenizer.json"

    if tokenizer_path.exists():
        print("Tokenizer already trained.")
        return _load_tokenizer(tokenizer_path)

    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"],
        show_progress=True,
        min_frequency=2,
    )

    tokenizer.train([str(train_path)], trainer)
    tokenizer.save(str(tokenizer_path))

    print(f"Tokenizer saved: {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def _load_tokenizer(tokenizer_path: Path):
    """Load a trained tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(tokenizer_path))


def get_tokenizer():
    """Load the trained tokenizer. Call after prepare() has been run."""
    tokenizer_path = CACHE_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run: uv run prepare.py"
        )
    return _load_tokenizer(tokenizer_path)


# ─── Tokenization & Sharding ────────────────────────────────────────────────

def _tokenize_and_shard(tokenizer, text_path: Path, split: str):
    """Tokenize text file and save as uint16 numpy shards."""
    shard_dir = CACHE_DIR / "shards" / split
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Check if shards already exist
    existing = list(shard_dir.glob("*.npy"))
    if existing:
        print(f"Shards already exist for {split} ({len(existing)} shards).")
        return

    print(f"Tokenizing {split} data...")
    all_tokens = []
    total_bytes = 0

    with open(text_path, "r", encoding="utf-8") as f:
        batch_texts = []
        for line in f:
            batch_texts.append(line)
            if len(batch_texts) >= 1000:
                for text in batch_texts:
                    total_bytes += len(text.encode("utf-8"))
                    encoded = tokenizer.encode(text)
                    all_tokens.extend(encoded.ids)
                batch_texts = []

        # Process remaining
        for text in batch_texts:
            total_bytes += len(text.encode("utf-8"))
            encoded = tokenizer.encode(text)
            all_tokens.extend(encoded.ids)

    tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  Total bytes: {total_bytes:,}")
    print(f"  Bytes per token: {total_bytes / len(tokens):.2f}")

    # Save bytes_per_token for BPB calculation
    meta_path = CACHE_DIR / f"{split}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "total_tokens": len(tokens),
            "total_bytes": total_bytes,
            "bytes_per_token": total_bytes / len(tokens),
        }, f)

    # Split into shards
    n_shards = max(1, len(tokens) // SHARD_SIZE)
    for i in range(n_shards):
        start = i * SHARD_SIZE
        end = min(start + SHARD_SIZE, len(tokens))
        shard = tokens[start:end]
        shard_path = shard_dir / f"shard_{i:04d}.npy"
        np.save(shard_path, shard)

    # Save any remaining tokens
    if n_shards * SHARD_SIZE < len(tokens):
        shard = tokens[n_shards * SHARD_SIZE:]
        shard_path = shard_dir / f"shard_{n_shards:04d}.npy"
        np.save(shard_path, shard)
        n_shards += 1

    print(f"  Saved {n_shards} shards to {shard_dir}")


# ─── Dataloader ──────────────────────────────────────────────────────────────

def get_dataloader(split: str, batch_size: int, seq_len: int):
    """
    Yields (x, y) pairs where:
    - x: (batch_size, seq_len) int32 token indices
    - y: (batch_size, seq_len) int32 token targets (x shifted by 1)
    Both are mx.array on the default MLX device.
    split: "train" or "val"

    Infinite iterator — loops over shards with shuffling.
    """
    shard_dir = CACHE_DIR / "shards" / split
    shard_paths = sorted(shard_dir.glob("*.npy"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found in {shard_dir}. Run: uv run prepare.py"
        )

    rng = np.random.default_rng(seed=42 if split == "val" else None)

    while True:
        # Shuffle shard order each epoch (deterministic for val)
        indices = rng.permutation(len(shard_paths))

        for idx in indices:
            data = np.load(shard_paths[idx]).astype(np.int32)

            if len(data) < (seq_len + 1) * batch_size:
                continue  # skip tiny shards

            # Random start positions within this shard
            n_sequences = len(data) // (seq_len + 1)
            n_batches = n_sequences // batch_size

            if n_batches == 0:
                continue

            # Shuffle positions within shard
            positions = rng.permutation(n_sequences)

            for b in range(n_batches):
                batch_pos = positions[b * batch_size : (b + 1) * batch_size]
                batch_data = np.stack([
                    data[p * (seq_len + 1) : p * (seq_len + 1) + seq_len + 1]
                    for p in batch_pos
                ])

                x = mx.array(batch_data[:, :-1])
                y = mx.array(batch_data[:, 1:])
                yield x, y


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_bpb(model, n_tokens: int = 500_000) -> float:
    """
    Evaluate validation bits per byte.
    Runs model in inference mode (no gradients).
    Deterministic — same result for same model weights.
    Returns float: lower is better, vocab-size-independent.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Load actual bytes_per_token from tokenization metadata
    meta_path = CACHE_DIR / "val_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        bytes_per_token = meta["bytes_per_token"]
    else:
        # Fallback approximation
        bytes_per_token = math.log(VOCAB_SIZE) / math.log(256)

    for x, y in get_dataloader("val", batch_size=8, seq_len=MAX_SEQ_LEN):
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        ))
        mx.eval(loss)
        n = y.size
        total_loss += loss.item() * n
        total_tokens += n
        if total_tokens >= n_tokens:
            break

    avg_loss_nats = total_loss / total_tokens

    # Convert: nats/token → bits/byte
    # bits/token = nats/token / ln(2)
    # bpb = bits/token / bytes_per_token
    bits_per_token = avg_loss_nats / math.log(2)
    bpb = bits_per_token / bytes_per_token

    model.train()
    return bpb


# ─── Main: Run Data Preparation ─────────────────────────────────────────────

def prepare():
    """Run the full data preparation pipeline."""
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print()

    # Step 1: Download
    train_path, val_path = _download_tinystories()
    print()

    # Step 2: Train tokenizer
    tokenizer = _train_tokenizer(train_path)
    print()

    # Step 3: Tokenize and shard
    _tokenize_and_shard(tokenizer, train_path, "train")
    print()
    _tokenize_and_shard(tokenizer, val_path, "val")
    print()

    # Summary
    train_shards = list((CACHE_DIR / "shards" / "train").glob("*.npy"))
    val_shards = list((CACHE_DIR / "shards" / "val").glob("*.npy"))
    print("=" * 60)
    print("Preparation complete!")
    print(f"  Train shards: {len(train_shards)}")
    print(f"  Val shards:   {len(val_shards)}")
    print(f"  Cache dir:    {CACHE_DIR}")
    print()
    print("Next step: uv run train.py")


if __name__ == "__main__":
    prepare()
