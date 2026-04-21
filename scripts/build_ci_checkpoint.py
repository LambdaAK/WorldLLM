#!/usr/bin/env python3
"""
Build a lightweight TinyGPT checkpoint for CI regression tests.

The production checkpoint is intentionally gitignored and large. CI uses this
helper to generate a much smaller, deterministic checkpoint that is sufficient
to exercise the serving stack and performance gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ModelConfig
from model import TinyGPT
from vocabulary import VOCAB_SIZE


DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 2
DEFAULT_FFN_DIM = 256
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_VAL_LOSS = 0.0
DEFAULT_EPOCH = 0
DEFAULT_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small CI checkpoint for TinyGPT")
    parser.add_argument("--output", required=True, help="Path to write the checkpoint")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Torch seed for deterministic weights")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--ffn-dim", type=int, default=DEFAULT_FFN_DIM)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--val-loss", type=float, default=DEFAULT_VAL_LOSS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=0.0,
    )
    model = TinyGPT(config)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": args.epoch,
        "val_loss": args.val_loss,
    }
    torch.save(checkpoint, output_path)
    print(f"Wrote CI checkpoint: {output_path}")
    print(
        "Tiny config: "
        f"embed_dim={config.embed_dim}, "
        f"num_heads={config.num_heads}, "
        f"num_layers={config.num_layers}, "
        f"ffn_dim={config.ffn_dim}, "
        f"max_seq_len={config.max_seq_len}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
