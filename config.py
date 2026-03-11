"""
Hyperparameters for TinyGPT.

ModelConfig controls the transformer architecture (layer count, dimensions, etc.).
TrainConfig controls the training loop (paths, optimizer, schedule, etc.).
Both are serialized into checkpoints so trained models are self-describing.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the decoder-only transformer."""

    vocab_size: int = 137          # set at runtime from vocabulary.VOCAB_SIZE
    max_seq_len: int = 384         # absolute positional embedding table size
    embed_dim: int = 384           # token + position embedding dimension
    num_heads: int = 12            # must evenly divide embed_dim
    num_layers: int = 12           # number of TransformerBlock layers
    ffn_dim: int = 1536            # inner dimension of the two-layer FFN
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training loop hyperparameters and I/O paths."""

    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    batch_size: int = 256
    learning_rate: float = 3e-4    # peak LR after warmup
    weight_decay: float = 0.1
    epochs: int = 20               # unused — train.py loops indefinitely (Ctrl+C to stop)
    grad_clip: float = 1.0         # max gradient norm; 0 disables clipping
    save_dir: str = "checkpoints"
    device: str = "auto"           # "auto" picks cuda > mps > cpu
    num_workers: int = 4           # DataLoader worker processes
    warmup_steps: int = 200        # linear LR warmup before cosine decay
    min_lr_fraction: float = 0.1   # cosine decay floor as fraction of peak LR
