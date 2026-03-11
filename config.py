"""
Hyperparameters for TinyGPT.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 137
    max_seq_len: int = 256
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1


@dataclass
class TrainConfig:
    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 20
    grad_clip: float = 1.0
    save_dir: str = "checkpoints"
    device: str = "auto"
    num_workers: int = 4
    warmup_steps: int = 200
    min_lr_fraction: float = 0.1
