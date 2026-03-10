"""
Hyperparameters for WorldLLM.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 137
    max_seq_len: int = 256
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1


@dataclass
class TrainConfig:
    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 20
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 500
    save_dir: str = "checkpoints"
    device: str = "auto"
