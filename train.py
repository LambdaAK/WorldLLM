"""
Training script for WorldLLM.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 32 --lr 1e-4
"""

import argparse
import math
import os
import time
import torch
import torch.nn as nn
from config import ModelConfig, TrainConfig
from dataset import create_dataloader
from model import WorldLLM
from vocabulary import PAD_ID, VOCAB_SIZE


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def configure_a100_optimizations(device: torch.device):
    """Enable hardware-specific optimizations for A100 GPUs."""
    if device.type != "cuda":
        return

    # TF32 tensor cores: ~3x faster matmuls with negligible precision loss
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    print("A100 optimizations enabled: TF32, cuDNN benchmark, FlashAttention")


def masked_loss(logits, targets, loss_mask):
    """Compute cross-entropy only on positions where loss_mask == 1."""
    per_token_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=PAD_ID,
        reduction="none",
    )
    flat_mask = loss_mask.view(-1)
    masked = per_token_loss * flat_mask
    num_tokens = flat_mask.sum()
    if num_tokens == 0:
        return masked.sum(), 0
    return masked.sum(), num_tokens.item()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        for inputs, targets, loss_mask in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            logits = model(inputs)
            loss, n = masked_loss(logits, targets, loss_mask)
            total_loss += loss.item()
            total_tokens += n

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)
    model.train()
    return avg_loss, perplexity


def train(model_config: ModelConfig, train_config: TrainConfig):
    device = get_device(train_config.device)
    print(f"Device: {device}")

    configure_a100_optimizations(device)

    model_config.vocab_size = VOCAB_SIZE
    model = WorldLLM(model_config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    if device.type == "cuda":
        model = torch.compile(model)
        print("Model compiled with torch.compile()")

    train_loader = create_dataloader(
        train_config.train_path,
        max_seq_len=model_config.max_seq_len,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = create_dataloader(
        train_config.val_path,
        max_seq_len=model_config.max_seq_len,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    print(f"Train examples: {len(train_loader.dataset):,}")
    print(f"Val examples: {len(val_loader.dataset):,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        fused=(device.type == "cuda"),
    )

    steps_per_epoch = len(train_loader)
    global_step = 0

    def get_lr(step):
        """Linear warmup then cosine decay to min_lr."""
        if step < train_config.warmup_steps:
            return train_config.learning_rate * (step + 1) / train_config.warmup_steps
        decay_steps = step - train_config.warmup_steps
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * decay_steps / max(1, steps_per_epoch * 100)))
        min_lr = train_config.learning_rate * train_config.min_lr_fraction
        return min_lr + (train_config.learning_rate - min_lr) * cos_decay

    os.makedirs(train_config.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    use_amp = device.type == "cuda"

    print(f"Steps/epoch: {steps_per_epoch} | "
          f"Warmup: {train_config.warmup_steps} steps | "
          f"Peak LR: {train_config.learning_rate} | "
          f"Min LR: {train_config.learning_rate * train_config.min_lr_fraction}")

    epoch = 0
    while True:
        epoch += 1
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for inputs, targets, loss_mask in train_loader:
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(inputs)
                loss, n = masked_loss(logits, targets, loss_mask)
                if n > 0:
                    loss = loss / n
                else:
                    continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * n
            epoch_tokens += n
            global_step += 1

        elapsed = time.time() - t0
        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss, val_ppl = evaluate(model, val_loader, device)

        current_lr = get_lr(global_step)
        toks_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
        print(f"Epoch {epoch} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_ppl {val_ppl:.2f} | lr {current_lr:.2e} | "
              f"{elapsed:.1f}s | {toks_per_sec:.0f} tok/s")

        # Save latest checkpoint (save uncompiled state_dict)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        path = os.path.join(train_config.save_dir, "latest.pt")
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "config": model_config,
            "epoch": epoch,
            "val_loss": val_loss,
        }, path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(train_config.save_dir, "best.pt")
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "config": model_config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print(f"  -> new best model (val_loss {val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train WorldLLM")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainConfig()

    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        train_config.learning_rate = args.lr
    if args.embed_dim is not None:
        model_config.embed_dim = args.embed_dim
    if args.num_layers is not None:
        model_config.num_layers = args.num_layers
    if args.num_heads is not None:
        model_config.num_heads = args.num_heads
    if args.train_path is not None:
        train_config.train_path = args.train_path
    if args.val_path is not None:
        train_config.val_path = args.val_path
    if args.device is not None:
        train_config.device = args.device
    if args.num_workers is not None:
        train_config.num_workers = args.num_workers

    train(model_config, train_config)


if __name__ == "__main__":
    main()
