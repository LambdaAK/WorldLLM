# TinyGPT

A small decoder-only transformer trained from scratch to track object possession across multi-turn conversations. Given statements like "Alice has the ball" and "Alice gives the ball to Bob", it answers questions like "Who has the ball?" correctly.

## Quickstart

### 1. Install dependencies

```bash
pip install torch>=2.0
```

### 2. Generate training data

```bash
python data_generator.py --train 300000 --val 2000 --test 2000 --outdir data
```

### 3. Train

If you get a `libcuda.so cannot found` error from `torch.compile`, refresh the linker cache first:

```bash
sudo ldconfig
```

```bash
python train.py
```

Training runs indefinitely. Stop with Ctrl+C once val_loss has plateaued (typically ~15 epochs). Best checkpoint is saved to `checkpoints/best.pt`. A100 optimizations are enabled by default (FlashAttention, bfloat16, TF32, `torch.compile`).

To customize batch size or learning rate:

```bash
python train.py --batch_size 512 --lr 1e-4
```

### 4. Chat with the model

```bash
python interact.py
```

### 5. Run tests

```bash
python run_examples.py   # full 22-test suite
python test_model.py     # quick smoke test
```

## Example conversation

```
CLIENT:
Alice has the ball.

OUTPUT:
Got it.

CLIENT:
Bob has the key.

OUTPUT:
Got it.

CLIENT:
Alice gives the ball to Bob.

OUTPUT:
Got it.

CLIENT:
Who has the ball?

OUTPUT:
Bob has the ball.

CLIENT:
What does Bob have?

OUTPUT:
the ball and the key.

CLIENT:
How many things does Alice have?

OUTPUT:
none.
```

## Architecture

- GPT-style decoder-only transformer
- 12 layers, 384 embedding dim, 12 attention heads, 1536 FFN dim, 384 max seq len
- ~21.5M parameters
- Weight-tied token embeddings and output head

## Training

- 300,000 training examples (generate with `data_generator.py` before training)
- ~15 epochs to converge; stop with Ctrl+C when val_loss plateaus
- Data paths: `data/train.txt`, `data/val.txt` (configurable via `--train_path`, `--val_path`)

## A100 Optimizations

On CUDA GPUs, training uses FlashAttention, bfloat16, TF32, compiled kernels, fused AdamW, and other fast defaults.
