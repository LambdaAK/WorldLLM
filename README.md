# TinyGPT

A small decoder-only transformer trained from scratch to track object possession across multi-turn conversations. Given statements like "Alice has the ball" and "Alice gives the ball to Bob", it answers questions like "Who has the ball?" correctly.

## Quickstart

### 1. Install dependencies

```bash
pip install torch>=2.0
```

### 2. Generate training data

```bash
python data_generator.py --train 200000 --val 2000 --test 2000 --outdir data
```

### 3. Train

If you get a `libcuda.so cannot found` error from `torch.compile`, refresh the linker cache first:

```bash
sudo ldconfig
```

```bash
python train.py
```

This trains indefinitely with all A100 optimizations enabled by default (FlashAttention, bfloat16, TF32, `torch.compile`). The model converges in ~5 epochs (~3 minutes). Ctrl+C to stop. Best checkpoint is saved to `checkpoints/best.pt`.

To customize:

```bash
python train.py --epochs 10 --batch_size 512 --lr 1e-4
```

### 4. Chat with the model

```bash
python interact.py
```

### 5. Run tests

```bash
python test_model.py
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
- 8 layers, 256 embedding dim, 8 attention heads, 1024 FFN dim
- ~6.4M parameters
- Weight-tied token embeddings and output head

## A100 Optimizations

On CUDA GPUs, training uses FlashAttention, bfloat16, TF32, compiled kernels, fused AdamW, and other fast defaults.
