# TinyGPT

A small decoder-only transformer trained from scratch to track object possession across multi-turn conversations. Given statements like "Alice has the ball" and "Alice gives the ball to Bob", it answers questions like "Who has the ball?" correctly.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
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
python interact.py --checkpoint checkpoints/best.pt --temperature 0.5 --no-color
```

### 5. Run tests

```bash
python run_examples.py   # full 22-test suite
python test_model.py     # quick smoke test
```

## Redis Serving Mode (API + Worker)

Run the web app in production-style mode with a Redis queue and a separate inference worker.

### 1. Start Redis (Homebrew)

```bash
brew services start redis
redis-cli ping
```

### 2. Start the worker

```bash
python worker.py --redis_url redis://127.0.0.1:6379/0
```

### 3. Start the API/web server

```bash
python app.py --host 0.0.0.0 --port 8000 --redis_url redis://127.0.0.1:6379/0
```

Open `http://localhost:8000`.

### Optional: Docker Compose

```bash
docker compose up --build
```

## Example conversation

```
You: Alice has the ball and 5 apples. Bob has the key. Charlie has the clock and 3 oranges.

TinyGPT: Got it.

You: Alice gives the ball to Charlie. Bob gives the key to Alice. Alice gives 2 apples to Bob.

TinyGPT: Got it.

You: Charlie gives the clock to Bob.

TinyGPT: Got it.

You: Who has the ball?

TinyGPT: Charlie has the ball.

You: What does Alice have?

TinyGPT: three apples and the key.

You: How many apples does Bob have?

TinyGPT: two.

You: Bob gives the clock to Alice. Who has the clock now?

TinyGPT: Alice has the clock.

You: Who has the most oranges?

TinyGPT: Charlie.
```

Commands: `reset`/`clear` to clear history, `help` for help, `quit` to exit.

## Architecture

- GPT-style decoder-only transformer
- 12 layers, 384 embedding dim, 12 attention heads, 1536 FFN dim, 384 max seq len
- ~50.7M parameters
- Weight-tied token embeddings and output head

## Training

- 300,000 training examples (generate with `data_generator.py` before training)
- ~15 epochs to converge; stop with Ctrl+C when val_loss plateaus
- Data paths: `data/train.txt`, `data/val.txt` (configurable via `--train_path`, `--val_path`)

## A100 Optimizations

On CUDA GPUs, training uses FlashAttention, bfloat16, TF32, compiled kernels, fused AdamW, and other fast defaults.
