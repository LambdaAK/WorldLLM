# TinyGPT

A small decoder-only transformer trained from scratch to track object possession across multi-turn conversations. Given statements like "Alice has the ball" and "Alice gives the ball to Bob", it answers questions like "Who has the ball?" correctly.

## Run Modes (Choose One)

This project can be run in four different ways:

1. Train + CLI chat (`interact.py`): run the model directly in your terminal with no Redis or web server; best for quick model iteration.
2. Local web serving stack (FastAPI + Redis + worker + PostgreSQL): run API, queue, worker, and database as separate local services.
3. Docker Compose (API + Redis + worker + PostgreSQL containers): run the full serving stack with reproducible containerized setup.
4. Kubernetes on Minikube (namespace + deployments + services + StatefulSet): run the full stack with orchestration, scaling, and persistent DB volume.

## Prerequisites

- Python 3.11
- A trained checkpoint at `checkpoints/best.pt` for any inference mode (web/API/worker)
- Redis (serving modes 2/3/4)
- PostgreSQL (mode 2, unless using Docker/Kubernetes for modes 3/4)
- Docker Desktop (for modes 3/4)
- Minikube + kubectl (for mode 4)

## 1) Train + CLI Chat (No Redis)

### Install dependencies

```bash
cd /Users/alex/Desktop/TinyGPT
python -m pip install -r requirements.txt
```

### Generate training data

```bash
python data_generator.py --train 300000 --val 2000 --test 2000 --outdir data
```

### Train model

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

### Chat in terminal

```bash
python interact.py
python interact.py --checkpoint checkpoints/best.pt --temperature 0.5 --no-color
```

### Run tests

```bash
python run_examples.py   # full 22-test suite
python test_model.py     # quick smoke test
```

### Stop/Cleanup

Stop running commands with `Ctrl+C`.

Optional cleanup (generated training data + checkpoints):

```bash
rm -rf data checkpoints
```

## 2) Local Web App (FastAPI + Redis + Worker + PostgreSQL)

Run this when you want the browser UI (`/`) and streaming responses.

### Terminal 1: Start PostgreSQL

```bash
brew services start postgresql@16
```

### Terminal 2: Create database + user (first time only)

```bash
psql postgres <<'SQL'
CREATE USER tinygpt WITH PASSWORD 'tinygpt';
CREATE DATABASE tinygpt OWNER tinygpt;
SQL
```

### Terminal 3: Start Redis

```bash
brew services start redis
redis-cli ping
```

Expected output: `PONG`

### Terminal 4: Start worker

```bash
python worker.py --redis_url redis://127.0.0.1:6379/0
```

### Terminal 5: Start API/web server

```bash
DATABASE_URL=postgresql+asyncpg://tinygpt:tinygpt@127.0.0.1:5432/tinygpt \
python app.py --host 0.0.0.0 --port 8000 --redis_url redis://127.0.0.1:6379/0
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Optional load test (local stack)

```bash
python scripts/load_test.py --base-url http://127.0.0.1:8000 --users 10 --requests-per-user 10
```

### Stop/Cleanup

- In API and worker terminals: `Ctrl+C`
- Stop Redis service:

```bash
brew services stop redis
```

Optional: stop local PostgreSQL

```bash
brew services stop postgresql@16
```

## 3) Docker Compose

This runs PostgreSQL, Redis, API, and worker together in containers.

```bash
cd /Users/alex/Desktop/TinyGPT
docker compose up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Useful compose commands:

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f worker
docker compose down
```

### Stop/Cleanup

Stop containers:

```bash
docker compose down
```

Stop + remove volumes (clears Redis data in compose):

```bash
docker compose down -v
```

## 4) Kubernetes (Minikube)

### Start local cluster

```bash
cd /Users/alex/Desktop/TinyGPT
minikube start --driver=docker --cpus=3 --memory=4096
kubectl config use-context minikube
kubectl get nodes
```

### Build image inside Minikube Docker daemon

```bash
eval $(minikube docker-env)
DOCKER_BUILDKIT=0 docker build -t tinygpt:latest .
eval $(minikube docker-env -u)
```

### Deploy manifests

```bash
kubectl apply -k k8s/
kubectl get pods -n tinygpt
kubectl get svc -n tinygpt
```

Wait until `tinygpt-api`, `tinygpt-worker`, and `tinygpt-redis` are `1/1 Running`.
You should also see `tinygpt-postgres-0` running.

### Access API

```bash
kubectl port-forward -n tinygpt svc/tinygpt-api 8000:8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Scale worker replicas

```bash
kubectl scale -n tinygpt deploy/tinygpt-worker --replicas=2
```

### Optional ingress

```bash
minikube addons enable ingress
kubectl apply -f k8s/optional/ingress.yaml
```

Add this to `/etc/hosts`:

```text
<MINIKUBE_IP> tinygpt.local
```

Get `<MINIKUBE_IP>` with:

```bash
minikube ip
```

Then open [http://tinygpt.local](http://tinygpt.local).

### Teardown

```bash
kubectl delete -k k8s/
minikube stop
```

### Stop/Cleanup

Stop port-forward with `Ctrl+C`.

Delete only app resources:

```bash
kubectl delete -k k8s/
```

Stop cluster:

```bash
minikube stop
```

Delete cluster entirely:

```bash
minikube delete
```

## Load Test Example (100x100)

```bash
python scripts/load_test.py --base-url http://127.0.0.1:8000 --users 100 --requests-per-user 100
```

This reports latency percentiles, throughput, and average batch size from worker stats.

## Database Logging + Metrics

Authentication is removed. The database is used for inference logging and metrics only.

When `DATABASE_URL` is set, each `/chat` request is written to `request_logs` with:

- `request_id`
- `status` (`ok`, `timeout`, `worker_error`)
- generation settings (`temperature`, `top_k`, `max_tokens`)
- `token_events`
- `latency_ms`
- `created_at`

`GET /info` also includes an aggregated `database.request_logs` summary (totals, status counts, averages, latest timestamp).

Quick check:

```bash
curl -s http://127.0.0.1:8000/info
```

## Troubleshooting

### `No checkpoint found...`

You must train first (or copy an existing checkpoint to `checkpoints/best.pt`):

```bash
python data_generator.py --train 300000 --val 2000 --outdir data
python train.py
```

### `ImagePullBackOff` in Kubernetes

Image is not in Minikube's local Docker cache. Rebuild into Minikube and restart:

```bash
eval $(minikube docker-env)
DOCKER_BUILDKIT=0 docker build -t tinygpt:latest .
eval $(minikube docker-env -u)
kubectl rollout restart -n tinygpt deployment/tinygpt-api deployment/tinygpt-worker
kubectl get pods -n tinygpt
```

### `no space left on device` or Docker `input/output error`

Docker Desktop disk is full/corrupted. Free space, restart Docker Desktop, then retry build/start.

### Load test says `Connection refused`

API is not reachable on `127.0.0.1:8000`. Check:

```bash
curl http://127.0.0.1:8000/info
```

If on Kubernetes, make sure `kubectl port-forward -n tinygpt svc/tinygpt-api 8000:8000` is running.

### API says database is not configured

Set `DATABASE_URL` before starting API, for example:

```bash
export DATABASE_URL=postgresql+asyncpg://tinygpt:tinygpt@127.0.0.1:5432/tinygpt
```

## Full Cleanup (Everything)

If you want to fully reset local project runtime artifacts:

```bash
cd /Users/alex/Desktop/TinyGPT
docker compose down -v 2>/dev/null || true
kubectl delete -k k8s/ 2>/dev/null || true
minikube delete 2>/dev/null || true
brew services stop redis 2>/dev/null || true
rm -rf data checkpoints
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
