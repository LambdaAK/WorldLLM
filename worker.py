"""
TinyGPT inference worker.

Consumes requests from Redis, forms dynamic micro-batches, runs model inference,
and publishes token events back to per-request Redis channels.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import redis
import torch

from interact import list_checkpoints, load_model
from redis_protocol import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_TOKEN,
    REQUEST_QUEUE_KEY,
    WORKER_STATS_KEY,
    decode_request,
    encode_event,
    stream_channel,
)
from vocabulary import CLIENT_ID, EOS_ID, PAD_ID

DEFAULT_CHECKPOINT = "checkpoints/best.pt"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"


def _resolve_checkpoint(checkpoint: str | None, checkpoint_dir: str) -> str:
    if checkpoint and os.path.isfile(checkpoint):
        return checkpoint
    if os.path.isfile(DEFAULT_CHECKPOINT):
        return DEFAULT_CHECKPOINT
    if checkpoint:
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    pattern = os.path.join(checkpoint_dir, "*.pt")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if candidates:
        return candidates[0]
    checkpoints = list_checkpoints(checkpoint_dir)
    if checkpoints:
        return checkpoints[0][0]
    raise SystemExit("No checkpoint found.")


def _resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
    temp = max(float(temperature), 1e-5)
    logits = logits / temp

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[-1]] = float("-inf")

    probs = torch.nn.functional.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return int(token.item())


def _publish_done(rds: redis.Redis, request_id: str):
    channel = stream_channel(request_id)
    rds.publish(channel, encode_event(EVENT_DONE))


def _publish_error(rds: redis.Redis, request_id: str, message: str):
    channel = stream_channel(request_id)
    rds.publish(channel, encode_event(EVENT_ERROR, message=message))
    rds.publish(channel, encode_event(EVENT_DONE))


def _collect_batch(
    rds: redis.Redis,
    queue_key: str,
    *,
    max_batch_size: int,
    batch_timeout_ms: int,
) -> list[dict]:
    first = rds.blpop(queue_key, timeout=1)
    if first is None:
        return []

    _, raw = first
    raw_items = [raw]

    deadline = time.monotonic() + (max(1, batch_timeout_ms) / 1000.0)
    while len(raw_items) < max(1, max_batch_size) and time.monotonic() < deadline:
        item = rds.lpop(queue_key)
        if item is None:
            time.sleep(0.001)
            continue
        raw_items.append(item)

    requests = []
    for raw_item in raw_items:
        try:
            req = decode_request(raw_item)
        except Exception:
            continue
        request_id = str(req.get("request_id", "")).strip()
        token_ids = req.get("token_ids")
        if not request_id or not isinstance(token_ids, list) or len(token_ids) == 0:
            continue
        requests.append({
            "request_id": request_id,
            "token_ids": [int(t) for t in token_ids],
            "temperature": float(req.get("temperature", 0.1)),
            "top_k": int(req["top_k"]) if req.get("top_k") is not None else None,
            "max_tokens": max(1, int(req.get("max_tokens", 40))),
            "submitted_at_ms": int(req.get("submitted_at_ms", 0)),
        })
    return requests


def _update_stats(rds: redis.Redis, stats: dict):
    average_queue_wait = (
        stats["total_queue_wait_ms"] / stats["total_requests"]
        if stats["total_requests"] > 0
        else 0.0
    )
    rds.hset(
        WORKER_STATS_KEY,
        mapping={
            "status": "running",
            "total_requests": stats["total_requests"],
            "total_batches": stats["total_batches"],
            "last_batch_size": stats["last_batch_size"],
            "total_streamed_tokens": stats["total_streamed_tokens"],
            "avg_queue_wait_ms": round(average_queue_wait, 2),
            "last_error": stats["last_error"],
            "last_heartbeat_ms": int(time.time() * 1000),
        },
    )


def _process_batch(
    rds: redis.Redis,
    model,
    config,
    device: torch.device,
    batch: list[dict],
    stats: dict,
):
    states = []
    now_ms = int(time.time() * 1000)
    for req in batch:
        prompt = req["token_ids"][-config.max_seq_len :]
        if not prompt:
            _publish_done(rds, req["request_id"])
            continue
        submitted_at_ms = int(req.get("submitted_at_ms", 0))
        if submitted_at_ms > 0 and now_ms >= submitted_at_ms:
            stats["total_queue_wait_ms"] += (now_ms - submitted_at_ms)
        states.append({
            "request_id": req["request_id"],
            "tokens": prompt,
            "temperature": req["temperature"],
            "top_k": req["top_k"],
            "max_tokens": req["max_tokens"],
            "generated": 0,
        })

    if not states:
        return

    model.eval()
    with torch.no_grad():
        while states:
            active = []
            for state in states:
                if state["generated"] >= state["max_tokens"]:
                    _publish_done(rds, state["request_id"])
                    continue
                active.append(state)
            states = active
            if not states:
                break

            lengths = [len(state["tokens"]) for state in states]
            max_len = max(lengths)
            batch_input = torch.full(
                (len(states), max_len),
                PAD_ID,
                dtype=torch.long,
                device=device,
            )
            for row, state in enumerate(states):
                tokens = state["tokens"]
                batch_input[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)

            logits = model(batch_input)
            sampled = []
            for row, state in enumerate(states):
                row_logits = logits[row, lengths[row] - 1, :].float()
                sampled.append(_sample_next_token(row_logits, state["temperature"], state["top_k"]))

            next_states = []
            for state, token_id in zip(states, sampled):
                request_id = state["request_id"]
                if token_id in (EOS_ID, CLIENT_ID):
                    _publish_done(rds, request_id)
                    continue

                state["tokens"].append(token_id)
                if len(state["tokens"]) > config.max_seq_len:
                    state["tokens"] = state["tokens"][-config.max_seq_len :]

                state["generated"] += 1
                stats["total_streamed_tokens"] += 1
                rds.publish(stream_channel(request_id), encode_event(EVENT_TOKEN, token_id=token_id))

                if state["generated"] >= state["max_tokens"]:
                    _publish_done(rds, request_id)
                else:
                    next_states.append(state)

            states = next_states


def main():
    parser = argparse.ArgumentParser(description="TinyGPT Redis inference worker")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_timeout_ms", type=int, default=25)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument(
        "--redis_url",
        type=str,
        default=os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
        help="Redis URL, e.g. redis://127.0.0.1:6379/0",
    )
    parser.add_argument("--request_queue_key", type=str, default=REQUEST_QUEUE_KEY)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    model, config = load_model(checkpoint_path, device)

    redis_client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    redis_client.ping()

    stats = {
        "total_requests": 0,
        "total_batches": 0,
        "last_batch_size": 0,
        "total_streamed_tokens": 0,
        "total_queue_wait_ms": 0.0,
        "last_error": "",
    }
    _update_stats(redis_client, stats)

    print(f"Worker ready on {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Redis URL: {args.redis_url}")
    print(f"Request queue key: {args.request_queue_key}")
    print(f"Batching: timeout={args.batch_timeout_ms}ms max_batch_size={args.max_batch_size}")

    try:
        while True:
            batch = _collect_batch(
                redis_client,
                args.request_queue_key,
                max_batch_size=args.max_batch_size,
                batch_timeout_ms=args.batch_timeout_ms,
            )
            if not batch:
                _update_stats(redis_client, stats)
                continue

            stats["total_requests"] += len(batch)
            stats["total_batches"] += 1
            stats["last_batch_size"] = len(batch)

            try:
                _process_batch(redis_client, model, config, device, batch, stats)
                stats["last_error"] = ""
            except Exception as exc:
                stats["last_error"] = str(exc)
                for req in batch:
                    _publish_error(redis_client, req["request_id"], str(exc))

            _update_stats(redis_client, stats)
    except KeyboardInterrupt:
        pass
    finally:
        redis_client.hset(
            WORKER_STATS_KEY,
            mapping={
                "status": "stopped",
                "last_error": stats["last_error"],
                "last_heartbeat_ms": int(time.time() * 1000),
            },
        )
        redis_client.close()


if __name__ == "__main__":
    main()
