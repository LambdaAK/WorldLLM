"""
FastAPI web UI + API gateway for TinyGPT.

This process handles:
- request validation
- enqueueing generation jobs into Redis
- streaming worker output back to the browser over SSE
- optional database-backed request logging and metrics

Inference runs in `worker.py`.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, select

from db import (
    RequestLog,
    close_db,
    get_database_url,
    get_session_factory,
    init_db,
)
from interact import build_conversation_tokens
from redis_protocol import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_TOKEN,
    REQUEST_QUEUE_KEY,
    WORKER_STATS_KEY,
    decode_event,
    encode_request,
    stream_channel,
)
from vocabulary import EOS_ID, ID_TO_WORD, PAD_ID, SOS_ID

_PUNCT_NO_SPACE = frozenset(".?,!")
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
DEFAULT_MAX_TOKENS = 40
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"

_redis_client: redis.Redis | None = None
_checkpoint_path: str | None = None
_info_payload: dict[str, Any] = {}
_redis_url = DEFAULT_REDIS_URL
_request_queue_key = REQUEST_QUEUE_KEY
_stream_idle_timeout_sec = 60.0

_db_enabled = False
_db_url_display = ""

logger = logging.getLogger("tinygpt.api")


class ChatRequest(BaseModel):
    messages: list[dict]  # [{role: "user"|"assistant", content: str}]
    temperature: float = 0.1
    top_k: int = 5
    max_tokens: int = DEFAULT_MAX_TOKENS


def _messages_to_turns(messages: list[dict]) -> tuple[list[tuple], str]:
    """Convert [{role, content}] to (turns, current_msg)."""
    turns = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            turns.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    current_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
    return turns, current_msg


def _token_to_word(token_id: int) -> str | None:
    """Convert token ID to displayable word; hide special tokens."""
    if token_id in (PAD_ID, SOS_ID, EOS_ID):
        return None
    return ID_TO_WORD.get(token_id, "<unk>")


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

    raise SystemExit(
        "No checkpoint found. Train a model first:\n"
        "  python data_generator.py --train 300000 --val 2000 --outdir data\n"
        "  python train.py\n"
        "Then run API + worker."
    )


def _load_checkpoint_info(checkpoint_path: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint": os.path.basename(checkpoint_path),
        "epoch": "?",
        "val_loss": "?",
        "parameters": 0,
        "device": "worker",
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["epoch"] = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    if isinstance(val_loss, float):
        val_loss = round(val_loss, 4)
    payload["val_loss"] = val_loss
    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict):
        payload["parameters"] = int(sum(param.numel() for param in state_dict.values()))
    return payload


def _coerce_worker_stats(stats: dict) -> dict:
    parsed: dict[str, Any] = {}
    for key, value in stats.items():
        if value is None:
            continue
        if isinstance(value, str):
            lowered = value.lower()
            if lowered.isdigit():
                parsed[key] = int(lowered)
                continue
            try:
                parsed[key] = float(value)
                continue
            except ValueError:
                parsed[key] = value
                continue
        parsed[key] = value
    return parsed


def _safe_top_k(top_k: int | None) -> int:
    if top_k is None:
        return 0
    return max(0, int(top_k))


async def _database_metrics() -> dict[str, Any]:
    if not _db_enabled:
        return {
            "enabled": False,
            "url": _db_url_display,
            "request_logs": None,
        }

    metrics: dict[str, Any] = {
        "enabled": True,
        "url": _db_url_display,
        "request_logs": {
            "total": 0,
            "ok": 0,
            "timeout": 0,
            "worker_error": 0,
            "avg_latency_ms": 0.0,
            "avg_token_events": 0.0,
            "last_request_at": None,
        },
    }

    try:
        async with get_session_factory()() as db_session:
            total = (await db_session.execute(select(func.count(RequestLog.id)))).scalar_one() or 0
            ok = (
                await db_session.execute(
                    select(func.count(RequestLog.id)).where(RequestLog.status == "ok")
                )
            ).scalar_one() or 0
            timeout = (
                await db_session.execute(
                    select(func.count(RequestLog.id)).where(RequestLog.status == "timeout")
                )
            ).scalar_one() or 0
            worker_error = (
                await db_session.execute(
                    select(func.count(RequestLog.id)).where(RequestLog.status == "worker_error")
                )
            ).scalar_one() or 0
            avg_latency = (await db_session.execute(select(func.avg(RequestLog.latency_ms)))).scalar_one()
            avg_tokens = (await db_session.execute(select(func.avg(RequestLog.token_events)))).scalar_one()
            last_request_at = (
                await db_session.execute(
                    select(RequestLog.created_at).order_by(RequestLog.created_at.desc()).limit(1)
                )
            ).scalar_one_or_none()

        metrics["request_logs"] = {
            "total": int(total),
            "ok": int(ok),
            "timeout": int(timeout),
            "worker_error": int(worker_error),
            "avg_latency_ms": round(float(avg_latency or 0.0), 2),
            "avg_token_events": round(float(avg_tokens or 0.0), 2),
            "last_request_at": last_request_at.isoformat() if last_request_at is not None else None,
        }
    except Exception as exc:
        metrics["request_logs"] = {"error": str(exc)}

    return metrics


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _redis_client, _db_enabled, _db_url_display

    _redis_client = redis.from_url(_redis_url, decode_responses=True)
    await _redis_client.ping()

    _db_enabled = await init_db(os.getenv("DATABASE_URL"))
    db_url = get_database_url()
    _db_url_display = db_url or "disabled"

    try:
        yield
    finally:
        await close_db()
        if _redis_client is not None:
            await _redis_client.aclose()
            _redis_client = None


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html")) as f:
        return f.read()


@app.post("/chat")
async def chat(req: ChatRequest):
    if _redis_client is None:
        return JSONResponse({"error": "Redis client not ready"}, status_code=503)

    if not req.messages or req.messages[-1]["role"] != "user":
        return JSONResponse({"error": "Last message must be from user"}, status_code=400)

    turns, current_msg = _messages_to_turns(req.messages)
    if not current_msg.strip():
        return JSONResponse({"error": "Empty message"}, status_code=400)

    token_ids = build_conversation_tokens(turns, current_msg)
    request_id = uuid.uuid4().hex
    channel = stream_channel(request_id)

    pubsub = _redis_client.pubsub()
    await pubsub.subscribe(channel)

    request_payload = {
        "request_id": request_id,
        "token_ids": token_ids,
        "temperature": float(req.temperature),
        "top_k": int(req.top_k) if req.top_k is not None else None,
        "max_tokens": max(1, int(req.max_tokens)),
        "submitted_at_ms": int(time.time() * 1000),
    }

    try:
        await _redis_client.rpush(_request_queue_key, encode_request(request_payload))
    except Exception:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        return JSONResponse({"error": "Failed to enqueue request"}, status_code=503)

    async def event_stream():
        first = True
        last_event_at = time.monotonic()
        request_started = time.perf_counter()
        token_events = 0
        status_text = "ok"

        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    if time.monotonic() - last_event_at > _stream_idle_timeout_sec:
                        status_text = "timeout"
                        yield "data: [ERROR] generation timeout\n\n"
                        break
                    continue

                last_event_at = time.monotonic()
                raw_data = message.get("data")
                if not isinstance(raw_data, str):
                    continue

                try:
                    event = decode_event(raw_data)
                except Exception:
                    continue

                event_type = event.get("type")
                if event_type == EVENT_TOKEN:
                    token_id = event.get("token_id")
                    if token_id is None:
                        continue
                    word = _token_to_word(int(token_id))
                    if word is None:
                        continue
                    prefix = "" if first or word in _PUNCT_NO_SPACE else " "
                    chunk = f"{prefix}{word}"
                    token_events += 1
                    yield f"data: {chunk}\n\n"
                    first = False
                elif event_type == EVENT_ERROR:
                    status_text = "worker_error"
                    error_message = str(event.get("message", "worker error")).replace("\n", " ").strip()
                    yield f"data: [ERROR] {error_message}\n\n"
                elif event_type == EVENT_DONE:
                    break
        finally:
            try:
                await pubsub.unsubscribe(channel)
            finally:
                await pubsub.aclose()

            if _db_enabled:
                latency_ms = (time.perf_counter() - request_started) * 1000.0
                try:
                    async with get_session_factory()() as db_session:
                        db_session.add(
                            RequestLog(
                                request_id=request_id,
                                status=status_text,
                                temperature=float(req.temperature),
                                top_k=_safe_top_k(req.top_k),
                                max_tokens=max(1, int(req.max_tokens)),
                                token_events=token_events,
                                latency_ms=round(latency_ms, 2),
                            )
                        )
                        await db_session.commit()
                except Exception:
                    logger.exception("Failed to persist request log")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/info")
async def info():
    payload = dict(_info_payload)
    payload["database"] = await _database_metrics()

    if _redis_client is None:
        payload["redis"] = {"connected": False}
        return payload

    try:
        queue_length = await _redis_client.llen(_request_queue_key)
        worker_stats_raw = await _redis_client.hgetall(WORKER_STATS_KEY)
        payload["redis"] = {
            "connected": True,
            "url": _redis_url,
            "request_queue_key": _request_queue_key,
            "pending_requests": queue_length,
            "worker": _coerce_worker_stats(worker_stats_raw),
        }
    except Exception as exc:
        payload["redis"] = {"connected": False, "error": str(exc)}
    return payload


def main():
    global _checkpoint_path, _info_payload, _redis_url, _request_queue_key
    global _stream_idle_timeout_sec

    parser = argparse.ArgumentParser(description="TinyGPT FastAPI API gateway")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--redis_url",
        type=str,
        default=os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
        help="Redis URL, e.g. redis://127.0.0.1:6379/0",
    )
    parser.add_argument(
        "--request_queue_key",
        type=str,
        default=REQUEST_QUEUE_KEY,
        help="Redis list key used as the shared inference queue",
    )
    parser.add_argument(
        "--stream_idle_timeout_sec",
        type=float,
        default=60.0,
        help="Stop SSE stream if worker is idle for this many seconds",
    )
    args = parser.parse_args()

    _checkpoint_path = _resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    _info_payload = _load_checkpoint_info(_checkpoint_path)
    _redis_url = args.redis_url
    _request_queue_key = args.request_queue_key
    _stream_idle_timeout_sec = max(5.0, float(args.stream_idle_timeout_sec))

    print(f"API using checkpoint metadata: {_checkpoint_path}")
    print(f"Redis URL: {_redis_url}")
    print(f"Request queue key: {_request_queue_key}")
    print(f"Open http://{args.host}:{args.port} in your browser")

    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
        name="static",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
