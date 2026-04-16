"""
Shared Redis keys and message helpers for API <-> worker communication.
"""

from __future__ import annotations

import json
from typing import Any

REQUEST_QUEUE_KEY = "tinygpt:requests"
WORKER_STATS_KEY = "tinygpt:worker:stats"
STREAM_CHANNEL_PREFIX = "tinygpt:stream:"

EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def stream_channel(request_id: str) -> str:
    return f"{STREAM_CHANNEL_PREFIX}{request_id}"


def encode_request(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def decode_request(raw: str) -> dict[str, Any]:
    return json.loads(raw)


def encode_event(event_type: str, **payload: Any) -> str:
    body = {"type": event_type}
    body.update(payload)
    return json.dumps(body, separators=(",", ":"))


def decode_event(raw: str) -> dict[str, Any]:
    return json.loads(raw)
