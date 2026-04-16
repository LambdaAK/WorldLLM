"""
Concurrent load test for TinyGPT API.

Simulates multiple users sending chat requests concurrently, then reports:
- latency (avg/p50/p95/p99)
- throughput (requests/sec, tokens/sec)
- batching efficiency (average batch size from worker stats)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


SCENARIOS = [
    [
        "Alice has the ball. Bob has the key. Charlie has the clock.",
        "Who has the ball?",
        "Bob gives the key to Alice. Who has the key?",
        "Who has what?",
    ],
    [
        "Alice has 5 apples. Bob has 3 apples.",
        "Who has the most apples?",
        "Alice gives 2 apples to Bob.",
        "How many apples does Bob have?",
    ],
    [
        "Charlie has the book. Diana has the pen.",
        "Charlie gives the book to Diana.",
        "Who has the book?",
        "What does Diana have?",
    ],
    [
        "Eve has the lamp. Frank has the ring.",
        "Frank gives the ring to Eve.",
        "Who has the ring?",
        "Who has what?",
    ],
    [
        "Grace has the cup. Henry has the coin.",
        "Henry gives the coin to Grace.",
        "What does Grace have?",
        "Who has the coin?",
    ],
]


@dataclass
class RequestResult:
    user_id: int
    turn_index: int
    ok: bool
    latency_ms: float
    token_events: int
    error: str = ""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def fetch_info(base_url: str, timeout_sec: float) -> dict[str, Any]:
    req = urllib.request.Request(f"{base_url}/info", method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def parse_worker_stats(info: dict[str, Any]) -> dict[str, float]:
    worker = (((info or {}).get("redis") or {}).get("worker") or {})

    def as_float(key: str) -> float:
        value = worker.get(key, 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return {
        "total_requests": as_float("total_requests"),
        "total_batches": as_float("total_batches"),
        "total_streamed_tokens": as_float("total_streamed_tokens"),
        "avg_queue_wait_ms": as_float("avg_queue_wait_ms"),
    }


def stream_chat_once(
    base_url: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_k: int,
    max_tokens: int,
    timeout_sec: float,
) -> tuple[bool, float, int, str, str]:
    body = json.dumps(
        {
            "messages": messages,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    token_events = 0
    chunks: list[str] = []

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    return True, latency_ms, token_events, "".join(chunks).strip(), ""
                if payload.startswith("[ERROR]"):
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    return False, latency_ms, token_events, "".join(chunks).strip(), payload
                token_events += 1
                chunks.append(payload)

        latency_ms = (time.perf_counter() - start) * 1000.0
        return False, latency_ms, token_events, "".join(chunks).strip(), "Stream ended without [DONE]"
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        try:
            details = exc.read().decode("utf-8", errors="replace")
        except Exception:
            details = str(exc)
        return False, latency_ms, token_events, "", f"HTTP {exc.code}: {details}"
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return False, latency_ms, token_events, "", str(exc)


async def run_user(
    user_id: int,
    *,
    base_url: str,
    turns: int,
    temperature: float,
    top_k: int,
    max_tokens: int,
    timeout_sec: float,
    stagger_ms: float,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    messages: list[dict[str, str]] = []
    scenario = SCENARIOS[user_id % len(SCENARIOS)]

    if stagger_ms > 0:
        await asyncio.sleep((user_id * stagger_ms) / 1000.0)

    for turn in range(turns):
        prompt = scenario[turn % len(scenario)]
        messages.append({"role": "user", "content": prompt})

        ok, latency_ms, token_events, assistant_text, err = await asyncio.to_thread(
            stream_chat_once,
            base_url,
            messages,
            temperature,
            top_k,
            max_tokens,
            timeout_sec,
        )

        results.append(
            RequestResult(
                user_id=user_id,
                turn_index=turn,
                ok=ok,
                latency_ms=latency_ms,
                token_events=token_events,
                error=err,
            )
        )

        if ok:
            if not assistant_text:
                assistant_text = "..."
            messages.append({"role": "assistant", "content": assistant_text})
        else:
            # Keep message flow valid for subsequent turns.
            messages.append({"role": "assistant", "content": "[error]"})

        # Tiny jitter to avoid lockstep behavior.
        await asyncio.sleep(random.uniform(0.005, 0.03))

    return results


async def run_load_test(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    total_planned = args.users * args.requests_per_user

    try:
        info_before = await asyncio.to_thread(fetch_info, base_url, args.info_timeout_sec)
    except Exception as exc:
        print(f"Failed to fetch {base_url}/info before test: {exc}")
        return 2

    worker_before = parse_worker_stats(info_before)

    started = time.perf_counter()
    tasks = [
        run_user(
            user_id=u,
            base_url=base_url,
            turns=args.requests_per_user,
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            timeout_sec=args.request_timeout_sec,
            stagger_ms=args.stagger_ms,
        )
        for u in range(args.users)
    ]
    grouped = await asyncio.gather(*tasks)
    duration_sec = time.perf_counter() - started

    try:
        info_after = await asyncio.to_thread(fetch_info, base_url, args.info_timeout_sec)
        worker_after = parse_worker_stats(info_after)
    except Exception:
        info_after = {}
        worker_after = {"total_requests": 0.0, "total_batches": 0.0, "total_streamed_tokens": 0.0, "avg_queue_wait_ms": 0.0}

    results = [r for group in grouped for r in group]
    successes = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]

    latencies = [r.latency_ms for r in successes]
    total_token_events = sum(r.token_events for r in successes)

    throughput_rps = (len(successes) / duration_sec) if duration_sec > 0 else 0.0
    throughput_tokens = (total_token_events / duration_sec) if duration_sec > 0 else 0.0

    delta_requests = max(0.0, worker_after["total_requests"] - worker_before["total_requests"])
    delta_batches = max(0.0, worker_after["total_batches"] - worker_before["total_batches"])
    delta_worker_tokens = max(0.0, worker_after["total_streamed_tokens"] - worker_before["total_streamed_tokens"])
    avg_batch_size = (delta_requests / delta_batches) if delta_batches > 0 else 0.0

    print("=== TinyGPT Load Test ===")
    print(f"Base URL:               {base_url}")
    print(f"Simulated users:        {args.users}")
    print(f"Requests per user:      {args.requests_per_user}")
    print(f"Total planned requests: {total_planned}")
    print(f"Total wall time:        {duration_sec:.2f}s")
    print()

    print("=== Request Results ===")
    print(f"Successful requests:    {len(successes)}")
    print(f"Failed requests:        {len(failures)}")
    print(f"Throughput:             {throughput_rps:.2f} req/s")
    print(f"Token throughput:       {throughput_tokens:.2f} token-events/s (client observed)")
    if latencies:
        print(f"Latency avg:            {statistics.mean(latencies):.2f} ms")
        print(f"Latency p50:            {percentile(latencies, 50):.2f} ms")
        print(f"Latency p95:            {percentile(latencies, 95):.2f} ms")
        print(f"Latency p99:            {percentile(latencies, 99):.2f} ms")
        print(f"Latency min/max:        {min(latencies):.2f} / {max(latencies):.2f} ms")
    else:
        print("Latency:                no successful requests")
    print()

    print("=== Worker / Batching ===")
    print(f"Worker requests delta:  {delta_requests:.0f}")
    print(f"Worker batches delta:   {delta_batches:.0f}")
    print(f"Average batch size:     {avg_batch_size:.2f}")
    print(f"Worker tokens delta:    {delta_worker_tokens:.0f}")
    print(f"Worker avg queue wait:  {worker_after['avg_queue_wait_ms']:.2f} ms")
    print()

    if failures:
        print("=== Sample Errors ===")
        shown = 0
        for entry in failures:
            if shown >= 5:
                break
            print(f"user={entry.user_id} turn={entry.turn_index} -> {entry.error}")
            shown += 1
        print()

    if args.json_out:
        report = {
            "base_url": base_url,
            "users": args.users,
            "requests_per_user": args.requests_per_user,
            "duration_sec": duration_sec,
            "planned_requests": total_planned,
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "throughput_rps": throughput_rps,
            "token_throughput_client": throughput_tokens,
            "latency_ms": {
                "avg": statistics.mean(latencies) if latencies else None,
                "p50": percentile(latencies, 50) if latencies else None,
                "p95": percentile(latencies, 95) if latencies else None,
                "p99": percentile(latencies, 99) if latencies else None,
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "worker": {
                "requests_delta": delta_requests,
                "batches_delta": delta_batches,
                "avg_batch_size": avg_batch_size,
                "tokens_delta": delta_worker_tokens,
                "avg_queue_wait_ms": worker_after["avg_queue_wait_ms"],
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report: {args.json_out}")

    return 0 if not failures else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test TinyGPT API with concurrent users.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=100, help="Concurrent simulated users")
    parser.add_argument("--requests-per-user", type=int, default=100, help="Sequential requests per user")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--request-timeout-sec", type=float, default=90.0)
    parser.add_argument("--info-timeout-sec", type=float, default=5.0)
    parser.add_argument("--stagger-ms", type=float, default=8.0, help="Per-user start stagger")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write JSON metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.users < 1 or args.requests_per_user < 1:
        raise SystemExit("users and requests-per-user must be >= 1")
    code = asyncio.run(run_load_test(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
