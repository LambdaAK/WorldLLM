"""
Evaluate TinyGPT load-test output and fail if metrics regress past thresholds.

Usage:
  python scripts/check_regression.py \
    --report artifacts/load_test_report.json \
    --thresholds benchmarks/ci_thresholds.json

Optional baseline comparison:
  python scripts/check_regression.py \
    --report artifacts/load_test_report.json \
    --baseline-report benchmarks/baseline_report.json \
    --max-throughput-drop-pct 20 \
    --max-batch-size-drop-pct 20 \
    --max-latency-p95-increase-pct 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    planned = _as_float(report.get("planned_requests"), 0.0)
    success = _as_float(report.get("successful_requests"), 0.0)
    failed = _as_float(report.get("failed_requests"), 0.0)
    throughput = _as_float(report.get("throughput_rps"), 0.0)
    token_throughput = _as_float(report.get("token_throughput_client"), 0.0)

    latency = report.get("latency_ms") or {}
    if not isinstance(latency, dict):
        latency = {}

    worker = report.get("worker") or {}
    if not isinstance(worker, dict):
        worker = {}

    success_rate = (success / planned) if planned > 0 else 0.0

    return {
        "planned_requests": planned,
        "successful_requests": success,
        "failed_requests": failed,
        "success_rate": success_rate,
        "throughput_rps": throughput,
        "token_throughput_client": token_throughput,
        "latency_p95_ms": _as_float(latency.get("p95"), 0.0),
        "latency_p99_ms": _as_float(latency.get("p99"), 0.0),
        "avg_batch_size": _as_float(worker.get("avg_batch_size"), 0.0),
        "avg_queue_wait_ms": _as_float(worker.get("avg_queue_wait_ms"), 0.0),
    }


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _check_thresholds(metrics: dict[str, float], thresholds: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    max_failed_requests = _as_float(thresholds.get("max_failed_requests"), float("inf"))
    min_success_rate = _as_float(thresholds.get("min_success_rate"), 0.0)
    min_throughput_rps = _as_float(thresholds.get("min_throughput_rps"), 0.0)
    min_token_tps = _as_float(thresholds.get("min_token_throughput_client"), 0.0)
    max_latency_p95_ms = _as_float(thresholds.get("max_latency_p95_ms"), float("inf"))
    max_latency_p99_ms = _as_float(thresholds.get("max_latency_p99_ms"), float("inf"))
    min_avg_batch_size = _as_float(thresholds.get("min_avg_batch_size"), 0.0)
    max_queue_wait_ms = _as_float(thresholds.get("max_avg_queue_wait_ms"), float("inf"))

    if metrics["failed_requests"] > max_failed_requests:
        failures.append(
            f"failed_requests={metrics['failed_requests']:.0f} exceeds max_failed_requests={max_failed_requests:.0f}"
        )
    if metrics["success_rate"] < min_success_rate:
        failures.append(
            f"success_rate={_format_pct(metrics['success_rate'])} below min_success_rate={_format_pct(min_success_rate)}"
        )
    if metrics["throughput_rps"] < min_throughput_rps:
        failures.append(
            f"throughput_rps={metrics['throughput_rps']:.2f} below min_throughput_rps={min_throughput_rps:.2f}"
        )
    if metrics["token_throughput_client"] < min_token_tps:
        failures.append(
            "token_throughput_client="
            f"{metrics['token_throughput_client']:.2f} below min_token_throughput_client={min_token_tps:.2f}"
        )
    if metrics["latency_p95_ms"] > max_latency_p95_ms:
        failures.append(
            f"latency_p95_ms={metrics['latency_p95_ms']:.2f} exceeds max_latency_p95_ms={max_latency_p95_ms:.2f}"
        )
    if metrics["latency_p99_ms"] > max_latency_p99_ms:
        failures.append(
            f"latency_p99_ms={metrics['latency_p99_ms']:.2f} exceeds max_latency_p99_ms={max_latency_p99_ms:.2f}"
        )
    if metrics["avg_batch_size"] < min_avg_batch_size:
        failures.append(
            f"avg_batch_size={metrics['avg_batch_size']:.2f} below min_avg_batch_size={min_avg_batch_size:.2f}"
        )
    if metrics["avg_queue_wait_ms"] > max_queue_wait_ms:
        failures.append(
            f"avg_queue_wait_ms={metrics['avg_queue_wait_ms']:.2f} exceeds max_avg_queue_wait_ms={max_queue_wait_ms:.2f}"
        )

    return failures


def _check_baseline_regression(
    metrics: dict[str, float],
    baseline: dict[str, float],
    *,
    max_throughput_drop_pct: float,
    max_batch_drop_pct: float,
    max_latency_p95_increase_pct: float,
    max_queue_wait_increase_pct: float,
) -> list[str]:
    failures: list[str] = []

    baseline_throughput = baseline["throughput_rps"]
    if baseline_throughput > 0:
        min_allowed = baseline_throughput * (1.0 - max_throughput_drop_pct / 100.0)
        if metrics["throughput_rps"] < min_allowed:
            failures.append(
                "throughput regression: "
                f"current={metrics['throughput_rps']:.2f} req/s, "
                f"baseline={baseline_throughput:.2f} req/s, "
                f"allowed minimum={min_allowed:.2f} req/s"
            )

    baseline_batch = baseline["avg_batch_size"]
    if baseline_batch > 0:
        min_allowed = baseline_batch * (1.0 - max_batch_drop_pct / 100.0)
        if metrics["avg_batch_size"] < min_allowed:
            failures.append(
                "batch size regression: "
                f"current={metrics['avg_batch_size']:.2f}, "
                f"baseline={baseline_batch:.2f}, "
                f"allowed minimum={min_allowed:.2f}"
            )

    baseline_latency_p95 = baseline["latency_p95_ms"]
    if baseline_latency_p95 > 0:
        max_allowed = baseline_latency_p95 * (1.0 + max_latency_p95_increase_pct / 100.0)
        if metrics["latency_p95_ms"] > max_allowed:
            failures.append(
                "latency p95 regression: "
                f"current={metrics['latency_p95_ms']:.2f} ms, "
                f"baseline={baseline_latency_p95:.2f} ms, "
                f"allowed maximum={max_allowed:.2f} ms"
            )

    baseline_queue_wait = baseline["avg_queue_wait_ms"]
    if baseline_queue_wait > 0:
        max_allowed = baseline_queue_wait * (1.0 + max_queue_wait_increase_pct / 100.0)
        if metrics["avg_queue_wait_ms"] > max_allowed:
            failures.append(
                "queue wait regression: "
                f"current={metrics['avg_queue_wait_ms']:.2f} ms, "
                f"baseline={baseline_queue_wait:.2f} ms, "
                f"allowed maximum={max_allowed:.2f} ms"
            )

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail build if load-test metrics regress")
    parser.add_argument("--report", required=True, help="Path to load_test.py JSON report")
    parser.add_argument(
        "--thresholds",
        default="",
        help="Optional JSON file with absolute threshold keys",
    )
    parser.add_argument(
        "--baseline-report",
        default="",
        help="Optional prior load-test JSON report for relative regression checks",
    )
    parser.add_argument("--max-throughput-drop-pct", type=float, default=20.0)
    parser.add_argument("--max-batch-size-drop-pct", type=float, default=20.0)
    parser.add_argument("--max-latency-p95-increase-pct", type=float, default=30.0)
    parser.add_argument("--max-queue-wait-increase-pct", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Report file not found: {report_path}")

    report = _load_json(str(report_path))
    metrics = _extract_metrics(report)

    print("=== Regression Gate Input ===")
    print(f"report:                 {report_path}")
    print(f"success_rate:           {_format_pct(metrics['success_rate'])}")
    print(f"throughput_rps:         {metrics['throughput_rps']:.2f}")
    print(f"token_throughput:       {metrics['token_throughput_client']:.2f}")
    print(f"latency_p95_ms:         {metrics['latency_p95_ms']:.2f}")
    print(f"latency_p99_ms:         {metrics['latency_p99_ms']:.2f}")
    print(f"avg_batch_size:         {metrics['avg_batch_size']:.2f}")
    print(f"avg_queue_wait_ms:      {metrics['avg_queue_wait_ms']:.2f}")

    failures: list[str] = []

    if args.thresholds:
        thresholds_path = Path(args.thresholds)
        if not thresholds_path.exists():
            raise SystemExit(f"Threshold file not found: {thresholds_path}")
        thresholds = _load_json(str(thresholds_path))
        print(f"thresholds:             {thresholds_path}")
        failures.extend(_check_thresholds(metrics, thresholds))

    if args.baseline_report:
        baseline_path = Path(args.baseline_report)
        if not baseline_path.exists():
            raise SystemExit(f"Baseline report file not found: {baseline_path}")

        baseline_report = _load_json(str(baseline_path))
        baseline_metrics = _extract_metrics(baseline_report)
        print(f"baseline_report:        {baseline_path}")
        failures.extend(
            _check_baseline_regression(
                metrics,
                baseline_metrics,
                max_throughput_drop_pct=args.max_throughput_drop_pct,
                max_batch_drop_pct=args.max_batch_size_drop_pct,
                max_latency_p95_increase_pct=args.max_latency_p95_increase_pct,
                max_queue_wait_increase_pct=args.max_queue_wait_increase_pct,
            )
        )

    print()
    if failures:
        print("Regression gate: FAIL")
        for issue in failures:
            print(f"- {issue}")
        return 1

    print("Regression gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
