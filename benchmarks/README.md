# Benchmark And Regression Gate

This folder defines performance gates for TinyGPT load tests.

## Files

- `ci_thresholds.json`: absolute minimum/maximum thresholds used by CI.

## Local usage

Run a load test and write a report:

```bash
python scripts/load_test.py \
  --base-url http://127.0.0.1:8000 \
  --users 10 \
  --requests-per-user 10 \
  --json-out artifacts/load_test_report.json
```

Evaluate that report against thresholds:

```bash
python scripts/check_regression.py \
  --report artifacts/load_test_report.json \
  --thresholds benchmarks/ci_thresholds.json
```

Optional relative regression check versus a baseline report:

```bash
python scripts/check_regression.py \
  --report artifacts/load_test_report.json \
  --thresholds benchmarks/ci_thresholds.json \
  --baseline-report benchmarks/baseline_report.json \
  --max-throughput-drop-pct 20 \
  --max-batch-size-drop-pct 20 \
  --max-latency-p95-increase-pct 30
```
