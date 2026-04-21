#!/usr/bin/env python3
"""
Unified TinyGPT launcher.

Examples:
  ./tinygpt up --mode local
  ./tinygpt up --mode docker
  ./tinygpt up --mode k8s --with-observability
  ./tinygpt status --mode k8s
  ./tinygpt down --mode docker
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / ".tinygpt" / "runtime"
LOCAL_RUNTIME = RUNTIME_ROOT / "local"
K8S_RUNTIME = RUNTIME_ROOT / "k8s"

LOCAL_API_PID = LOCAL_RUNTIME / "api.pid"
LOCAL_WORKER_PID = LOCAL_RUNTIME / "worker.pid"
LOCAL_API_LOG = LOCAL_RUNTIME / "api.log"
LOCAL_WORKER_LOG = LOCAL_RUNTIME / "worker.log"

K8S_PORT_FORWARD_PID = K8S_RUNTIME / "api-port-forward.pid"
K8S_PORT_FORWARD_LOG = K8S_RUNTIME / "api-port-forward.log"

DEFAULT_DATABASE_URL = "postgresql+asyncpg://tinygpt:tinygpt@127.0.0.1:5432/tinygpt"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
DEFAULT_NAMESPACE = "tinygpt"
DEFAULT_IMAGE = "tinygpt:latest"
DEFAULT_POSTGRES_CANDIDATES = ("postgresql@17", "postgresql@16", "postgresql@15", "postgresql@14", "postgresql")


def _print_step(message: str) -> None:
    print(f"[tinygpt] {message}")


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"Required command not found in PATH: {command}")


def _run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    _print_step(f"$ {shlex.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, env=env, check=check)


def _run_shell(command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    _print_step(f"$ {command}")
    return subprocess.run(["/bin/zsh", "-lc", command], cwd=ROOT, check=check)


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid), encoding="utf-8")


def _stop_pid_file(path: Path, label: str) -> None:
    pid = _read_pid(path)
    if pid is None:
        _print_step(f"{label}: not running")
        path.unlink(missing_ok=True)
        return

    if not _is_pid_running(pid):
        _print_step(f"{label}: stale pid {pid}, cleaning up")
        path.unlink(missing_ok=True)
        return

    _print_step(f"Stopping {label} (pid {pid})")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        path.unlink(missing_ok=True)
        return

    deadline = time.time() + 8.0
    while time.time() < deadline:
        if not _is_pid_running(pid):
            path.unlink(missing_ok=True)
            _print_step(f"{label}: stopped")
            return
        time.sleep(0.2)

    _print_step(f"{label}: did not exit after SIGTERM, sending SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    path.unlink(missing_ok=True)


def _start_background_process(
    cmd: list[str],
    *,
    env: dict[str, str] | None,
    pid_file: Path,
    log_file: Path,
    label: str,
) -> int:
    existing_pid = _read_pid(pid_file)
    if existing_pid is not None and _is_pid_running(existing_pid):
        _print_step(f"{label}: already running (pid {existing_pid})")
        return existing_pid

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    _write_pid(pid_file, proc.pid)
    _print_step(f"{label}: started pid {proc.pid}, logs -> {log_file}")
    return proc.pid


def _wait_for_http(url: str, *, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 500:
                    return True
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _checkpoint_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _brew_formula_installed(formula: str) -> bool:
    result = subprocess.run(
        ["brew", "list", "--formula", formula],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _postgres_service_candidates(preferred: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in (preferred, *DEFAULT_POSTGRES_CANDIDATES):
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _select_installed_postgres_service(preferred: str) -> str | None:
    for candidate in _postgres_service_candidates(preferred):
        if _brew_formula_installed(candidate):
            return candidate
    return None


def _start_minikube_with_fallback(args: argparse.Namespace) -> None:
    candidate_memories: list[int] = []
    for memory_mb in (args.minikube_memory_mb, 6144, 4096, 3072, 2048):
        if memory_mb not in candidate_memories:
            candidate_memories.append(memory_mb)

    for memory_mb in candidate_memories:
        result = _run(
            [
                "minikube",
                "start",
                "--driver=docker",
                "--cpus",
                str(args.minikube_cpus),
                "--memory",
                str(memory_mb),
            ],
            check=False,
        )
        if result.returncode == 0:
            if memory_mb != args.minikube_memory_mb:
                _print_step(f"minikube started with fallback memory {memory_mb}MB")
            return

        _print_step(f"minikube start failed with --memory {memory_mb}. Cleaning partial state before retry.")
        _run(["minikube", "delete"], check=False)

    raise SystemExit(
        "Failed to start minikube after multiple retries. "
        "Check Docker Desktop disk/health and resource limits, then run: minikube delete"
    )


def _print_k8s_diagnostics(namespace: str) -> None:
    _print_step("Collecting kubernetes diagnostics...")
    _run(["kubectl", "get", "pods", "-n", namespace, "-o", "wide"], check=False)
    _run(["kubectl", "get", "svc", "-n", namespace], check=False)
    _run(["kubectl", "get", "hpa", "-n", namespace], check=False)
    _run(["kubectl", "describe", "deployment", "-n", namespace, "tinygpt-api"], check=False)
    _run(["kubectl", "describe", "deployment", "-n", namespace, "tinygpt-worker"], check=False)
    _run(["kubectl", "get", "events", "-n", namespace, "--sort-by=.metadata.creationTimestamp"], check=False)


def _handle_local_up(args: argparse.Namespace) -> int:
    checkpoint = _checkpoint_path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            "Train first or pass --checkpoint to a valid .pt file."
        )

    db_enabled = not args.no_db
    selected_postgres_service = args.postgres_service

    if not args.skip_brew_services:
        _require_command("brew")
        _run(["brew", "services", "start", "redis"])
        if db_enabled:
            selected_postgres_service = _select_installed_postgres_service(args.postgres_service) or ""
            if not selected_postgres_service:
                _print_step(
                    "No Homebrew PostgreSQL formula found. "
                    "Continuing without DB (same as --no-db)."
                )
                db_enabled = False
            else:
                if selected_postgres_service != args.postgres_service:
                    _print_step(
                        f"Requested {args.postgres_service} is not installed; "
                        f"using {selected_postgres_service}."
                    )
                pg_start = _run(
                    ["brew", "services", "start", selected_postgres_service],
                    check=False,
                )
                if pg_start.returncode != 0:
                    _print_step(
                        f"Failed to start {selected_postgres_service}. "
                        "Continuing without DB (same as --no-db)."
                    )
                    db_enabled = False

    worker_cmd = [
        sys.executable,
        "worker.py",
        "--checkpoint",
        str(checkpoint),
        "--redis_url",
        args.redis_url,
        "--batch_timeout_ms",
        str(args.batch_timeout_ms),
        "--max_batch_size",
        str(args.max_batch_size),
        "--device",
        args.device,
    ]
    _start_background_process(
        worker_cmd,
        env=os.environ.copy(),
        pid_file=LOCAL_WORKER_PID,
        log_file=LOCAL_WORKER_LOG,
        label="local worker",
    )

    api_env = os.environ.copy()
    if not db_enabled:
        api_env.pop("DATABASE_URL", None)
    else:
        api_env["DATABASE_URL"] = args.database_url

    api_cmd = [
        sys.executable,
        "app.py",
        "--checkpoint",
        str(checkpoint),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--redis_url",
        args.redis_url,
        "--stream_idle_timeout_sec",
        str(args.stream_idle_timeout_sec),
    ]
    _start_background_process(
        api_cmd,
        env=api_env,
        pid_file=LOCAL_API_PID,
        log_file=LOCAL_API_LOG,
        label="local api",
    )

    health_url = f"http://127.0.0.1:{args.port}/info"
    if _wait_for_http(health_url, timeout_sec=args.wait_timeout_sec):
        _print_step(f"Local stack is ready: http://127.0.0.1:{args.port}")
    else:
        _print_step(
            "Local stack started, but readiness check timed out. "
            f"Check logs: {LOCAL_API_LOG} and {LOCAL_WORKER_LOG}"
        )

    return 0


def _handle_local_down(args: argparse.Namespace) -> int:
    _stop_pid_file(LOCAL_API_PID, "local api")
    _stop_pid_file(LOCAL_WORKER_PID, "local worker")

    if args.stop_brew_services:
        _require_command("brew")
        _run(["brew", "services", "stop", "redis"], check=False)
        if not args.no_db:
            for service in _postgres_service_candidates(args.postgres_service):
                _run(["brew", "services", "stop", service], check=False)

    return 0


def _handle_local_status(_: argparse.Namespace) -> int:
    for label, path in (
        ("local api", LOCAL_API_PID),
        ("local worker", LOCAL_WORKER_PID),
    ):
        pid = _read_pid(path)
        if pid is not None and _is_pid_running(pid):
            _print_step(f"{label}: running (pid {pid})")
        elif pid is not None:
            _print_step(f"{label}: stale pid file ({pid})")
        else:
            _print_step(f"{label}: not running")

    _print_step(f"api log: {LOCAL_API_LOG}")
    _print_step(f"worker log: {LOCAL_WORKER_LOG}")
    return 0


def _handle_docker_up(args: argparse.Namespace) -> int:
    _require_command("docker")

    cmd = ["docker", "compose", "-f", args.compose_file, "up"]
    if args.build:
        cmd.append("--build")
    if args.detach:
        cmd.append("-d")
    _run(cmd)

    if args.detach:
        health_url = f"http://127.0.0.1:{args.port}/info"
        if _wait_for_http(health_url, timeout_sec=args.wait_timeout_sec):
            _print_step(f"Docker stack is ready: http://127.0.0.1:{args.port}")
        else:
            _print_step("Docker stack started, but readiness check timed out.")

    return 0


def _handle_docker_down(args: argparse.Namespace) -> int:
    _require_command("docker")
    cmd = ["docker", "compose", "-f", args.compose_file, "down"]
    if args.down_remove_volumes:
        cmd.append("-v")
    _run(cmd)
    return 0


def _handle_docker_status(args: argparse.Namespace) -> int:
    _require_command("docker")
    _run(["docker", "compose", "-f", args.compose_file, "ps"], check=False)
    return 0


def _minikube_running() -> bool:
    result = subprocess.run(
        ["minikube", "status"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _build_image_for_minikube(image: str) -> None:
    build_cmd = ["minikube", "image", "build", "-t", image, "."]
    result = _run(build_cmd, check=False)
    if result.returncode == 0:
        return

    _print_step("minikube image build failed, falling back to docker-env build path")
    shell_image = shlex.quote(image)
    _run_shell(
        f"eval $(minikube docker-env) && DOCKER_BUILDKIT=0 docker build -t {shell_image} . && eval $(minikube docker-env -u)"
    )


def _handle_k8s_up(args: argparse.Namespace) -> int:
    _require_command("minikube")
    _require_command("kubectl")

    if not _minikube_running():
        _start_minikube_with_fallback(args)
    else:
        _print_step("minikube already running")

    _run(["kubectl", "config", "use-context", "minikube"])

    if not args.skip_metrics_server:
        _run(["minikube", "addons", "enable", "metrics-server"], check=False)

    if not args.skip_image_build:
        _build_image_for_minikube(args.image)

    try:
        _run(["kubectl", "apply", "-k", args.k8s_path])

        if args.with_observability:
            _run(["kubectl", "apply", "-k", args.observability_path])

        _run(
            [
                "kubectl",
                "rollout",
                "status",
                "-n",
                args.namespace,
                "deployment/tinygpt-api",
                f"--timeout={args.rollout_timeout_sec}s",
            ]
        )
        _run(
            [
                "kubectl",
                "rollout",
                "status",
                "-n",
                args.namespace,
                "deployment/tinygpt-worker",
                f"--timeout={args.rollout_timeout_sec}s",
            ]
        )
    except subprocess.CalledProcessError:
        _print_step("k8s deployment failed or timed out.")
        _print_k8s_diagnostics(args.namespace)
        raise SystemExit(
            "Kubernetes startup failed. See diagnostics above. "
            "Common fix: ./tinygpt up --mode k8s --skip-image-build (if image already exists) "
            "or rerun after Docker Desktop restart."
        )

    if args.port_forward:
        pf_cmd = [
            "kubectl",
            "port-forward",
            "-n",
            args.namespace,
            "svc/tinygpt-api",
            f"{args.port}:8000",
        ]
        _start_background_process(
            pf_cmd,
            env=os.environ.copy(),
            pid_file=K8S_PORT_FORWARD_PID,
            log_file=K8S_PORT_FORWARD_LOG,
            label="k8s api port-forward",
        )

        health_url = f"http://127.0.0.1:{args.port}/info"
        if _wait_for_http(health_url, timeout_sec=args.wait_timeout_sec):
            _print_step(f"Kubernetes stack is ready via port-forward: http://127.0.0.1:{args.port}")
        else:
            _print_step(
                "Kubernetes deployed, but API readiness via port-forward timed out. "
                f"Check {K8S_PORT_FORWARD_LOG}."
            )
    else:
        _print_step(
            "Kubernetes stack deployed. Start API access manually with: "
            f"kubectl port-forward -n {args.namespace} svc/tinygpt-api {args.port}:8000"
        )

    return 0


def _handle_k8s_down(args: argparse.Namespace) -> int:
    _require_command("kubectl")

    _stop_pid_file(K8S_PORT_FORWARD_PID, "k8s api port-forward")

    if args.with_observability:
        _run(["kubectl", "delete", "-k", args.observability_path], check=False)

    _run(["kubectl", "delete", "-k", args.k8s_path], check=False)

    if args.stop_minikube:
        _require_command("minikube")
        _run(["minikube", "stop"], check=False)

    return 0


def _handle_k8s_status(args: argparse.Namespace) -> int:
    _require_command("kubectl")

    _run(["minikube", "status"], check=False)
    _run(["kubectl", "get", "pods", "-n", args.namespace], check=False)
    _run(["kubectl", "get", "svc", "-n", args.namespace], check=False)
    _run(["kubectl", "get", "hpa", "-n", args.namespace], check=False)

    pid = _read_pid(K8S_PORT_FORWARD_PID)
    if pid is not None and _is_pid_running(pid):
        _print_step(f"k8s api port-forward: running (pid {pid}) on localhost:{args.port}")
    elif pid is not None:
        _print_step(f"k8s api port-forward: stale pid file ({pid})")
    else:
        _print_step("k8s api port-forward: not running")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command TinyGPT stack launcher")
    parser.add_argument("action", choices=["up", "down", "status"], help="Action to run")
    parser.add_argument(
        "--mode",
        choices=["local", "docker", "k8s"],
        required=True,
        help="Runtime mode to manage",
    )

    parser.add_argument("--host", default="0.0.0.0", help="Host for local API")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--redis-url", default=DEFAULT_REDIS_URL, help="Redis URL (local mode)")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL, help="DATABASE_URL for local mode")
    parser.add_argument("--no-db", action="store_true", help="Disable DB for local mode")
    parser.add_argument("--skip-brew-services", action="store_true", help="Do not start brew services in local mode")
    parser.add_argument("--stop-brew-services", action="store_true", help="Stop brew redis/postgres on local down")
    parser.add_argument("--postgres-service", default="postgresql@16", help="brew postgres service name")
    parser.add_argument("--batch-timeout-ms", type=int, default=25, help="Worker batch timeout for local mode")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Worker max batch size for local mode")
    parser.add_argument("--device", default="cpu", help="Worker device for local mode")
    parser.add_argument("--stream-idle-timeout-sec", type=float, default=60.0, help="API stream idle timeout")

    parser.add_argument("--compose-file", default="docker-compose.yml", help="Docker compose file path")
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build images during docker up (default: true)",
    )
    parser.add_argument(
        "--detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run docker compose in detached mode (default: true)",
    )
    parser.add_argument("--down-remove-volumes", action="store_true", help="Use docker compose down -v")

    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Image tag used for k8s deployments")
    parser.add_argument("--k8s-path", default="k8s", help="Path passed to kubectl apply -k")
    parser.add_argument(
        "--observability-path",
        default="k8s/optional/observability",
        help="Path for optional observability kustomize manifests",
    )
    parser.add_argument("--with-observability", action="store_true", help="Apply optional observability manifests")
    parser.add_argument("--skip-image-build", action="store_true", help="Skip image build before k8s apply")
    parser.add_argument("--skip-metrics-server", action="store_true", help="Skip enabling minikube metrics-server")
    parser.add_argument("--minikube-cpus", type=int, default=3, help="CPUs for minikube start")
    parser.add_argument("--minikube-memory-mb", type=int, default=6144, help="Memory for minikube start")
    parser.add_argument(
        "--port-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run API port-forward for k8s up (default: true)",
    )
    parser.add_argument("--stop-minikube", action="store_true", help="Stop minikube on k8s down")
    parser.add_argument("--rollout-timeout-sec", type=int, default=300, help="K8s rollout timeout")

    parser.add_argument("--wait-timeout-sec", type=float, default=120.0, help="Readiness wait timeout")

    args = parser.parse_args()

    if args.mode == "k8s" and args.action == "up" and not args.port_forward:
        _print_step("k8s up requested without port-forward; API won't be reachable on localhost automatically")

    return args


def main() -> int:
    args = parse_args()

    if args.mode == "local":
        if args.action == "up":
            return _handle_local_up(args)
        if args.action == "down":
            return _handle_local_down(args)
        return _handle_local_status(args)

    if args.mode == "docker":
        if args.action == "up":
            return _handle_docker_up(args)
        if args.action == "down":
            return _handle_docker_down(args)
        return _handle_docker_status(args)

    if args.mode == "k8s":
        if args.action == "up":
            return _handle_k8s_up(args)
        if args.action == "down":
            return _handle_k8s_down(args)
        return _handle_k8s_status(args)

    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
