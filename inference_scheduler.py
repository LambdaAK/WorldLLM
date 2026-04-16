"""
In-process dynamic batching scheduler for TinyGPT inference.

This keeps the current deployment simple (single backend process) while
implementing the serving behavior we need for production-style evolution:
queueing, micro-batching, and per-request streaming.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch

from vocabulary import CLIENT_ID, EOS_ID, PAD_ID


@dataclass
class GenerationRequest:
    request_id: int
    token_ids: list[int]
    temperature: float
    top_k: int | None
    max_tokens: int
    queue: asyncio.Queue[int | None]
    submitted_at: float = field(default_factory=time.perf_counter)
    cancelled: bool = False
    finished: bool = False
    _finish_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_finished(self) -> bool:
        with self._finish_lock:
            if self.finished:
                return False
            self.finished = True
            return True


class BatchScheduler:
    """Queue requests, form dynamic batches, and run batched autoregressive decode."""

    def __init__(
        self,
        model,
        config,
        device: torch.device,
        *,
        batch_timeout_ms: int = 25,
        max_batch_size: int = 8,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.batch_timeout_ms = max(1, int(batch_timeout_ms))
        self.max_batch_size = max(1, int(max_batch_size))

        self._pending: Deque[GenerationRequest] = deque()
        self._pending_lock: asyncio.Lock | None = None
        self._pending_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task | None = None
        self._shutdown = False

        self._id_lock = threading.Lock()
        self._next_request_id = 0

        self._stats_lock = threading.Lock()
        self._total_requests = 0
        self._total_batches = 0
        self._total_tokens = 0
        self._total_queue_wait_ms = 0.0
        self._last_batch_size = 0
        self._max_queue_depth = 0

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    async def start(self):
        if self.is_running:
            return
        self._loop = asyncio.get_running_loop()
        self._pending_lock = asyncio.Lock()
        self._pending_event = asyncio.Event()
        self._shutdown = False
        self._worker_task = asyncio.create_task(self._run(), name="tinygpt-batch-scheduler")

    async def stop(self):
        self._shutdown = True
        if self._pending_event is not None:
            self._pending_event.set()
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None
        await self._drain_pending()
        self._pending_lock = None
        self._pending_event = None
        self._loop = None

    async def submit(
        self,
        token_ids: list[int],
        *,
        temperature: float,
        top_k: int | None,
        max_tokens: int,
    ) -> GenerationRequest:
        running_loop = asyncio.get_running_loop()
        if (
            not self.is_running
            or self._loop is None
            or self._pending_lock is None
            or self._pending_event is None
        ):
            raise RuntimeError("BatchScheduler is not running")
        if self._loop is not running_loop:
            raise RuntimeError("BatchScheduler is bound to a different event loop; restart the server")

        request = GenerationRequest(
            request_id=self._allocate_id(),
            token_ids=list(token_ids),
            temperature=float(temperature),
            top_k=top_k,
            max_tokens=max(1, int(max_tokens)),
            queue=asyncio.Queue(),
        )

        async with self._pending_lock:
            self._pending.append(request)
            queue_depth = len(self._pending)

        with self._stats_lock:
            self._total_requests += 1
            if queue_depth > self._max_queue_depth:
                self._max_queue_depth = queue_depth

        self._pending_event.set()
        return request

    async def snapshot(self) -> dict:
        pending_lock = self._pending_lock
        if pending_lock is None:
            pending = 0
        else:
            async with pending_lock:
                pending = len(self._pending)
        with self._stats_lock:
            avg_wait = (
                self._total_queue_wait_ms / self._total_requests
                if self._total_requests > 0
                else 0.0
            )
            return {
                "batch_timeout_ms": self.batch_timeout_ms,
                "max_batch_size": self.max_batch_size,
                "pending_requests": pending,
                "total_requests": self._total_requests,
                "total_batches": self._total_batches,
                "last_batch_size": self._last_batch_size,
                "total_streamed_tokens": self._total_tokens,
                "avg_queue_wait_ms": round(avg_wait, 2),
                "max_queue_depth": self._max_queue_depth,
            }

    def _allocate_id(self) -> int:
        with self._id_lock:
            self._next_request_id += 1
            return self._next_request_id

    async def _run(self):
        while not self._shutdown:
            await self._wait_for_pending()
            if self._shutdown:
                break

            await asyncio.sleep(self.batch_timeout_ms / 1000.0)
            batch = await self._pop_batch()
            if not batch:
                continue

            with self._stats_lock:
                self._total_batches += 1
                self._last_batch_size = len(batch)
                now = time.perf_counter()
                for req in batch:
                    self._total_queue_wait_ms += (now - req.submitted_at) * 1000.0

            try:
                await asyncio.to_thread(self._process_batch, batch)
            except Exception:
                for req in batch:
                    self._finish_request(req)

    async def _wait_for_pending(self):
        pending_lock = self._pending_lock
        pending_event = self._pending_event
        if pending_lock is None or pending_event is None:
            return
        while True:
            async with pending_lock:
                if self._pending or self._shutdown:
                    return
                pending_event.clear()
            await pending_event.wait()

    async def _pop_batch(self) -> list[GenerationRequest]:
        batch: list[GenerationRequest] = []
        pending_lock = self._pending_lock
        pending_event = self._pending_event
        if pending_lock is None:
            return batch
        async with pending_lock:
            while self._pending and len(batch) < self.max_batch_size:
                req = self._pending.popleft()
                if req.cancelled:
                    self._finish_request(req)
                    continue
                batch.append(req)
            if not self._pending and pending_event is not None:
                pending_event.clear()
        return batch

    async def _drain_pending(self):
        pending_lock = self._pending_lock
        pending_event = self._pending_event
        if pending_lock is None:
            return
        async with pending_lock:
            while self._pending:
                req = self._pending.popleft()
                self._finish_request(req)
            if pending_event is not None:
                pending_event.clear()

    def _process_batch(self, batch: list[GenerationRequest]):
        # Per-request decode state.
        states = []
        for req in batch:
            if req.cancelled:
                self._finish_request(req)
                continue
            prompt = req.token_ids[-self.config.max_seq_len :]
            if not prompt:
                self._finish_request(req)
                continue
            states.append({
                "request": req,
                "tokens": prompt,
                "generated": 0,
            })

        if not states:
            return

        self.model.eval()
        with torch.no_grad():
            while states:
                active = []
                for state in states:
                    req = state["request"]
                    if req.cancelled:
                        self._finish_request(req)
                        continue
                    if state["generated"] >= req.max_tokens:
                        self._finish_request(req)
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
                    device=self.device,
                )
                for row, state in enumerate(states):
                    tokens = state["tokens"]
                    batch_input[row, : len(tokens)] = torch.tensor(
                        tokens,
                        dtype=torch.long,
                        device=self.device,
                    )

                logits = self.model(batch_input)
                sampled = []
                for row, state in enumerate(states):
                    req = state["request"]
                    row_logits = logits[row, lengths[row] - 1, :].float()
                    sampled.append(self._sample_next_token(row_logits, req.temperature, req.top_k))

                next_states = []
                for state, token_id in zip(states, sampled):
                    req = state["request"]
                    if token_id in (EOS_ID, CLIENT_ID):
                        self._finish_request(req)
                        continue

                    state["tokens"].append(token_id)
                    if len(state["tokens"]) > self.config.max_seq_len:
                        state["tokens"] = state["tokens"][-self.config.max_seq_len :]
                    state["generated"] += 1

                    self._emit_token(req, token_id)
                    with self._stats_lock:
                        self._total_tokens += 1

                    if state["generated"] >= req.max_tokens:
                        self._finish_request(req)
                    else:
                        next_states.append(state)

                states = next_states

    @staticmethod
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

    def _emit_token(self, req: GenerationRequest, token_id: int):
        if req.cancelled or req.finished:
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(req.queue.put_nowait, token_id)

    def _finish_request(self, req: GenerationRequest):
        if not req.mark_finished():
            return
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(req.queue.put_nowait, None)
