# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared per-session worker queues for PaddleOCR pipelines."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol, Sequence


class _Worker(Protocol):
    """A task adapter that owns one predictor/session for its lifetime."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the owned predictor once."""


class _WorkerPool:
    """Fan one stage's work over serial queues, one queue per session.

    The caller owns the worker adapters and therefore the predictor/session
    lifetimes. This pool owns only one single-thread executor per worker and
    shuts those executors down when leaving the context; it does not close or
    otherwise clean up predictor/session objects. A queue never runs two calls
    for its worker concurrently, and submissions select workers round-robin.
    Futures are returned in submission order when collected by the caller.
    """

    def __init__(self, workers: Sequence[_Worker]) -> None:
        self._workers = list(workers)
        self._queues: list[ThreadPoolExecutor] = []
        self._next = 0

    def __enter__(self) -> _WorkerPool:
        self._queues = [ThreadPoolExecutor(max_workers=1) for _ in self._workers]
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        for queue in self._queues:
            queue.shutdown(wait=True)
        self._queues = []
        return False

    def submit(self, *args: Any, **kwargs: Any) -> Future[Any]:
        worker_index = self._next % len(self._workers)
        self._next += 1
        return self._queues[worker_index].submit(
            self._workers[worker_index].run, *args, **kwargs,
        )
