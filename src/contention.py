"""Device-appropriate compute-contention generator and latency-measurement harness.

Section 0c: the contention must actually contend with inference on the device in
use. If inference is CPU-bound, induce CPU-pinned compute plus memory pressure. If
inference is GPU-bound, a co-running GPU workload (competing for SMs and memory
bandwidth) plus host-to-device transfer pressure is required; CPU stress barely
touches GPU inference. We measure END-TO-END per-step latency (preprocess +
inference + postprocess analogue), not kernel-only time.

IMPORTANT: latencies measured here are environment-specific. Locally measured
numbers are dev-only and never reported as the result; reportable numbers come
from run_on_colab.ipynb in the target GPU environment.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Contention generators
# ---------------------------------------------------------------------------
def _sync(device: str):
    """Device-appropriate synchronize: CUDA, Apple MPS, or no-op on CPU."""
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class _StopFlag:
    def __init__(self):
        self.stop = False


class CPUContention:
    """CPU-pinned compute (BLAS releases the GIL) plus memory pressure."""

    def __init__(self, cfg: dict):
        c = cfg["contention"]["cpu"]
        self.n_workers = int(c["n_stress_workers"])
        self.mem_mb = int(c["mem_pressure_mb"])
        self._threads = []
        self._flag = _StopFlag()
        self._mem = None

    def _worker(self):
        a = np.random.rand(512, 512)
        b = np.random.rand(512, 512)
        while not self._flag.stop:
            a = (a @ b) * 1.0000001
            if not np.isfinite(a).all():
                a = np.random.rand(512, 512)

    def __enter__(self):
        self._flag.stop = False
        # memory pressure: allocate and touch
        self._mem = np.ones((max(self.mem_mb, 1), 1024, 256), dtype=np.uint8)
        self._mem[::7] = 2
        for _ in range(self.n_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def __exit__(self, *exc):
        self._flag.stop = True
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()
        self._mem = None


class GPUContention:
    """Co-running GPU workload (SM + bandwidth competition) plus H2D transfer pressure."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self._threads = []
        self._flag = _StopFlag()

    def _compute_worker(self):
        import time as _t
        import torch
        g = self.cfg["contention"]["gpu"]
        dim = int(g.get("competitor_matmul_dim", 2048))
        sleep_s = float(g.get("competitor_sleep_s", 0.0005))
        dev = torch.device(self.device)
        a = torch.randn(dim, dim, device=dev)
        b = torch.randn(dim, dim, device=dev)
        while not self._flag.stop:
            a = (a @ b).relu() * 1.00001
            a = a / (a.abs().max() + 1e-3)
            # synchronize each iteration so the competitor keeps the GPU busy but
            # does NOT build an unbounded kernel backlog (which would make a
            # co-running step's synchronize() block pathologically). sleep_s tunes
            # the contention intensity so the frontier stays non-degenerate.
            _sync(self.device)
            _t.sleep(sleep_s)

    def _transfer_worker(self):
        import time as _t
        import torch
        g = self.cfg["contention"]["gpu"]
        sleep_s = float(g.get("competitor_sleep_s", 0.0005))
        dev = torch.device(self.device)
        pin = self.device == "cuda"  # pinned host memory is a CUDA feature
        host = torch.randn(32, 1024, 1024, pin_memory=pin)  # ~128 MB
        while not self._flag.stop:
            _ = host.to(dev, non_blocking=pin)
            _sync(self.device)
            _t.sleep(sleep_s)

    def __enter__(self):
        self._flag.stop = False
        t = threading.Thread(target=self._compute_worker, daemon=True)
        t.start()
        self._threads.append(t)
        if self.cfg["contention"]["gpu"].get("stress_h2d_transfer", True):
            t2 = threading.Thread(target=self._transfer_worker, daemon=True)
            t2.start()
            self._threads.append(t2)
        return self

    def __exit__(self, *exc):
        self._flag.stop = True
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()


def make_contention(cfg: dict, device: str):
    # GPU-appropriate contention for CUDA and Apple MPS; CPU stress otherwise.
    return GPUContention(cfg, device) if device in ("cuda", "mps") else CPUContention(cfg)


# ---------------------------------------------------------------------------
# Default timed workloads (a representative "perception step")
# ---------------------------------------------------------------------------
def default_step_fn(device: str) -> Callable[[], None]:
    """A representative compute step used for the Phase 1 contention demo. In
    Phase 3 the timed step is a real YOLO inference; this stand-in is only to show
    that the contention generator shifts the latency distribution."""
    import torch
    dev = torch.device(device)
    x = torch.randn(1, 64, 320, 320, device=dev)
    w = torch.randn(64, 64, 3, 3, device=dev)

    def step():
        y = x
        for _ in range(8):
            y = torch.nn.functional.conv2d(y, w, padding=1).relu()
        _sync(device)
        return float(y.mean().item())

    return step


# ---------------------------------------------------------------------------
# Latency measurement + sliding-window stats
# ---------------------------------------------------------------------------
def measure_latency(step_fn: Callable[[], object], n_steps: int, cfg: dict,
                    device: str, contended: bool, warmup: int = 10) -> np.ndarray:
    """Time n_steps of step_fn, optionally under contention. Returns latencies (s)."""
    for _ in range(warmup):
        step_fn()
    lat = np.empty(n_steps, dtype=np.float64)
    ctx = make_contention(cfg, device) if contended else None
    if ctx is not None:
        ctx.__enter__()
        for _ in range(3):  # let competitor ramp up
            step_fn()
    try:
        for i in range(n_steps):
            t0 = time.perf_counter()
            step_fn()
            lat[i] = time.perf_counter() - t0
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return lat


def sliding_window_stats(lat_s: np.ndarray, window: int, fps: float) -> pd.DataFrame:
    """Sliding p50/p95/p99 and a single-server queue-depth signal (in frames).

    Queue model: frames arrive every dt = 1/fps and are served in lat_s[i] seconds.
    backlog_seconds_i = max(0, backlog_{i-1} + lat_i - dt); queue_depth = backlog/dt.
    """
    n = len(lat_s)
    dt = 1.0 / fps
    p50 = np.full(n, np.nan)
    p95 = np.full(n, np.nan)
    p99 = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        w = lat_s[lo:i + 1]
        p50[i] = np.percentile(w, 50)
        p95[i] = np.percentile(w, 95)
        p99[i] = np.percentile(w, 99)
    backlog = np.zeros(n)
    b = 0.0
    for i in range(n):
        b = max(0.0, b + lat_s[i] - dt)
        backlog[i] = b
    return pd.DataFrame({
        "latency_s": lat_s,
        "p50_s": p50, "p95_s": p95, "p99_s": p99,
        "queue_depth": backlog / dt,
    })
