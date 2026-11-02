"""Warmup, CUDA-event timing, bandwidth / TFLOPS, and CSV output for kernel benches."""

import csv
from pathlib import Path

import torch


def bench(fn, warmup=20, iters=100):
    """Return the mean runtime of ``fn`` in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def gbps(bytes_moved, ms):
    """Effective bandwidth in GB/s. ``bytes_moved`` is the bytes touched per call."""
    if ms <= 0:
        return 0.0
    return bytes_moved / ms / 1e6


def tflops(flops, ms):
    """Throughput in TFLOP/s."""
    if ms <= 0:
        return 0.0
    return flops / ms / 1e9


def report(name, ms, bytes_moved=0, flops=0):
    """Print one result line and return its metrics."""
    row = {
        "name": name,
        "us": ms * 1e3,
        "ms": ms,
        "gbps": gbps(bytes_moved, ms) if bytes_moved else 0.0,
        "tflops": tflops(flops, ms) if flops else 0.0,
    }
    print(f"{name:<32}{row['us']:10.1f} us {row['gbps']:9.0f} GB/s {row['tflops']:8.1f} TFLOPS")
    return row


def write_csv(path, rows):
    """Append ``report`` rows to ``path``, writing the header once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "us", "ms", "gbps", "tflops"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
