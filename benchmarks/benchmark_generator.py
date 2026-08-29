"""
Benchmark script measuring synthetic episode generation throughput (episodes/sec).
Target: >= 100 episodes/sec.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.data import get_rng
from zeroshot_pfn.generator import sample_episode


def _worker_generate(n_episodes: int, seed: int) -> int:
    rng = get_rng(seed)
    config = PriorConfig()
    for _ in range(n_episodes):
        _ = sample_episode(config=config, rng=rng)
    return n_episodes


def benchmark_workers(n_workers: int, total_episodes: int = 1000) -> float:
    """Benchmark throughput with a given number of worker processes."""
    if n_workers <= 1:
        start_time = time.perf_counter()
        _worker_generate(total_episodes, seed=42)
        elapsed = time.perf_counter() - start_time
    else:
        episodes_per_worker = total_episodes // n_workers
        args = [(episodes_per_worker, 1000 + i * 37) for i in range(n_workers)]
        start_time = time.perf_counter()
        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(_worker_generate, args)
        total_episodes = sum(results)
        elapsed = time.perf_counter() - start_time

    eps_per_sec = total_episodes / elapsed
    return eps_per_sec


def main() -> None:
    print("=" * 60)
    print("Zero-Shot Tabular PFN: Synthetic Prior Generator Benchmark")
    print("=" * 60)

    cpu_count = mp.cpu_count()
    print(f"Detected CPU Cores: {cpu_count}")

    worker_counts = [w for w in [1, 2, 4, 8] if w <= cpu_count]
    if cpu_count not in worker_counts:
        worker_counts.append(cpu_count)

    results = {}
    target_throughput = 100.0

    for w in worker_counts:
        print(f"Benchmarking with {w} worker(s) across 1,000 episodes...", end="", flush=True)
        eps = benchmark_workers(w, total_episodes=1000)
        results[w] = eps
        print(f" -> {eps:.1f} episodes/sec")

    max_throughput = max(results.values())
    passed = max_throughput >= target_throughput

    # Generate benchmark markdown report
    report_lines = [
        "# Synthetic Prior Generator Benchmark Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**CPU Count:** {cpu_count}",
        f"**Target Throughput:** $\\ge {target_throughput}$ episodes/sec",
        f"**Max Observed Throughput:** **{max_throughput:.1f} episodes/sec**",
        f"**Benchmark Status:** {'PASSED (GPU Starvation Free)' if passed else 'WARNING (Below Target)'}",
        "",
        "## Worker Scaling Results",
        "",
        "| Workers | Throughput (episodes/sec) | Speedup vs 1 Worker |",
        "| :---: | :---: | :---: |",
    ]

    base_eps = results[1]
    for w, eps in results.items():
        speedup = eps / base_eps
        report_lines.append(f"| {w} | {eps:.1f} ep/s | {speedup:.2f}x |")

    report_lines.append("")
    report_content = "\n".join(report_lines)

    report_path = Path("reports/generator_benchmark.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 60)
    print(f"Max Throughput: {max_throughput:.1f} ep/s (Target: >= {target_throughput} ep/s)")
    print(f"Saved report to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
