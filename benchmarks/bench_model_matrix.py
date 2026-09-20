import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from logit_router.router import LogitRouter


def calculate_percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Benchmark Comparison Matrix"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct",
        help="Comma-separated list of model IDs to benchmark",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/data/deep_eval_cases.json",
        help="Path to the evaluation dataset JSON",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="benchmarks/reports/model_comparison_matrix.md",
        help="Path to save the markdown report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the models on (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of test cases to run per model",
    )

    args = parser.parse_args()

    models_to_test = [m.strip() for m in args.models.split(",") if m.strip()]

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        fallback_path = Path("benchmarks/data/eval_cases.json")
        if fallback_path.exists():
            print(f"Dataset {dataset_path} not found. Using fallback {fallback_path}.")
            dataset_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Neither {dataset_path} nor fallback {fallback_path} found."
            )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[: args.limit]

    results = {}
    domains = set(item.get("domain", "unknown") for item in dataset)
    domains = sorted(list(domains))

    console = Console()
    console.print(
        f"[bold green]Starting benchmark on {len(models_to_test)} models with {len(dataset)} cases.[/bold green]"
    )

    for model_id in models_to_test:
        console.print(f"\n[bold blue]Evaluating Model: {model_id}[/bold blue]")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        router = LogitRouter(model_id=model_id, device=args.device)

        latencies = []
        correct_count = 0
        domain_correct = defaultdict(int)
        domain_total = defaultdict(int)
        confidences = []
        entropies = []

        for item in dataset:
            domain = item.get("domain", "unknown")
            domain_total[domain] += 1

            start_time = time.perf_counter()
            result = router.route(
                context=item["context"],
                instruction=item["instruction"],
                choices=item["choices"],
            )
            latency = (time.perf_counter() - start_time) * 1000  # in ms

            latencies.append(latency)
            confidences.append(result["confidence"])
            entropies.append(result["entropy"])

            if result["best_choice"] == item["expected_choice"]:
                correct_count += 1
                domain_correct[domain] += 1

        overall_accuracy = (correct_count / len(dataset)) * 100 if dataset else 0.0
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        min_latency = min(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        p50_latency = calculate_percentile(latencies, 50)
        p95_latency = calculate_percentile(latencies, 95)
        p99_latency = calculate_percentile(latencies, 99)

        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        avg_entropy = statistics.mean(entropies) if entropies else 0.0

        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        acc_per_ms = overall_accuracy / avg_latency if avg_latency > 0 else 0.0

        domain_acc = {}
        for d in domains:
            d_total = domain_total[d]
            if d_total > 0:
                domain_acc[d] = (domain_correct[d] / d_total) * 100
            else:
                domain_acc[d] = 0.0

        results[model_id] = {
            "overall_accuracy": overall_accuracy,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "avg_confidence": avg_confidence,
            "avg_entropy": avg_entropy,
            "peak_vram_gb": peak_vram_gb,
            "acc_per_ms": acc_per_ms,
            "domain_acc": domain_acc,
        }

        del router
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create Console Tables
    summary_table = Table(title="Multi-Model Benchmark Comparison Matrix")
    summary_table.add_column("Model ID", style="cyan")
    summary_table.add_column("Accuracy (%)", justify="right")
    summary_table.add_column("Avg Latency (ms)", justify="right")
    summary_table.add_column("p95 Latency (ms)", justify="right")
    summary_table.add_column("Peak VRAM (GB)", justify="right")
    summary_table.add_column("Acc / ms", justify="right")
    summary_table.add_column("Avg Conf", justify="right")
    summary_table.add_column("Avg Entropy", justify="right")

    for model_id, res in results.items():
        summary_table.add_row(
            model_id,
            f"{res['overall_accuracy']:.1f}",
            f"{res['avg_latency']:.1f}",
            f"{res['p95_latency']:.1f}",
            f"{res['peak_vram_gb']:.2f}",
            f"{res['acc_per_ms']:.4f}",
            f"{res['avg_confidence']:.3f}",
            f"{res['avg_entropy']:.3f}",
        )

    console.print("\n")
    console.print(summary_table)

    domain_table = Table(title="Domain Accuracy Comparison (%)")
    domain_table.add_column("Model ID", style="cyan")
    for d in domains:
        domain_table.add_column(d, justify="right")

    for model_id, res in results.items():
        row = [model_id]
        for d in domains:
            row.append(f"{res['domain_acc'][d]:.1f}")
        domain_table.add_row(*row)

    console.print("\n")
    console.print(domain_table)

    # Generate Markdown Report
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write("type: metrics\n")
        f.write("title: Multi-Model Benchmark Comparison Matrix\n")
        f.write(
            "description: 複数モデルを横断比較し、回答速度と正解率のトレードオフを深層分析したベンチマーク結果\n"
        )
        f.write("status: active\n")
        f.write("generated: true\n")
        f.write("tags: [benchmark, evaluation, performance]\n")
        f.write("sources:\n")
        f.write("  - benchmarks/bench_model_matrix.py\n")
        f.write("---\n\n")

        f.write("# Multi-Model Benchmark Comparison Matrix\n\n")

        f.write("## Overall Comparison\n\n")
        f.write(
            "| Model ID | Accuracy (%) | Avg Latency (ms) | Min (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Peak VRAM (GB) | Acc / ms | Avg Confidence | Avg Entropy |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for model_id, res in results.items():
            f.write(
                f"| {model_id} | {res['overall_accuracy']:.1f} | {res['avg_latency']:.1f} | {res['min_latency']:.1f} | {res['p50_latency']:.1f} | {res['p95_latency']:.1f} | {res['p99_latency']:.1f} | {res['max_latency']:.1f} | {res['peak_vram_gb']:.2f} | {res['acc_per_ms']:.4f} | {res['avg_confidence']:.3f} | {res['avg_entropy']:.3f} |\n"
            )

        f.write("\n## Domain Accuracy Comparison (%)\n\n")
        header = "| Model ID | " + " | ".join(domains) + " |\n"
        f.write(header)
        separator = "|---| " + " | ".join(["---:"] * len(domains)) + " |\n"
        f.write(separator)

        for model_id, res in results.items():
            row = f"| {model_id} | "
            row += " | ".join([f"{res['domain_acc'][d]:.1f}" for d in domains]) + " |\n"
            f.write(row)

    console.print(f"\n[bold green]Report saved to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
