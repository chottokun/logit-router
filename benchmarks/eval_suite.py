import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from logit_router.router import LogitRouter


def measure_latency_cuda(func, *args, **kwargs):
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        return result, latency_ms
    else:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return result, latency_ms


def main():
    parser = argparse.ArgumentParser(description="LogitRouter Evaluation Suite")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="モデルID"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/data/eval_cases.json",
        help="データセットのパス",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="benchmarks/reports/eval_report.md",
        help="出力レポートのパス",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmarks/reports/eval_results.json",
        help="出力JSONのパス",
    )
    parser.add_argument("--device", type=str, default=None, help="デバイス (cuda, cpu)")
    parser.add_argument(
        "--limit", type=int, default=None, help="テストケース件数の上限"
    )

    args = parser.parse_args()

    console = Console()
    console.print("[bold green]LogitRouter 評価ハーネスを開始します[/bold green]")
    console.print(f"Model: {args.model}")
    console.print(f"Dataset: {args.dataset}")

    # データセットの読み込み
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit is not None:
        dataset = dataset[: args.limit]

    console.print(f"Total test cases: {len(dataset)}")

    # ルーターの初期化
    console.print("[cyan]モデルを初期化中...[/cyan]")
    router = LogitRouter(model_id=args.model, device=args.device)

    results = []
    failure_cases = []
    latencies = []
    confidences = []
    entropies = []
    correct_count = 0

    domain_stats = {}

    console.print("[cyan]評価を実行中...[/cyan]")
    for i, item in enumerate(dataset):
        instruction = item.get("input_text", "")
        choices = item.get("choices", [])
        expected = item.get("expected_choice", "")
        domain = item.get("domain", "default")

        # We assume context is empty for this eval or we can use a 'context' key if it exists
        context = item.get("context", "")

        # Route execution with latency measurement
        try:
            route_result, latency_ms = measure_latency_cuda(
                router.route, context=context, instruction=instruction, choices=choices
            )
        except Exception as e:
            console.print(f"[bold red]Error in case {i}: {e}[/bold red]")
            continue

        best_choice = route_result["best_choice"]
        confidence = route_result["confidence"]
        entropy = route_result["entropy"]

        is_correct = best_choice == expected
        if is_correct:
            correct_count += 1
        else:
            failure_cases.append(
                {
                    "input": instruction,
                    "expected": expected,
                    "predicted": best_choice,
                    "confidence": confidence,
                    "entropy": entropy,
                    "domain": domain,
                }
            )

        latencies.append(latency_ms)
        confidences.append(confidence)
        entropies.append(entropy)

        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "correct": 0}
        domain_stats[domain]["total"] += 1
        if is_correct:
            domain_stats[domain]["correct"] += 1

        results.append(
            {
                "input": instruction,
                "expected": expected,
                "predicted": best_choice,
                "is_correct": is_correct,
                "latency_ms": latency_ms,
                "confidence": confidence,
                "entropy": entropy,
                "domain": domain,
            }
        )

    total_cases = len(results)
    if total_cases == 0:
        console.print("[bold red]評価ケースがありません。[/bold red]")
        sys.exit(1)

    overall_accuracy = (correct_count / total_cases) * 100
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_confidence = statistics.mean(confidences)
    avg_entropy = statistics.mean(entropies)

    # ドメイン別集計
    domain_accuracies = {}
    for d, stats in domain_stats.items():
        domain_accuracies[d] = (stats["correct"] / stats["total"]) * 100

    # コンソール出力 (Rich Tables)
    console.print("\n[bold]評価結果サマリー[/bold]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("メトリクス", style="dim", width=25)
    summary_table.add_column("値", justify="right")

    summary_table.add_row(
        "全体正解率", f"{overall_accuracy:.2f}% ({correct_count}/{total_cases})"
    )
    summary_table.add_row("平均レイテンシ", f"{avg_latency:.2f} ms")
    summary_table.add_row("最小レイテンシ", f"{min_latency:.2f} ms")
    summary_table.add_row("最大レイテンシ", f"{max_latency:.2f} ms")
    summary_table.add_row("平均確信度", f"{avg_confidence:.4f}")
    summary_table.add_row("平均エントロピー", f"{avg_entropy:.4f}")

    console.print(summary_table)

    if domain_accuracies:
        console.print("\n[bold]ドメイン別正解率[/bold]")
        domain_table = Table(show_header=True, header_style="bold blue")
        domain_table.add_column("ドメイン")
        domain_table.add_column("正解率", justify="right")
        domain_table.add_column("件数", justify="right")

        for d, acc in domain_accuracies.items():
            total_d = domain_stats[d]["total"]
            correct_d = domain_stats[d]["correct"]
            domain_table.add_row(d, f"{acc:.2f}%", f"{correct_d}/{total_d}")

        console.print(domain_table)

    # JSON出力
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "overall_accuracy": overall_accuracy,
                    "avg_latency_ms": avg_latency,
                    "min_latency_ms": min_latency,
                    "max_latency_ms": max_latency,
                    "avg_confidence": avg_confidence,
                    "avg_entropy": avg_entropy,
                },
                "domain_accuracies": domain_accuracies,
                "results": results,
                "failures": failure_cases,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Markdownレポート出力
    now_str = datetime.now().isoformat()
    md_content = f"""---
type: metrics
title: Model Evaluation Report
description: LogitRouter モデルの包括的評価レポート
status: completed
generated:
  by: eval_suite.py
  at: {now_str}
tags: [evaluation, logit-router, metrics]
sources: []
---

# モデル評価レポート

## 評価設定

- **Model**: `{args.model}`
- **Dataset**: `{args.dataset}`
- **Device**: `{args.device if args.device else "Auto"}`
- **Total Cases**: `{total_cases}`

## 総合サマリー

| メトリクス | 値 |
| :--- | :--- |
| 全体正解率 (Overall Accuracy) | **{overall_accuracy:.2f}%** ({correct_count}/{total_cases}) |
| 平均レイテンシ (Avg Latency) | {avg_latency:.2f} ms |
| 最小レイテンシ (Min Latency) | {min_latency:.2f} ms |
| 最大レイテンシ (Max Latency) | {max_latency:.2f} ms |
| 平均確信度 (Avg Confidence) | {avg_confidence:.4f} |
| 平均エントロピー (Avg Entropy) | {avg_entropy:.4f} |

## ドメイン別サマリー

| ドメイン | 正解率 | 正解数 / 総数 |
| :--- | :--- | :--- |
"""
    for d, acc in domain_accuracies.items():
        total_d = domain_stats[d]["total"]
        correct_d = domain_stats[d]["correct"]
        md_content += f"| {d} | {acc:.2f}% | {correct_d} / {total_d} |\n"

    md_content += """
## 誤分類分析 (Failure Cases)
"""

    if not failure_cases:
        md_content += "\n誤分類されたケースはありません。素晴らしい結果です！\n"
    else:
        md_content += "\n| 入力 (Input) | 期待値 (Expected) | 予測値 (Predicted) | 確信度 | エントロピー | ドメイン |\n"
        md_content += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for fc in failure_cases:
            # Markdown table escaping for input
            safe_input = fc["input"].replace("|", "\\|").replace("\n", " ")
            md_content += f"| {safe_input} | `{fc['expected']}` | `{fc['predicted']}` | {fc['confidence']:.4f} | {fc['entropy']:.4f} | {fc['domain']} |\n"

    out_md_path = Path(args.output_report)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    console.print("\n[green]レポートを出力しました:[/green]")
    console.print(f"- Markdown: {args.output_report}")
    console.print(f"- JSON: {args.output_json}")


if __name__ == "__main__":
    main()
