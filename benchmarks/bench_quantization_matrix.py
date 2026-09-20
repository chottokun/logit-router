"""
Quantization Benchmark on 100 Cases:
Evaluates 4-bit quantized configurations (bitsandbytes 4-bit & AWQ)
against original full-precision baseline metrics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from logit_router.router import LogitRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysis:
    id: str
    domain: str
    context: str
    instruction: str
    expected: str
    predicted: str
    confidence: float
    entropy: float


@dataclass
class QuantizedModelResult:
    config_name: str
    model_id: str
    quant_method: str
    vocab_size: int
    total_cases: int
    correct_cases: int
    accuracy: float
    domain_results: dict[str, dict[str, Any]]
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    peak_vram_mb: float
    mean_confidence: float
    mean_entropy: float
    correct_mean_entropy: float
    incorrect_mean_entropy: float


def evaluate_quantized_model(
    config_name: str,
    model_id: str,
    quant_method: str,
    load_in_4bit: bool = False,
    is_awq: bool = False,
    cases_path: str = "benchmarks/data/deep_eval_cases.json",
    warmup_runs: int = 5,
) -> QuantizedModelResult:
    logger.info(f"Initializing {config_name} ({model_id}, method={quant_method})...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    router = LogitRouter(
        model_id=model_id,
        device="cuda",
        load_in_4bit=load_in_4bit,
        is_awq=is_awq,
    )
    vocab_size = router.tokenizer.vocab_size

    # Warmup
    logger.info("Executing warmup...")
    for _ in range(warmup_runs):
        router.route(
            context="System warmup request.",
            instruction="Route query.",
            choices=["Option A", "Option B", "Option C"],
        )
    torch.cuda.synchronize()

    domain_stats: dict[str, dict[str, int]] = {}
    latencies: list[float] = []
    confidences: list[float] = []
    entropies: list[float] = []
    correct_entropies: list[float] = []
    incorrect_entropies: list[float] = []
    correct_count = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    logger.info(f"Running 100 cases on {config_name}...")
    for i, case in enumerate(cases):
        domain = case.get("domain", "general")
        expected = case["expected_choice"]

        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}
        domain_stats[domain]["total"] += 1

        start_event.record()
        res = router.route(
            context=case["context"],
            instruction=case["instruction"],
            choices=case["choices"],
        )
        end_event.record()
        torch.cuda.synchronize()

        lat = start_event.elapsed_time(end_event)
        latencies.append(lat)
        confidences.append(res["confidence"])
        ent = res["entropy"]
        entropies.append(ent)

        is_correct = res["best_choice"] == expected
        if is_correct:
            correct_count += 1
            domain_stats[domain]["correct"] += 1
            correct_entropies.append(ent)
        else:
            incorrect_entropies.append(ent)

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    res_obj = QuantizedModelResult(
        config_name=config_name,
        model_id=model_id,
        quant_method=quant_method,
        vocab_size=vocab_size,
        total_cases=len(cases),
        correct_cases=correct_count,
        accuracy=round((correct_count / len(cases)) * 100, 2),
        domain_results=domain_stats,
        mean_latency_ms=round(float(np.mean(latencies)), 2),
        p50_latency_ms=round(float(np.median(latencies)), 2),
        p90_latency_ms=round(float(np.percentile(latencies, 90)), 2),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        min_latency_ms=round(float(np.min(latencies)), 2),
        max_latency_ms=round(float(np.max(latencies)), 2),
        peak_vram_mb=round(float(peak_vram), 1),
        mean_confidence=round(float(np.mean(confidences)), 4),
        mean_entropy=round(float(np.mean(entropies)), 4),
        correct_mean_entropy=round(float(np.mean(correct_entropies)), 4) if correct_entropies else 0.0,
        incorrect_mean_entropy=round(float(np.mean(incorrect_entropies)), 4) if incorrect_entropies else 0.0,
    )

    logger.info(
        f"DONE {config_name}: Accuracy={res_obj.accuracy}% ({correct_count}/{len(cases)}), "
        f"p50={res_obj.p50_latency_ms}ms, Peak VRAM={res_obj.peak_vram_mb}MB"
    )

    del router
    torch.cuda.empty_cache()
    return res_obj


def main() -> None:
    quant_configs = [
        {
            "config_name": "Gemma-4-E2B-it (4-bit bitsandbytes)",
            "model_id": "google/gemma-4-E2B-it",
            "quant_method": "bitsandbytes 4-bit (NF4)",
            "load_in_4bit": True,
            "is_awq": False,
        },
        {
            "config_name": "Qwen2.5-1.5B-Instruct-AWQ (Marlin 4-bit)",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            "quant_method": "AWQ 4-bit (Marlin)",
            "load_in_4bit": False,
            "is_awq": True,
        },
    ]

    all_results = []
    for cfg in quant_configs:
        res = evaluate_quantized_model(
            config_name=cfg["config_name"],
            model_id=cfg["model_id"],
            quant_method=cfg["quant_method"],
            load_in_4bit=cfg["load_in_4bit"],
            is_awq=cfg["is_awq"],
        )
        all_results.append(asdict(res))

    output_dir = Path("benchmarks/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "quantization_matrix_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {json_path}")

    # Generate Markdown Report
    lines = [
        "---",
        "type: metrics",
        "title: Quantization Benchmark Report (4-bit & AWQ) / 量子化実機ベンチマークレポート",
        "description: Empirical evaluation of 4-bit (bitsandbytes) and AWQ (Marlin) quantization on RTX 3060 / RTX 3060における4-bit量子化モデルの実機評価",
        "status: completed",
        "generated:",
        "  by: benchmarks/bench_quantization_matrix.py",
        f'  at: "{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}"',
        "tags: [quantization, 4bit, awq, marlin, bitsandbytes, on-device]",
        "sources:",
        "  - benchmarks/bench_quantization_matrix.py",
        "  - benchmarks/data/deep_eval_cases.json",
        "---",
        "",
        "# Quantization Benchmark Report / 量子化モデル実機ベンチマークレポート",
        "",
        "**[English]**",
        "This report evaluates the accuracy, latency, and VRAM reduction of 4-bit quantized models (bitsandbytes NF4 and AWQ Marlin) across 100 comprehensive test cases on an NVIDIA GeForce RTX 3060 (12GB VRAM).",
        "",
        "**[Japanese]**",
        "本レポートは、NVIDIA GeForce RTX 3060 (12GB VRAM) 実機環境において、4-bit 量子化モデル（bitsandbytes NF4 および AWQ Marlin）を10大ドメイン・計100問の実務データセットで評価し、FP16/BF16 ベースラインとの精度・レイテンシ・VRAM削減効果を実機実測した結果です。",
        "",
        "## 1. Quantized Models Performance Summary / 量子化モデル性能比較サマリー (100問実測)",
        "",
        "| Configuration / 設定 | 量子化手法 | Accuracy / 正解率 | p50 Latency | Mean Latency | Peak VRAM | 平均確信度 | 平均エントロピー |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in all_results:
        lines.append(
            f"| **`{r['config_name']}`** | {r['quant_method']} | **{r['accuracy']}%** ({r['correct_cases']}/100) | "
            f"**{r['p50_latency_ms']} ms** | {r['mean_latency_ms']} ms | **{r['peak_vram_mb']} MB** ({r['peak_vram_mb']/1024:.2f} GB) | "
            f"{r['mean_confidence']} | {r['mean_entropy']} |"
        )

    lines.extend([
        "",
        "## 2. Baseline Comparison (FP16/BF16 vs 4-bit) / ベースラインとの比較",
        "",
        "| モデル系列 | 設定 | 正解率 | p50 レイテンシ | ピーク VRAM | VRAM 削減量 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        "| **`google/gemma-4-E2B-it`** | BF16 (Baseline) | **95.0%** | 67.83 ms | 9765.8 MB (9.54 GB) | - |",
        f"| **`google/gemma-4-E2B-it`** | 4-bit (bitsandbytes) | **{all_results[0]['accuracy']}%** | {all_results[0]['p50_latency_ms']} ms | {all_results[0]['peak_vram_mb']} MB ({all_results[0]['peak_vram_mb']/1024:.2f} GB) | **-{9765.8 - all_results[0]['peak_vram_mb']:.1f} MB (約 {((9765.8 - all_results[0]['peak_vram_mb'])/9765.8)*100:.1f}% 削減)** |",
        "| `Qwen/Qwen2.5-1.5B-Instruct` | BF16 (Baseline) | **81.0%** | 34.84 ms | 2964.1 MB (2.89 GB) | - |",
        f"| `Qwen/Qwen2.5-1.5B-Instruct` | AWQ 4-bit (Marlin) | **{all_results[1]['accuracy']}%** | {all_results[1]['p50_latency_ms']} ms | {all_results[1]['peak_vram_mb']} MB ({all_results[1]['peak_vram_mb']/1024:.2f} GB) | **-{2964.1 - all_results[1]['peak_vram_mb']:.1f} MB (約 {((2964.1 - all_results[1]['peak_vram_mb'])/2964.1)*100:.1f}% 削減)** |",
    ])

    report_path = output_dir / "quantization_matrix_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
