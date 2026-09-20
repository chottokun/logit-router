"""
Cross-Model Robustness & Position Bias Evaluation Matrix.
Tests position bias (permutation consistency) and OOD entropy separation across multiple models.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from logit_router.router import LogitRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_robustness_for_model(model_id: str) -> dict[str, Any]:
    logger.info(f"Loading {model_id} for robustness testing...")
    torch.cuda.empty_cache()
    router = LogitRouter(model_id=model_id, device="cuda")

    # 1. Position Bias: Customer Support Permutations (6 perms)
    context1 = "I was charged twice for my subscription this month. Please help."
    instruction1 = "Classify the user's request."
    choices1 = ["Billing & Payments", "Technical Support", "Sales & Upgrades"]
    target1 = "Billing & Payments"
    perms1 = list(itertools.permutations(choices1))

    c1_correct = 0
    c1_entropies = []
    for perm in perms1:
        res = router.route(context1, instruction1, list(perm))
        if res["best_choice"] == target1:
            c1_correct += 1
        c1_entropies.append(res["entropy"])

    # 2. Position Bias: Tool Routing Permutations (24 perms)
    context2 = "Calculate the square root of 144."
    instruction2 = "Select the best tool."
    choices2 = ["Calculator", "Web Search", "Database Query", "Weather API"]
    target2 = "Calculator"
    perms2 = list(itertools.permutations(choices2))

    c2_correct = 0
    c2_entropies = []
    for perm in perms2:
        res = router.route(context2, instruction2, list(perm))
        if res["best_choice"] == target2:
            c2_correct += 1
        c2_entropies.append(res["entropy"])

    # 3. OOD (Out-of-Distribution) Rejection Test
    ood_queries = [
        "What is the airspeed velocity of an unladen swallow?",
        "Write a haiku about autumn leaves falling in Kyoto.",
        "42 is the answer to life, the universe, and everything.",
        "SELECT * FROM users WHERE active = true;",
        "Can you generate an image of a cybernetic cat in Tokyo neon rain?",
    ]
    standard_instruction = "Select the most appropriate support channel."
    standard_choices = ["HR Benefits", "Payroll Inquiry", "IT Helpdesk"]

    ood_entropies = []
    ood_confidences = []
    for q in ood_queries:
        res = router.route(q, standard_instruction, standard_choices)
        ood_entropies.append(res["entropy"])
        ood_confidences.append(res["confidence"])

    del router
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "support_permutation_consistency": round((c1_correct / len(perms1)) * 100, 2),
        "support_entropy_range": round(max(c1_entropies) - min(c1_entropies), 4),
        "tool_permutation_consistency": round((c2_correct / len(perms2)) * 100, 2),
        "tool_entropy_range": round(max(c2_entropies) - min(c2_entropies), 4),
        "ood_mean_entropy": round(float(np.mean(ood_entropies)), 4),
        "ood_mean_confidence": round(float(np.mean(ood_confidences)), 4),
    }


def main():
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]

    results = []
    for m in models:
        r = evaluate_robustness_for_model(m)
        results.append(r)

    # Generate Markdown Report
    output_path = Path("benchmarks/reports/robustness_matrix_report.md")
    content = [
        "---",
        "type: metrics",
        "title: Cross-Model Robustness & Position Bias Matrix Report / モデル別頑健性・位置バイアス実機評価レポート",
        "description: Measured permutation consistency and OOD entropy response across 0.5B, 1.5B, and 3B models / 3モデルにおける順列一貫性およびOODエントロピー実測結果",
        "status: completed",
        "generated:",
        "  by: benchmarks/bench_robustness_matrix.py",
        f'  at: "{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}"',
        "tags: [robustness, position-bias, ood, multi-model, on-device]",
        "sources:",
        "  - benchmarks/bench_robustness_matrix.py",
        "---",
        "",
        "# Cross-Model Robustness & Position Bias Report / モデル別頑健性・位置バイアス実機評価レポート",
        "",
        "**[English]**",
        "Empirical evaluation of candidate position bias (exhaustive permutations) and Out-of-Distribution (OOD) rejection behavior on an NVIDIA GeForce RTX 3060 (12GB VRAM).",
        "",
        "**[Japanese]**",
        "NVIDIA GeForce RTX 3060 (12GB VRAM) 実機環境における、候補選択肢の提示順序バイアス（全順列検証）および分布外（OOD）入力に対するエントロピー応答の実測評価結果です。",
        "",
        "## Robustness Metrics Summary / 頑健性評価サマリー",
        "",
        "| Model / モデル | Support Perm Consistency (3 choices, 6 perms) | Support Entropy Delta | Tool Perm Consistency (4 choices, 24 perms) | Tool Entropy Delta | OOD Mean Entropy (Separation Signal) | OOD Mean Confidence |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        content.append(
            f"| `{r['model_id']}` | **{r['support_permutation_consistency']}%** | {r['support_entropy_range']} | "
            f"**{r['tool_permutation_consistency']}%** | {r['tool_entropy_range']} | "
            f"**{r['ood_mean_entropy']}** | {r['ood_mean_confidence']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(content), encoding="utf-8")
    Path("docs/references/robustness_matrix_report.md").write_text("\n".join(content), encoding="utf-8")

    json_path = Path("benchmarks/reports/robustness_matrix_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Robustness matrix benchmark finished successfully!")


if __name__ == "__main__":
    main()
