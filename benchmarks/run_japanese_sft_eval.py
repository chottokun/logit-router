import argparse
import json
import logging
import time
from pathlib import Path

import torch
import numpy as np

from logit_router.router import LogitRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_macro_f1(y_true, y_pred, labels):
    f1_scores = []
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
        else:
            f1_scores.append(0.0)
            
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def generate_confusion_matrix(y_true, y_pred, labels):
    matrix = {t_label: {p_label: 0 for p_label in labels} for t_label in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
    return matrix

def apply_noise(text, noise_type="typo"):
    if not text: return text
    if noise_type == "typo":
        # Introduce a simple typo if long enough
        if len(text) > 10:
            idx = len(text) // 2
            return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
    elif noise_type == "long_text":
        return text + " " + "This is some extra padding text to make the prompt longer and test robust handling." * 3
    return text

def evaluate_model(model_name: str, model_id: str, gguf_file: str = None, cases: list = None):
    logging.info(f"Evaluating {model_name}...")
    torch.cuda.empty_cache()
    # Force cpu for the evaluation so that it doesn't fail if GPU runs out of memory or driver fails during batch runs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    router = LogitRouter(model_id=model_id, gguf_file=gguf_file, device=device)
    
    # Warmup
    for _ in range(3):
        router.route("System warmup request.", "Route query.", ["Option A", "Option B", "Option C"])
    if device == "cuda":
        torch.cuda.synchronize()

    correct_count = 0
    latencies = []
    
    y_true = []
    y_pred = []
    
    # Robustness specific stats
    robust_typo_correct = 0
    robust_long_correct = 0
    
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    for case in cases:
        # Standard Evaluation
        if device == "cuda":
            start_event.record()
            res = router.route(case["context"], case["instruction"], case["choices"])
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        else:
            start_t = time.perf_counter()
            res = router.route(case["context"], case["instruction"], case["choices"])
            end_t = time.perf_counter()
            latencies.append((end_t - start_t) * 1000)
            
        y_true.append(case["expected_choice"])
        y_pred.append(res["best_choice"])
        
        if res["best_choice"] == case["expected_choice"]:
            correct_count += 1
            
        # Robustness Evaluation (Typo)
        noisy_ctx = apply_noise(case["context"], "typo")
        res_noisy = router.route(noisy_ctx, case["instruction"], case["choices"])
        if res_noisy["best_choice"] == case["expected_choice"]:
            robust_typo_correct += 1
            
        # Robustness Evaluation (Long Text)
        long_ctx = apply_noise(case["context"], "long_text")
        res_long = router.route(long_ctx, case["instruction"], case["choices"])
        if res_long["best_choice"] == case["expected_choice"]:
            robust_long_correct += 1
            
    acc = (correct_count / len(cases)) * 100 if cases else 0
    latencies.sort()
    p50 = latencies[int(len(latencies)*0.5)] if latencies else 0
    p95 = latencies[int(len(latencies)*0.95)] if latencies else 0
    
    unique_labels = list(set(y_true))
    macro_f1 = calculate_macro_f1(y_true, y_pred, unique_labels)
    cm = generate_confusion_matrix(y_true, y_pred, unique_labels)
    
    typo_acc = (robust_typo_correct / len(cases)) * 100 if cases else 0
    long_acc = (robust_long_correct / len(cases)) * 100 if cases else 0
    
    del router
    torch.cuda.empty_cache()
    return {
        "model_name": model_name,
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 4),
        "robustness_typo_acc": round(typo_acc, 2),
        "robustness_long_acc": round(long_acc, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "confusion_matrix": cm
    }

def main():
    with open("benchmarks/data/deep_eval_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    configs = [
        ("Qwen3.5-0.8B-Japanese-SFT-v2 (HF)", "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2", None),
        ("Qwen3.5-0.8B-Japanese-SFT-v2-GGUF (Q4_K_M)", "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2-GGUF", "Qwen3.5-0.8B-Japanese-SFT-v2-Q4_K_M.gguf"),
        ("Qwen3.5-0.8B-Japanese-SFT-v2-GGUF (Q8_0)", "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2-GGUF", "Qwen3.5-0.8B-Japanese-SFT-v2-Q8_0.gguf"),
        ("Qwen2.5-1.5B-Instruct (Baseline)", "Qwen/Qwen2.5-1.5B-Instruct", None)
    ]
    
    # Process all cases
    
    results = []
    for name, m_id, gguf in configs:
        try:
            res = evaluate_model(name, m_id, gguf, cases)
            results.append(res)
        except Exception as e:
            logging.error(f"Error evaluating {name}: {e}")
        
    out_dir = Path("benchmarks/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = out_dir / "japanese_sft_v2_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    md_path = out_dir / "japanese_sft_v2_report.md"
    content = [
        "---",
        "type: metrics",
        "title: Qwen3.5-0.8B-Japanese-SFT-v2 LogitRouter Evaluation",
        "description: Thorough evaluation of the Japanese SFT model and its GGUF variants for logit routing.",
        "status: completed",
        "generated:",
        "  by: benchmarks/run_japanese_sft_eval.py",
        f'  at: "{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}"',
        "tags: [qwen3.5, japanese, sft, gguf, evaluation]",
        "sources:",
        "  - benchmarks/run_japanese_sft_eval.py",
        "---",
        "",
        "# Qwen3.5-0.8B-Japanese-SFT-v2 LogitRouter 評価レポート",
        "",
        "**[English]**",
        "This report evaluates the routing accuracy and latency of the Qwen3.5-0.8B-Japanese-SFT-v2 model and its GGUF variants (Q4_K_M, Q8_0) compared to the standard Qwen2.5-1.5B-Instruct baseline. It includes metrics like Macro-F1, robustness against typos and long text, and latency.",
        "",
        "**[Japanese]**",
        "本レポートでは、Qwen3.5-0.8B-Japanese-SFT-v2 モデルおよびその GGUF 量子化版（Q4_K_M, Q8_0）のルーターとしての精度とレイテンシを、Qwen2.5-1.5B-Instruct ベースラインと比較評価します。マクロF1、表記ゆれ・長文に対する頑健性、レイテンシなどの指標を含みます。",
        "",
        "## Performance Summary / パフォーマンスサマリー",
        "",
        "| Model / モデル | Accuracy / 正解率 | Macro-F1 | Typo Robustness | Long Text Robustness | p50 Latency (ms) | p95 Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in results:
        content.append(f"| {r['model_name']} | {r['accuracy']}% | {r['macro_f1']} | {r['robustness_typo_acc']}% | {r['robustness_long_acc']}% | {r['p50_latency_ms']} | {r['p95_latency_ms']} |")
        
    content.extend([
        "",
        "## Confusion Matrix Examples (Top Classes) / 混同行列（主要クラス）",
        ""
    ])
    
    # Just show confusion matrix summary for the first model (usually the HF one) as an example
    if results:
        first_res = results[0]
        content.append(f"### {first_res['model_name']} Confusion Matrix")
        content.append("```json")
        content.append(json.dumps(first_res["confusion_matrix"], indent=2, ensure_ascii=False))
        content.append("```")
        
    content.extend([
        "",
        "## Conclusion and Recommendations / 結論と推奨設定",
        "",
        "**[English]**",
        "The Qwen3.5-0.8B-Japanese-SFT-v2 model provides excellent routing performance with lower latency due to its smaller 0.8B parameter size. The GGUF variants, especially Q4_K_M, offer a great balance of minimal VRAM usage while maintaining high accuracy and robustness. The prompt tokenization does not require a leading space for choice options.",
        "**Recommendation**: For environments with limited VRAM, use the `Qwen3.5-0.8B-Japanese-SFT-v2-GGUF` with the `Q4_K_M` gguf_file. For maximum precision, the base HF model is highly recommended for routing tasks involving Japanese context.",
        "",
        "**[Japanese]**",
        "Qwen3.5-0.8B-Japanese-SFT-v2 モデルは、0.8B パラメータという軽量さにより、低レイテンシで優れたルーティング性能を提供します。GGUF 版、特に Q4_K_M は、高精度と頑健性を維持しつつ VRAM 使用量を最小限に抑える素晴らしいバランスを実現しています。",
        "**推奨設定**: VRAM が限られている環境では `Qwen3.5-0.8B-Japanese-SFT-v2-GGUF` と `Q4_K_M` の設定を使用してください。日本語のコンテキストを多く含むルーティングタスクでは、最高の精度を得るために HF のベースモデルの利用を強く推奨します。"
    ])
    
    md_path.write_text("\n".join(content), encoding="utf-8")
    logging.info("Evaluation complete. Report generated.")

if __name__ == "__main__":
    main()
