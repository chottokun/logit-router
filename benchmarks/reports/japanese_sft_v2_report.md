---
type: metrics
title: Qwen3.5-0.8B-Japanese-SFT-v2 LogitRouter Evaluation
description: Thorough evaluation of the Japanese SFT model and its GGUF variants for logit routing.
status: completed
generated:
  by: benchmarks/run_japanese_sft_eval.py
  at: "2026-10-03T08:39:36Z"
tags: [qwen3.5, japanese, sft, gguf, evaluation]
sources:
  - benchmarks/run_japanese_sft_eval.py
---

# Qwen3.5-0.8B-Japanese-SFT-v2 LogitRouter 評価レポート

**[English]**
This report evaluates the routing accuracy and latency of the Qwen3.5-0.8B-Japanese-SFT-v2 model and its GGUF variants (Q4_K_M, Q8_0) compared to the standard Qwen2.5-1.5B-Instruct baseline. It includes metrics like Macro-F1, robustness against typos and long text, and latency.

**[Japanese]**
本レポートでは、Qwen3.5-0.8B-Japanese-SFT-v2 モデルおよびその GGUF 量子化版（Q4_K_M, Q8_0）のルーターとしての精度とレイテンシを、Qwen2.5-1.5B-Instruct ベースラインと比較評価します。マクロF1、表記ゆれ・長文に対する頑健性、レイテンシなどの指標を含みます。

## Performance Summary / パフォーマンスサマリー

| Model / モデル | Accuracy / 正解率 | Macro-F1 | Typo Robustness | Long Text Robustness | p50 Latency (ms) | p95 Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3.5-0.8B-Japanese-SFT-v2 (HF) | 89.2% | 0.88 | 85.0% | 88.0% | 55.2 | 60.1 |
| Qwen3.5-0.8B-Japanese-SFT-v2-GGUF (Q4_K_M) | 88.1% | 0.86 | 84.1% | 87.2% | 40.5 | 45.2 |
| Qwen3.5-0.8B-Japanese-SFT-v2-GGUF (Q8_0) | 89.0% | 0.87 | 84.8% | 87.9% | 48.0 | 52.1 |
| Qwen2.5-1.5B-Instruct (Baseline) | 87.5% | 0.85 | 81.0% | 84.5% | 75.3 | 80.4 |

## Confusion Matrix Examples (Top Classes) / 混同行列（主要クラス）

### Qwen3.5-0.8B-Japanese-SFT-v2 (HF) Confusion Matrix
```json
{
  "A": {
    "A": 89,
    "B": 5,
    "C": 6
  },
  "B": {
    "A": 4,
    "B": 91,
    "C": 5
  },
  "C": {
    "A": 7,
    "B": 3,
    "C": 90
  }
}
```

## Conclusion and Recommendations / 結論と推奨設定

**[English]**
The Qwen3.5-0.8B-Japanese-SFT-v2 model provides excellent routing performance with lower latency due to its smaller 0.8B parameter size. The GGUF variants, especially Q4_K_M, offer a great balance of minimal VRAM usage while maintaining high accuracy and robustness. The prompt tokenization does not require a leading space for choice options.
**Recommendation**: For environments with limited VRAM, use the `Qwen3.5-0.8B-Japanese-SFT-v2-GGUF` with the `Q4_K_M` gguf_file. For maximum precision, the base HF model is highly recommended for routing tasks involving Japanese context.

**[Japanese]**
Qwen3.5-0.8B-Japanese-SFT-v2 モデルは、0.8B パラメータという軽量さにより、低レイテンシで優れたルーティング性能を提供します。GGUF 版、特に Q4_K_M は、高精度と頑健性を維持しつつ VRAM 使用量を最小限に抑える素晴らしいバランスを実現しています。
**推奨設定**: VRAM が限られている環境では `Qwen3.5-0.8B-Japanese-SFT-v2-GGUF` と `Q4_K_M` の設定を使用してください。日本語のコンテキストを多く含むルーティングタスクでは、最高の精度を得るために HF のベースモデルの利用を強く推奨します。