---
type: metrics
title: Quantization Benchmark Report (4-bit & AWQ) / 量子化実機ベンチマークレポート
description: Empirical evaluation of 4-bit (bitsandbytes) and AWQ (Marlin) quantization on RTX 3060 / RTX 3060における4-bit量子化モデルの実機評価
status: completed
generated:
  by: benchmarks/bench_quantization_matrix.py
  at: "2026-09-20T07:50:21Z"
tags: [quantization, 4bit, awq, marlin, bitsandbytes, on-device]
sources:
  - benchmarks/bench_quantization_matrix.py
  - benchmarks/data/deep_eval_cases.json
---

# Quantization Benchmark Report / 量子化モデル実機ベンチマークレポート

**[English]**
This report evaluates the accuracy, latency, and VRAM reduction of 4-bit quantized models (bitsandbytes NF4 and AWQ Marlin) across 100 comprehensive test cases on an NVIDIA GeForce RTX 3060 (12GB VRAM).

**[Japanese]**
本レポートは、NVIDIA GeForce RTX 3060 (12GB VRAM) 実機環境において、4-bit 量子化モデル（bitsandbytes NF4 および AWQ Marlin）を10大ドメイン・計100問の実務データセットで評価し、FP16/BF16 ベースラインとの精度・レイテンシ・VRAM削減効果を実機実測した結果です。

## 1. Quantized Models Performance Summary / 量子化モデル性能比較サマリー (100問実測)

| Configuration / 設定 | 量子化手法 | Accuracy / 正解率 | p50 Latency | Mean Latency | Peak VRAM | 平均確信度 | 平均エントロピー |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Gemma-4-E2B-it (4-bit bitsandbytes)`** | bitsandbytes 4-bit (NF4) | **91.0%** (91/100) | **171.89 ms** | 166.17 ms | **6643.8 MB** (6.49 GB) | 0.9801 | 0.0588 |
| **`Qwen2.5-1.5B-Instruct-AWQ (Marlin 4-bit)`** | AWQ 4-bit (Marlin) | **84.0%** (84/100) | **31.41 ms** | 35.78 ms | **3033.0 MB** (2.96 GB) | 0.8544 | 0.3901 |

## 2. Baseline Comparison (FP16/BF16 vs 4-bit) / ベースラインとの比較

| モデル系列 | 設定 | 正解率 | p50 レイテンシ | ピーク VRAM | VRAM 削減量 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-4-E2B-it`** | BF16 (Baseline) | **95.0%** | 67.83 ms | 9765.8 MB (9.54 GB) | - |
| **`google/gemma-4-E2B-it`** | 4-bit (bitsandbytes) | **91.0%** | 171.89 ms | 6643.8 MB (6.49 GB) | **-3122.0 MB (約 32.0% 削減)** |
| `Qwen/Qwen2.5-1.5B-Instruct` | BF16 (Baseline) | **81.0%** | 34.84 ms | 2964.1 MB (2.89 GB) | - |
| `Qwen/Qwen2.5-1.5B-Instruct` | AWQ 4-bit (Marlin) | **84.0%** | 31.41 ms | 3033.0 MB (2.96 GB) | **--68.9 MB (約 -2.3% 削減)** |