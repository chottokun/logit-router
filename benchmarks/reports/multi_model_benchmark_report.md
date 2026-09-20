---
type: metrics
title: Multi-Model Benchmark Comparison Report / 複数モデル横断ベンチマーク比較レポート
description: Empirical on-device benchmark results across multiple models on RTX 3060 / RTX 3060 での実機マルチモデル測定結果
status: completed
generated:
  by: benchmarks/bench_comprehensive.py
  at: "2026-09-20T06:05:48Z"
tags: [benchmarks, multi-model, latency, accuracy, on-device]
sources:
  - benchmarks/bench_comprehensive.py
  - benchmarks/data/eval_cases.json
---

# Multi-Model Benchmark Comparison Report / 複数モデル横断実機ベンチマーク比較レポート

**[English]**
This report presents strictly measured on-device performance metrics comparing multiple model sizes on an NVIDIA GeForce RTX 3060 (12GB VRAM). All evaluations were executed using PyTorch SDPA with bfloat16 precision across a 50-case benchmark dataset spanning 5 distinct domains.

**[Japanese]**
本レポートは、NVIDIA GeForce RTX 3060 (12GB VRAM) 実機環境において、複数モデルを同一の50問データセット（5大ドメイン）で実行・測定した実測ベンチマーク結果を記録したものです。推測値や未測定データは一切含まれていません。

## Overall Comparison Summary / 総合比較サマリー

| Model / モデル | Accuracy / 正解率 | Mean Latency / 平均遅延 | p50 / 中央値 | p95 / 95%値 | Peak VRAM / ピークメモリ | Mean Conf / 平均確信度 | Mean Ent / 平均エントロピー |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B-Instruct` | **66.0%** (33/50) | 26.91 ms | 18.28 ms | 53.23 ms | 955.6 MB | 0.7557 | 0.5189 |
| `Qwen/Qwen2.5-1.5B-Instruct` | **82.0%** (41/50) | 29.88 ms | 25.25 ms | 55.93 ms | 2961.1 MB | 0.9144 | 0.2180 |
| `Qwen/Qwen2.5-3B-Instruct` | **84.0%** (42/50) | 60.45 ms | 55.86 ms | 87.34 ms | 5904.7 MB | 0.9745 | 0.0569 |

## Domain-Specific Accuracy / ドメイン別正解率比較

| Domain / ドメイン | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` |
| :--- | :--- | :--- | :--- |
| `customer_support` | 60.0% (6/10) | 90.0% (9/10) | 80.0% (8/10) |
| `tool_selection` | 90.0% (9/10) | 100.0% (10/10) | 90.0% (9/10) |
| `model_routing` | 70.0% (7/10) | 60.0% (6/10) | 90.0% (9/10) |
| `security_guardrail` | 60.0% (6/10) | 60.0% (6/10) | 60.0% (6/10) |
| `intent_sentiment` | 50.0% (5/10) | 100.0% (10/10) | 100.0% (10/10) |

## Detailed Latency Distribution (ms) / 詳細レイテンシ分布

| Model / モデル | Min | Mean | p50 | p90 | p95 | p99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B-Instruct` | 12.87 | 26.91 | 18.28 | 51.38 | 53.23 | 61.56 | 63.36 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 23.56 | 29.88 | 25.25 | 45.52 | 55.93 | 58.55 | 58.64 |
| `Qwen/Qwen2.5-3B-Instruct` | 46.22 | 60.45 | 55.86 | 83.38 | 87.34 | 90.91 | 91.53 |