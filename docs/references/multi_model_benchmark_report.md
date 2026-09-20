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

> [!IMPORTANT]
> **最新の包括的評価について (Latest Comprehensive Evaluation)**:
> 本ドキュメントは初期50問の予備測定レポートです。`google/gemma-4-E2B-it` や `SmolLM2-360M-Instruct` を含め、10大ドメイン・100問で実施した最新の深層実機評価は **[Deep 10-Domain Benchmark Report](deep_eval_report.md)**、および通常生成との比較実測は **[Comparison vs Generation Report](comparison_vs_generation_report.md)** をご覧ください。

**[English]**
This report presents strictly measured on-device performance metrics comparing multiple model sizes on an NVIDIA GeForce RTX 3060 (12GB VRAM). All evaluations were executed using PyTorch SDPA with bfloat16 precision across a benchmark dataset spanning multiple domains.

**[Japanese]**
本レポートは、NVIDIA GeForce RTX 3060 (12GB VRAM) 実機環境において、複数モデルを同一データセットで実行・測定した実測ベンチマーク結果を記録したものです。推測値や未測定データは一切含まれていません。

## Comprehensive Multi-Model Matrix / 全候補モデル総合比較マトリクス (100 Cases)

※ 最新の10大ドメイン100問実機測定（`benchmarks/reports/deep_eval_report.md` より）

| Model / モデル | Accuracy / 正解率 | p50 Latency | Mean Latency | Peak VRAM | Mean Conf | 特徴・日本語適合性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-4-E2B-it`** | **95.0%** (95/100) | **67.83 ms** | 75.27 ms | 9765.8 MB | 0.9945 | **最高精度**。日本語語彙圧縮・長文ニュアンス判定に優れた特性 |
| `Qwen/Qwen2.5-3B-Instruct` | **85.0%** (85/100) | **62.76 ms** | 60.44 ms | 5908.5 MB | 0.9688 | バランス型。エントロピー分離度良好 |
| `Qwen/Qwen2.5-1.5B-Instruct` | **81.0%** (81/100) | **34.84 ms** | 35.04 ms | 2964.1 MB | 0.8656 | 高速（34ms）。第1層カスケードゲートに最適 |
| `Qwen/Qwen2.5-0.5B-Instruct` | **73.0%** (73/100) | **16.26 ms** | 18.66 ms | 957.5 MB | 0.7331 | 最速（16ms）。構文判定は良好だが行間解釈に限界 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | **23.0%** (23/100) | **23.76 ms** | 29.19 ms | 706.2 MB | 0.4275 | 日本語・複合推論で大幅な過信誤分類が発生 |

## Preliminary 50-Case Benchmark (Legacy) / 初回50問予備ベンチマーク

| Model / モデル | Accuracy / 正解率 | Mean Latency / 平均遅延 | p50 / 中央値 | p95 / 95%値 | Peak VRAM / ピークメモリ | Mean Conf / 平均確信度 | Mean Ent / 平均エントロピー |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B-Instruct` | **66.0%** (33/50) | 26.91 ms | 18.28 ms | 53.23 ms | 955.6 MB | 0.7557 | 0.5189 |
| `Qwen/Qwen2.5-1.5B-Instruct` | **82.0%** (41/50) | 29.88 ms | 25.25 ms | 55.93 ms | 2961.1 MB | 0.9144 | 0.2180 |
| `Qwen/Qwen2.5-3B-Instruct` | **84.0%** (42/50) | 60.45 ms | 55.86 ms | 87.34 ms | 5904.7 MB | 0.9745 | 0.0569 |

## Domain-Specific Accuracy (50 Cases) / ドメイン別正解率比較 (50問)

| Domain / ドメイン | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` |
| :--- | :--- | :--- | :--- |
| `customer_support` | 60.0% (6/10) | 90.0% (9/10) | 80.0% (8/10) |
| `tool_selection` | 90.0% (9/10) | 100.0% (10/10) | 90.0% (9/10) |
| `model_routing` | 70.0% (7/10) | 60.0% (6/10) | 90.0% (9/10) |
| `security_guardrail` | 60.0% (6/10) | 60.0% (6/10) | 60.0% (6/10) |
| `intent_sentiment` | 50.0% (5/10) | 100.0% (10/10) | 100.0% (10/10) |