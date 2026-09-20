---
type: metrics
title: Cross-Model Robustness & Position Bias Matrix Report / モデル別頑健性・位置バイアス実機評価レポート
description: Measured permutation consistency and OOD entropy response across 0.5B, 1.5B, and 3B models / 3モデルにおける順列一貫性およびOODエントロピー実測結果
status: completed
generated:
  by: benchmarks/bench_robustness_matrix.py
  at: "2026-09-20T06:06:32Z"
tags: [robustness, position-bias, ood, multi-model, on-device]
sources:
  - benchmarks/bench_robustness_matrix.py
---

# Cross-Model Robustness & Position Bias Report / モデル別頑健性・位置バイアス実機評価レポート

**[English]**
Empirical evaluation of candidate position bias (exhaustive permutations) and Out-of-Distribution (OOD) rejection behavior on an NVIDIA GeForce RTX 3060 (12GB VRAM).

**[Japanese]**
NVIDIA GeForce RTX 3060 (12GB VRAM) 実機環境における、候補選択肢の提示順序バイアス（全順列検証）および分布外（OOD）入力に対するエントロピー応答の実測評価結果です。

## Robustness Metrics Summary / 頑健性評価サマリー

| Model / モデル | Support Perm Consistency (3 choices, 6 perms) | Support Entropy Delta | Tool Perm Consistency (4 choices, 24 perms) | Tool Entropy Delta | OOD Mean Entropy (Separation Signal) | OOD Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B-Instruct` | **83.33%** | 0.5797 | **100.0%** | 0.088 | **0.8752** | 0.6035 |
| `Qwen/Qwen2.5-1.5B-Instruct` | **100.0%** | 0.3461 | **100.0%** | 0.1028 | **0.5914** | 0.7797 |
| `Qwen/Qwen2.5-3B-Instruct` | **100.0%** | 0.1742 | **100.0%** | 0.0017 | **0.0225** | 0.9961 |