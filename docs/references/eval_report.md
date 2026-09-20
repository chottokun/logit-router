---
type: metrics
title: Model Evaluation Report / モデル評価レポート
description: Empirical evaluation report of LogitRouter across benchmark tasks / LogitRouter のタスク別実機評価レポート
status: completed
generated:
  by: eval_suite.py
  at: "2026-09-20T14:52:04Z"
tags: [evaluation, logit-router, metrics]
sources:
  - benchmarks/eval_suite.py
  - benchmarks/data/eval_cases.json
---

# Model Evaluation Report / モデル評価レポート

> [!IMPORTANT]
> **最新の評価レポートについて (Latest Comprehensive Reports)**:
> 本ドキュメントは初期の単一モデル予備測定結果です。`google/gemma-4-E2B-it` を含む5モデル・10大ドメイン100問の深層評価・批判的/建設的考察は **[Deep 10-Domain Benchmark Report](deep_eval_report.md)**、通常生成との直接対決実測は **[Comparison vs Generation Report](comparison_vs_generation_report.md)** をご覧ください。

**[English]**
This document presents benchmark evaluation metrics for `LogitRouter` executed on a local workstation. The evaluation covers overall accuracy, inference latency distributions, prediction confidence, entropy statistics, and error analysis across five domains.

**[Japanese]**
本ドキュメントは、ローカル環境で計測された `LogitRouter` のベンチマーク評価指標を記録したものです。5つのタスクドメインを対象に、全体正解率、推論遅延の統計値、確信度およびエントロピーの分布、ならびに誤分類事例の分析を示します。

## Evaluation Configuration (Legacy) / 評価環境と設定

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Device**: NVIDIA GeForce RTX 3060 (12GB VRAM, CUDA)
- **Dataset**: `benchmarks/data/eval_cases.json`
- **Evaluation Size**: 50 cases (10 cases per domain)

## Overall Metrics / 総合サマリー

| Metric / 指標 | Measured Value / 測定値 | Description / 備考 |
| :--- | :--- | :--- |
| Overall Accuracy / 全体正解率 | **80.00%** (40 / 50) | Exact match with ground truth |
| Mean Latency / 平均レイテンシ | 55.07 ms | End-to-end inference per query |
| Min Latency / 最小レイテンシ | 23.12 ms | Warm cache execution |
| Max Latency / 最大レイテンシ | 1238.11 ms | Includes first-pass memory allocation overhead |
| Mean Confidence / 平均確信度 | 0.8820 | Average max softmax probability |
| Mean Entropy / 平均エントロピー | 0.2941 | Shannon entropy across candidate distribution |

## Domain Breakdown / ドメイン別集計

| Domain / ドメイン | Accuracy / 正解率 | Correct / Total (件数) | Notes / 特性 |
| :--- | :--- | :--- | :--- |
| `customer_support` | 90.00% | 9 / 10 | High precision across general billing/ops |
| `tool_selection` | 100.00% | 10 / 10 | Clear syntactic and functional boundaries |
| `model_routing` | 50.00% | 5 / 10 | Difficulty in separating small vs. large model boundaries |
| `security_guardrail` | 70.00% | 7 / 10 | Tendency toward false positives on complex prompts |
| `intent_sentiment` | 90.00% | 9 / 10 | Consistent sentiment classification |

## Misclassification Analysis / 誤分類事例の分析

| Expected / 正解ラベル | Predicted / 予測ラベル | Confidence / 確信度 | Entropy / エントロピー | Domain / ドメイン |
| :--- | :--- | :--- | :--- | :--- |
| `決済・請求窓口` | `一般的な操作案内` | 0.6016 | 0.7884 | customer_support |
| `Small_Fast_Model` | `Large_Reasoning_Model` | 0.6367 | 0.6553 | model_routing |
| `Large_Reasoning_Model` | `Small_Fast_Model` | 0.7305 | 0.5828 | model_routing |
| `Small_Fast_Model` | `Large_Reasoning_Model` | 0.7969 | 0.5041 | model_routing |
| `Small_Fast_Model` | `Large_Reasoning_Model` | 0.8516 | 0.4200 | model_routing |
| `Small_Fast_Model` | `Large_Reasoning_Model` | 0.6914 | 0.6176 | model_routing |
| `Safe_Process` | `Unsafe_Reject` | 0.6523 | 0.6461 | security_guardrail |
| `Safe_Process` | `Unsafe_Reject` | 0.7969 | 0.5041 | security_guardrail |
| `Safe_Process` | `Unsafe_Reject` | 0.8281 | 0.4596 | security_guardrail |
| `Question` | `Purchase_Intent` | 0.9648 | 0.1889 | intent_sentiment |
