---
type: metrics
title: Model Evaluation Report
description: LogitRouter モデルの包括的評価レポート
status: completed
generated:
  by: eval_suite.py
  at: 2026-09-20T14:52:04.647836
tags: [evaluation, logit-router, metrics]
sources: []
---

# モデル評価レポート

> [!IMPORTANT]
> **最新の評価レポートについて**:
> 本ドキュメントは初期の単一モデル（Qwen 1.5B / 50問）の予備レポートです。`google/gemma-4-E2B-it` を含む5モデル横断・10大ドメイン100問の深層実機評価と考察は **[10大ドメイン100問深層実機評価レポート](deep_eval_report.md)**、通常生成との直接対決実測は **[通常生成との速度・レイテンシ比較レポート](comparison_vs_generation_report.md)** をご覧ください。

## 評価設定 (Legacy / 予備測定)

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Dataset**: `benchmarks/data/eval_cases.json`
- **Device**: `Auto`
- **Total Cases**: `50`

## 総合サマリー

| メトリクス | 値 |
| :--- | :--- |
| 全体正解率 (Overall Accuracy) | **80.00%** (40/50) |
| 平均レイテンシ (Avg Latency) | 55.07 ms |
| 最小レイテンシ (Min Latency) | 23.12 ms |
| 最大レイテンシ (Max Latency) | 1238.11 ms |
| 平均確信度 (Avg Confidence) | 0.8820 |
| 平均エントロピー (Avg Entropy) | 0.2941 |

## ドメイン別サマリー

| ドメイン | 正解率 | 正解数 / 総数 |
| :--- | :--- | :--- |
| customer_support | 90.00% | 9 / 10 |
| tool_selection | 100.00% | 10 / 10 |
| model_routing | 50.00% | 5 / 10 |
| security_guardrail | 70.00% | 7 / 10 |
| intent_sentiment | 90.00% | 9 / 10 |

## 誤分類分析 (Failure Cases)

| 入力 (Input) | 期待値 (Expected) | 予測値 (Predicted) | 確信度 | エントロピー | ドメイン |
| :--- | :--- | :--- | :--- | :--- | :--- |
|  | `決済・請求窓口` | `一般的な操作案内` | 0.6016 | 0.7884 | customer_support |
|  | `Small_Fast_Model` | `Large_Reasoning_Model` | 0.6367 | 0.6553 | model_routing |
|  | `Large_Reasoning_Model` | `Small_Fast_Model` | 0.7305 | 0.5828 | model_routing |
|  | `Small_Fast_Model` | `Large_Reasoning_Model` | 0.7969 | 0.5041 | model_routing |
|  | `Small_Fast_Model` | `Large_Reasoning_Model` | 0.8516 | 0.4200 | model_routing |
|  | `Small_Fast_Model` | `Large_Reasoning_Model` | 0.6914 | 0.6176 | model_routing |
|  | `Safe_Process` | `Unsafe_Reject` | 0.6523 | 0.6461 | security_guardrail |
|  | `Safe_Process` | `Unsafe_Reject` | 0.7969 | 0.5041 | security_guardrail |
|  | `Safe_Process` | `Unsafe_Reject` | 0.8281 | 0.4596 | security_guardrail |
|  | `Question` | `Purchase_Intent` | 0.9648 | 0.1889 | intent_sentiment |
