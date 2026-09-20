---
type: concept
title: References Overview / リファレンス概要
description: Overview of benchmarks, metrics, and references / ベンチマーク、指標、およびリファレンスの概要
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - references
  - overview
sources:
  - benchmarks/bench_latency.py
  - benchmarks/bench_profile.py
---

# References Overview / リファレンス概要

**[English]**
The References section contains empirical data, benchmark methodologies, and evaluation metrics used to validate the Logit Router. It proves that the architecture achieves its intended goal of ultra-low latency routing without sacrificing accuracy.

**[Japanese]**
リファレンスセクションには、Logit Router を検証するために使用された経験的データ、ベンチマーク手法、および評価指標が含まれています。これにより、アーキテクチャが精度を犠牲にすることなく超低遅延ルーティングという意図した目標を達成していることが証明されます。

## Benchmarks and Methodologies / ベンチマークと手法

**[English]**
- **Latency Profiling**: Rigorous measurement using hardware-level events to break down the cost of tokenization, forward passes, and post-processing.
- **Robustness and Metrics**: Evaluations against position bias and metrics for handling Out-of-Distribution (OOD) data.

For deep details on profiling techniques and metrics, refer to [Benchmarks and Metrics](benchmarks_and_metrics.md).
For comprehensive model accuracy across 5 major domains, see [Model Evaluation Report](eval_report.md).

**[Japanese]**
- **レイテンシのプロファイリング**: ハードウェアレベルのイベントを使用した厳密な測定により、トークナイズ、フォワードパス、および後処理のコストを分解します。
- **堅牢性と指標**: 位置バイアスに対する評価と、分布外（OOD）データを処理するための指標。
- **包括的評価レポート**: 5大ドメイン（50問）に対する正解率、レイテンシ、誤分類分析。

プロファイリング手法と指標の詳細については、[Benchmarks and Metrics](benchmarks_and_metrics.md) を参照してください。
最新の包括的評価結果については、[Model Evaluation Report (モデル評価レポート)](eval_report.md) を参照してください。

