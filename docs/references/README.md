---
type: concept
title: References Overview / リファレンス概要
description: Overview of benchmarks, metrics, and evaluation documentation / ベンチマーク、評価指標、および検証結果の概要
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T15:00:00Z"
tags:
  - references
  - overview
sources:
  - benchmarks/bench_latency.py
  - benchmarks/bench_profile.py
---

# References Overview / リファレンス概要

**[English]**
This section documents empirical benchmarks, profiling protocols, and validation datasets used to evaluate the Logit Router. It provides quantitative performance measurements including inference latency, layer decomposition, classification accuracy, and input robustness.

**[Japanese]**
本セクションでは、Logit Router の評価に用いた実測ベンチマーク、プロファイリング手順、および検証データセットについて記述します。推論遅延、処理層別の要因分解、分類精度、入力に対する頑健性などの定量的な測定結果を提示します。

## Benchmarks and Methodologies / ベンチマークと検証項目

**[English]**
- **Latency Profiling**: Measurement of host-device transfers, Transformer forward passes, and sliced projection layers using CUDA event synchronization.
- **Robustness and Sensitivity**: Quantitative evaluation of candidate permutation order (position bias) and entropy shifts on out-of-distribution (OOD) inputs.
- **Task Evaluation**: Domain-specific classification accuracy and latency characteristics across evaluation datasets.

For details on profiling protocols and metrics, refer to [Benchmarks and Metrics](benchmarks_and_metrics.md).
For empirical accuracy and latency measurements across evaluation tasks, see [Model Evaluation Report](eval_report.md).

**[Japanese]**
- **レイテンシプロファイリング**: CUDAイベント同期を用いた、ホスト-デバイス間転送、Transformerフォワードパス、および語彙スライス射影層の実行時間測定。
- **頑健性と感度評価**: 候補順序の入れ替え（位置バイアス）に対する一貫性、および分布外（OOD）入力時における情報エントロピーの変動検証。
- **タスク別評価**: 検証データセットを用いたドメイン別の分類正解率および遅延特性の測定。

計測プロトコルおよび評価指標の詳細については [Benchmarks and Metrics (ベンチマークと評価指標)](benchmarks_and_metrics.md) を参照してください。
検証タスクにおける正解率および測定結果については [Model Evaluation Report (モデル評価レポート)](eval_report.md) を参照してください。
