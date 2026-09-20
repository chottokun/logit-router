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

For details on profiling protocols and metrics, refer to:
- [Benchmarks and Metrics](benchmarks_and_metrics.md)
- [Deep Evaluation Suite Specification](deep_benchmarks_specification.md)

Empirical on-device reports:
- [Deep 10-Domain Benchmark Report](deep_eval_report.md) (100 cases across 5 models, Gemma 4: 95.0% accuracy)
- [Comparison vs Standard Generation](comparison_vs_generation_report.md) (LogitRouter vs model.generate speedup)
- [Multi-Model Benchmark Report](multi_model_benchmark_report.md) (0.5B vs 1.5B vs 3B latency & accuracy)
- [Cross-Model Robustness Report](robustness_matrix_report.md) (Permutation consistency & OOD separation)
- [Model Evaluation Report](eval_report.md)

**[Japanese]**
- **レイテンシプロファイリング**: CUDAイベント同期を用いた、ホスト-デバイス間転送、Transformerフォワードパス、および語彙スライス射影層の実行時間測定。
- **頑健性と感度評価**: 候補順序の入れ替え（位置バイアス）に対する一貫性、および分布外（OOD）入力時における情報エントロピーの変動検証。
- **タスク別評価**: 検証データセットを用いたドメイン別の分類正解率および遅延特性の測定。

計測プロトコルおよび評価指標の詳細：
- [Benchmarks and Metrics (ベンチマークと評価指標)](benchmarks_and_metrics.md)
- [Deep Evaluation Suite Specification (深層評価スイート仕様書)](deep_benchmarks_specification.md)

実機実測レポート：
- [Deep 10-Domain Benchmark Report (10大ドメイン100問深層実機評価レポート)](deep_eval_report.md) (Gemma 4 で正解率 95.0% 実証)
- [Comparison vs Standard Generation (通常生成との実機比較実測レポート)](comparison_vs_generation_report.md)
- [Multi-Model Benchmark Report (複数モデル横断実機比較レポート)](multi_model_benchmark_report.md)
- [Cross-Model Robustness Report (モデル別頑健性・位置バイアス実機評価レポート)](robustness_matrix_report.md)
- [Model Evaluation Report (モデル評価レポート)](eval_report.md)
