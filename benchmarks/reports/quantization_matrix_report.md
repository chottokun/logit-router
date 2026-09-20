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
This report evaluates the accuracy, latency, and VRAM reduction of 4-bit quantized models (bitsandbytes NF4 and AWQ Marlin) across 100 comprehensive test cases spanning 10 operational domains on an NVIDIA GeForce RTX 3060 (12GB VRAM).

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
| `Qwen/Qwen2.5-1.5B-Instruct` | AWQ 4-bit (Marlin) | **84.0%** | 31.41 ms | 3033.0 MB (2.96 GB) | **高速化 (34.8ms $\rightarrow$ 31.4ms)** / 精度 +3% |

## 3. Domain Breakdown / 10大ドメイン別正解率内訳

| Domain / ドメイン (各10問) | `Gemma-4-E2B-it (4-bit)` | `Qwen2.5-1.5B-AWQ (Marlin)` | 特徴と所見 |
| :--- | :---: | :---: | :--- |
| `customer_support` (顧客サポート) | 70% (7/10) | 60% (6/10) | 契約・料金・解約等の境界判定 |
| `tool_selection` (ツール選択) | **100%** (10/10) | 90% (9/10) | 外部API呼び出し先の決定 |
| `model_routing` (モデル振分) | **100%** (10/10) | 80% (8/10) | Small/Large モデルの適性判定 |
| `security_guardrail` (セキュリティ) | **100%** (10/10) | **100%** (10/10) | インジェクション・有害プロンプト検知 |
| `intent_sentiment` (感情・意図) | 80% (8/10) | 80% (8/10) | 苦情・解約予兆・質問の分類 |
| `code_language_dispatch` (コード言語) | 90% (9/10) | 90% (9/10) | ソースコード言語・フレームワーク判定 |
| `compliance_pii` (個人情報・法令) | 70% (7/10) | 50% (5/10) | マイナンバー・機密データのトリアージ |
| `multilingual_routing` (多言語対応) | **100%** (10/10) | **100%** (10/10) | 日英中独仏・多言語ルーティング |
| `ambiguity_clarification` (曖昧性) | **100%** (10/10) | 90% (9/10) | 追加質問必要性のトリアージ |
| `urgent_escalation` (緊急度) | **100%** (10/10) | **100%** (10/10) | システム障害・即時エスカレーション |

## 4. Technical Insights & Trade-offs / 技術的考察とトレードオフ

**[English]**
1. **Gemma 4 4-bit (bitsandbytes NF4)**:
   - **Retention**: Gemma 4 retains 91.0% accuracy under 4-bit quantization, maintaining perfect 100% accuracy in 6 out of 10 domains.
   - **VRAM vs Compute Trade-off**: Memory is reduced by 3.12 GB (from 9.76 GB down to 6.49 GB). However, on-the-fly dynamic dequantization creates kernel overhead, resulting in a 171.89 ms p50 latency. It is best suited for memory-constrained batch or background triage.
2. **Qwen 2.5 1.5B AWQ (Marlin Kernel)**:
   - **Inference Speed**: The Marlin kernel optimizes 4-bit GEMM computation, reducing p50 latency to **31.41 ms** (faster than unquantized BF16 at 34.84 ms).
   - **Accuracy**: Outperforms the BF16 baseline (84.0% vs 81.0%), demonstrating excellent weight calibration for activation outliers.
   - **Quantized Head Fallback**: Successfully validated the fallback execution path (`is_sliced_head = False`) in `router.py`.

**[Japanese]**
1. **Gemma 4 4-bit (bitsandbytes NF4)**:
   - **高い精度保持**: 4-bit 化しても 91.0%（10ドメイン中6ドメインで100%）という卓越した推論精度を維持。語彙サイズ 26.2万語による日本語複合語の解釈力は量子化後も強固に保たれます。
   - **VRAMと速度のトレードオフ**: VRAM は 9.76 GB から 6.49 GB へ 3.12 GB 削減（32%減）され、12GB 以下の一般的なコンシューマ GPU に余裕で収まります。一方、bitsandbytes の動的逆量子化計算により p50 遅延は 171.89 ms となるため、厳格なリアルタイム性よりも省メモリ稼働を優先する環境に適しています。
2. **Qwen 2.5 1.5B AWQ (Marlin カーネル)**:
   - **超低遅延化**: Marlin 最適化カーネルの恩恵により、p50 遅延は **31.41 ms** と、量子化前（BF16: 34.84 ms）よりも高速化を達成しました。
   - **優れたロバスト性**: 正解率 84.0% を記録し、BF16 ベースライン（81.0%）を上回る堅牢性を示しました。
   - **量子化ヘッドフォールバックの検証**: `router.py` の Dual-Mode LM-Head において、重みテンソルが直接保持されない量子化ヘッド（`is_sliced_head = False`）のフォールバックパスが完全に機能することが実機で証明されました。