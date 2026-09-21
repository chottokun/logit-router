---
type: metrics
title: VisionLogitRouter vs Standard Generation Benchmark Report / VisionLogitRouter vs 通常生成 実機比較レポート
description: Rigorous measured speedup of single forward pass logit extraction vs autoregressive generate() for multimodal models.
status: completed
generated:
  by: benchmarks/bench_vision_vs_autoregressive.py
  at: "2026-09-21T01:46:16Z"
tags: [speedup, autoregressive, vision, multimodal, on-device]
sources:
  - benchmarks/bench_vision_vs_autoregressive.py
---

# VisionLogitRouter vs Standard Generation Benchmark Report / 視覚モデル通常自己回帰生成との実機比較レポート

**[English]**
This report provides strictly measured empirical comparison between VisionLogitRouter (single forward pass prefill) and standard Hugging Face `model.generate()` (autoregressive generation) using identical model weights, prompts, synthetic PIL images, and hardware (cuda).

**[Japanese]**
本レポートは、同一のモデル重み、プロンプト、合成PIL画像、およびハードウェア環境（cuda）において、VisionLogitRouter（単一フォワードパスPrefill）と標準の Hugging Face `model.generate()`（自己回帰逐次生成）を実行し、レイテンシと高速化倍率を直接対決（A/Bテスト）で実機実測した結果です。

## Comparative Latency & Speedup Summary / 比較サマリー（実測値）

| Model / モデル | VisionLogitRouter (p50 / Mean) | 通常生成: 最短5トークン (p50 / Mean) | 通常生成: 簡潔推論30トークン (p50 / Mean) | 高速化倍率 (vs 最短生成) | 高速化倍率 (vs 簡潔推論) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `google/gemma-4-E2B-it` | **765.23 ms** / 704.38 ms | 1566.19 ms / 1536.41 ms | 1584.3 ms / 1560.18 ms | **2.18x 高速** | **2.21x 高速** |

## Technical Analysis / 技術的要因の分析

**[English]**
- **Autoregressive Overhead**: Multimodal `model.generate()` incurs multiple sequential kernel launches and KV cache allocations. For vision models, the cross-attention or large visual token sequence adds significant overhead to every generation step.
- **Single Forward Pass Advantage**: VisionLogitRouter halts immediately after the initial prefill phase. It processes the visual tokens and text prompt in a single pass and directly extracts logits for the candidate choices, bypassing the autoregressive generation loop entirely.

**[Japanese]**
- **自己回帰生成のオーバーヘッド**: マルチモーダルモデルの `model.generate()` では、逐次カーネル起動やKVキャッシュ確保に加え、視覚トークンの処理によるオーバーヘッドが生成ステップごとに発生します。
- **単一フォワードパスの優位性**: VisionLogitRouter は最初のPrefill完了直後に処理を完了します。画像とテキストのプロンプトを1回のパスで処理し、選択肢トークンのロジットを直接抽出するため、自己回帰のループを完全に回避できます。