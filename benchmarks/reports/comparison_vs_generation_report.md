---
type: metrics
title: LogitRouter vs Standard Generation Benchmark Report / LogitRouter vs 通常生成 実機比較レポート
description: Rigorous measured speedup of single forward pass logit extraction vs autoregressive generate() on RTX 3060 / RTX 3060での通常生成との実機比較実測値
status: completed
generated:
  by: benchmarks/bench_vs_autoregressive.py
  at: "2026-09-20T06:31:30Z"
tags: [speedup, autoregressive, generation, comparison, on-device]
sources:
  - benchmarks/bench_vs_autoregressive.py
---

# LogitRouter vs Standard Generation Benchmark Report / 通常自己回帰生成との実機比較レポート

**[English]**
This report provides strictly measured empirical comparison between LogitRouter (single forward pass prefill) and standard Hugging Face `model.generate()` (autoregressive generation) using identical model weights, prompts, and hardware (NVIDIA GeForce RTX 3060 12GB).

**[Japanese]**
本レポートは、同一のモデル重み、プロンプト、およびハードウェア環境（NVIDIA GeForce RTX 3060 12GB）において、LogitRouter（単一フォワードパスPrefill）と標準の Hugging Face `model.generate()`（自己回帰逐次生成）を実行し、レイテンシと高速化倍率を直接対決（A/Bテスト）で実機実測した結果です。

## Comparative Latency & Speedup Summary / 比較サマリー（実測値）

| Model / モデル | LogitRouter (p50 / Mean) | 通常生成: 最短1文字 (p50 / Mean) | 通常生成: 簡潔推論 (p50 / Mean) | 高速化倍率 (vs 最短生成) | 高速化倍率 (vs 簡潔推論) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `google/gemma-4-E2B-it` | **103.88 ms** / 108.71 ms | 621.49 ms / 628.16 ms | 622.95 ms / 619.2 ms | **5.78x 高速** | **5.7x 高速** |
| `Qwen/Qwen2.5-0.5B-Instruct` | **50.65 ms** / 59.49 ms | 101.96 ms / 97.86 ms | 94.99 ms / 92.08 ms | **1.64x 高速** | **1.55x 高速** |
| `Qwen/Qwen2.5-1.5B-Instruct` | **90.33 ms** / 103.7 ms | 206.74 ms / 241.02 ms | 206.99 ms / 233.15 ms | **2.32x 高速** | **2.25x 高速** |
| `Qwen/Qwen2.5-3B-Instruct` | **85.04 ms** / 96.45 ms | 200.42 ms / 228.07 ms | 197.7 ms / 219.68 ms | **2.36x 高速** | **2.28x 高速** |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | **25.97 ms** / 27.95 ms | 133.01 ms / 131.9 ms | 170.0 ms / 283.7 ms | **4.72x 高速** | **10.15x 高速** |

## Technical Analysis / 技術的要因の分析

**[English]**
- **Autoregressive Overhead**: `model.generate()` incurs multiple sequential kernel launches, KV cache memory allocations, and memory bandwidth contention for every additional token generated.
- **Single Forward Pass Advantage**: LogitRouter halts immediately after the prompt prefill phase, reducing the execution to exactly 1 Transformer forward pass plus a sliced linear projection of candidate tokens.

**[Japanese]**
- **自己回帰生成のオーバーヘッド**: 通常の `model.generate()` では、1トークン生成するごとに逐次カーネル起動、KVキャッシュ確保、およびメモリストリーミング（メモリバウンド制約）が発生します。
- **単一フォワードパスの優位性**: LogitRouter はプロンプトのPrefill完了直後に処理を完了し、全語彙（15万語）への射影を行わず選択肢トークンのみにスライスしてロジットを抽出するため、オーバーヘッドが極小化されます。