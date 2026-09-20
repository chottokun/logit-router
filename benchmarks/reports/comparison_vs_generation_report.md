---
type: metrics
title: LogitRouter vs Standard Generation Benchmark Report / LogitRouter vs 通常生成 実機比較レポート
description: Rigorous measured speedup of single forward pass logit extraction vs autoregressive generate() on RTX 3060 / RTX 3060での通常生成との実機比較実測値
status: completed
generated:
  by: benchmarks/bench_vs_autoregressive.py
  at: "2026-09-20T06:16:37Z"
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
| `Qwen/Qwen2.5-0.5B-Instruct` | **15.48 ms** / 17.78 ms | 33.64 ms / 48.74 ms | 33.37 ms / 44.62 ms | **2.74x 高速** | **2.51x 高速** |
| `Qwen/Qwen2.5-1.5B-Instruct` | **25.39 ms** / 30.45 ms | 57.91 ms / 63.87 ms | 47.95 ms / 52.25 ms | **2.1x 高速** | **1.72x 高速** |
| `Qwen/Qwen2.5-3B-Instruct` | **79.01 ms** / 77.86 ms | 247.89 ms / 249.06 ms | 248.63 ms / 252.3 ms | **3.2x 高速** | **3.24x 高速** |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | **23.31 ms** / 31.84 ms | 135.95 ms / 146.96 ms | 193.23 ms / 282.82 ms | **4.62x 高速** | **8.88x 高速** |

## Technical Analysis / 技術的要因の分析

**[English]**
- **Autoregressive Overhead**: `model.generate()` incurs multiple sequential kernel launches, KV cache memory allocations, and memory bandwidth contention for every additional token generated.
- **Single Forward Pass Advantage**: LogitRouter halts immediately after the prompt prefill phase, reducing the execution to exactly 1 Transformer forward pass plus a sliced linear projection of candidate tokens.

**[Japanese]**
- **自己回帰生成のオーバーヘッド**: 通常の `model.generate()` では、1トークン生成するごとに逐次カーネル起動、KVキャッシュ確保、およびメモリストリーミング（メモリバウンド制約）が発生します。
- **単一フォワードパスの優位性**: LogitRouter はプロンプトのPrefill完了直後に処理を完了し、全語彙（15万語）への射影を行わず選択肢トークンのみにスライスしてロジットを抽出するため、オーバーヘッドが極小化されます。