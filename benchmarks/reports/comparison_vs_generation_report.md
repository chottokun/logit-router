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

## Evaluation Conditions & Definition of Generation Modes / 測定モードの定義と条件の差異

**[English]**
The benchmark evaluates three distinct inference configurations to capture realistic deployment scenarios:

1. **LogitRouter (Single Forward Pass)**:
   - **Generation tokens**: Exactly **0 new tokens** (pure prefill).
   - **Mechanism**: Extracts hidden states from the final prompt token and computes a sliced matrix multiplication exclusively over the candidate indices (`A`, `B`, `C`). Fully eliminates autoregressive decoding loops, KV cache writes, and full-vocabulary Softmax operations.
2. **Standard Generation: Minimal 1-Char (`max_new_tokens=5`)**:
   - **Scenario**: Assumes an optimized operational setup where the prompt explicitly enforces single-token outputs (`Answer with only the single letter.`).
   - **Parameters**: `max_new_tokens=5, do_sample=False` (greedy decoding). The 5-token margin accommodates incidental leading whitespace or formatting prefixes before the option letter.
   - **Execution**: Incurs prompt prefill followed by at least 1–5 sequential autoregressive decode steps (each involving full-vocabulary projection and KV cache updates).
3. **Standard Generation: Brief Reasoning / CoT (`max_new_tokens=30`)**:
   - **Scenario**: Assumes a workflow where the router is expected to return a brief rationale alongside the classification (e.g., `A (Technical Support: Network outage)`).
   - **Parameters**: `max_new_tokens=30, do_sample=False`.
   - **Execution**: Runs up to 30 sequential autoregressive decode steps.

### Why Latencies Align in Advanced Models (Early Stopping Behavior)
- **High-Capacity Models (`Qwen2.5`, `Gemma 4`)**: These models adhere strictly to the system instruction (`Answer with only the option letter`). They emit the answer token followed immediately by an `<eos>` (End of Sequence) token within 1–2 steps. Consequently, in `max_new_tokens=30`, execution terminates early, resulting in latencies virtually identical to `max_new_tokens=5`.
- **Sub-1B Compact Models (`SmolLM2-360M`)**: These models exhibit weaker instruction adherence and fail to emit `<eos>` promptly, often generating repetitive justifications until hitting the token cap. Hence, `max_new_tokens=30` causes latency to swell from 133 ms to 283 ms, widening the LogitRouter speedup to **10.15x**.

**[Japanese]**
本ベンチマークでは、実運用で想定される3つの推論モードを定義し、厳密に同一のプロンプト・入力テンソルで比較しています。

1. **LogitRouter（単一フォワードパス）**:
   - **生成トークン数**: **0トークン**（完全な Prefill 単一パス）
   - **処理内容**: プロンプト末尾の隠れ状態から、選択肢インデックス（`A`, `B`, `C` 等）に対応する LM-Head 行のみ（3行）をスライスして行列積を実行。自己回帰デコードループ、KVキャッシュ確保・更新、全語彙（15万〜26万語）への Softmax を完全にバイパスします。
2. **通常生成: 最短1文字 (`max_new_tokens=5`)**:
   - **想定ユースケース**: 「選択肢の記号1文字のみで答えてください」と指示し、最速の自己回帰応答を狙う運用。
   - **設定値**: `max_new_tokens=5, do_sample=False`（Greedy探索）。モデルが出力する可能性のある先頭の空白や改行、わずかな前置き（`A` など）を許容するため上限を5に設定。
   - **処理内容**: Prefill 完了後、逐次デコードループに入り、1ステップごとに全語彙射影・サンプリング・KVキャッシュ更新を実行（最低1〜5ステップ）。
3. **通常生成: 簡潔推論 (`max_new_tokens=30`)**:
   - **想定ユースケース**: 分類結果に加えて簡単な理由やドメイン名を添えて返す運用（例: `A (IT Helpdesk: プリンタ接続障害)`）。
   - **設定値**: `max_new_tokens=30, do_sample=False`。
   - **処理内容**: 最大30ステップの自己回帰逐次デコードを実行。

### 高精度モデルで両者の遅延が一致する技術的理由（Early Stop現象）
- **高指示追従モデル（`Qwen2.5`, `Gemma 4`）**: 「Answer with only the single letter」という指示に忠実に従い、1〜2トークン目で選択肢文字を出力した直後に `<eos>`（End-of-Sequence）トークンを出力して自己回帰ループを**即座に早期終了（Early Stop）**します。そのため、`max_new_tokens=30` を指定しても実効的な生成ステップ数は 1〜2 回にとどまり、「最短1文字」と「簡潔推論」のレイテンシがほぼ同等になります。
- **1B未満モデル（`SmolLM2-360M`）**: 指示追従能力が弱く、`<eos>` を適切に出力できずに上限近くまで冗長な推論や繰り返し出力を継続します。その結果、デコード回数が増大して遅延が 133 ms から 283 ms へと急増し、LogitRouter 比で **10.15倍の高速化差** が生じます。

## Technical Analysis / 技術的要因の分析

**[English]**
- **Autoregressive Overhead**: `model.generate()` incurs multiple sequential kernel launches, KV cache memory allocations, and memory bandwidth contention for every additional token generated. Even when terminating after 1 token, the overhead of setting up generation configs, past_key_values cache structures, and full-vocabulary LM-Head projections adds substantial latency.
- **Single Forward Pass Advantage**: LogitRouter halts immediately after the prompt prefill phase, reducing the execution to exactly 1 Transformer forward pass plus a sliced linear projection of candidate tokens.

**[Japanese]**
- **自己回帰生成のオーバーヘッド**: 通常の `model.generate()` では、1トークン生成するごとに逐次カーネル起動、KVキャッシュ確保、およびメモリストリーミング（メモリバウンド制約）が発生します。仮に1トークンで終了した場合でも、generation_config の初期化、KVキャッシュバッファの確保、および全語彙（Qwen: 151,936語、Gemma: 262,144語）に対する巨大な射影・Softmax計算が必須となり、大きな固定オーバーヘッドとなります。
- **単一フォワードパスの優位性**: LogitRouter はプロンプトのPrefill完了直後に処理を完了し、全語彙射影を行わず選択肢トークン（数行）のみにスライスしてロジットを抽出するため、オーバーヘッドが極小化されます。