---
type: concept
title: Benchmarks and Metrics / ベンチマークと評価指標
description: Methodologies for GPU latency profiling, layer decomposition, position bias, and OOD detection / GPUレイテンシ計測、要因分解プロファイリング、位置バイアス検証、およびOOD検出手法
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T15:00:00Z"
tags:
  - references
  - benchmarks
  - metrics
sources:
  - benchmarks/bench_latency.py
  - benchmarks/bench_profile.py
  - benchmarks/test_robustness.py
---

# Benchmarks and Metrics / ベンチマークと評価指標

**[English]**
Evaluating a low-latency routing architecture requires measuring GPU execution time while accounting for asynchronous kernel launches. This document details the profiling methodologies and validation metrics used to assess inference latency, layer decomposition, position bias, and out-of-distribution (OOD) rejection.

**[Japanese]**
低遅延ルーティングアーキテクチャの評価では、非同期カーネル実行を考慮したGPU実行時間の正確な計測が求められます。本ドキュメントでは、推論遅延のプロファイリング手法、各処理層の要因分解、選択肢順序に対する位置バイアス検証、および分布外（OOD）入力の検出評価指標について記述します。

## Latency Measurement Methodology / レイテンシ計測手法

**[English]**
PyTorch operations targeting CUDA devices are executed asynchronously; host code dispatches kernels to the stream and returns immediately. Relying on host-side wall clocks such as `time.perf_counter()` captures only kernel launch overhead rather than compute completion.
Accurate profiling is performed using `torch.cuda.Event(enable_timing=True)`:
1. Warmup runs are executed to initialize GPU memory allocations and JIT compile any dynamic elements.
2. A `start_event` is placed onto the active stream.
3. The forward pass is invoked.
4. An `end_event` is placed on the stream, followed by explicit stream synchronization (`torch.cuda.synchronize()`).
5. Execution duration is retrieved via `start_event.elapsed_time(end_event)`.

**[Japanese]**
CUDAデバイスを対象とするPyTorchの処理は非同期に実行され、ホスト側のコードはカーネルをストリームに投入すると即座に制御を戻します。そのため、ホスト側の壁時計（`time.perf_counter()` 等）による計測では、カーネルの投入オーバーヘッドのみが反映され、計算完了までの実時間は計測されません。
正確な時間計測のため、`torch.cuda.Event(enable_timing=True)` を使用します：
1. メモリ確保や動的要素の初期化を完了させるため、ウォームアップ反復を実行します。
2. アクティブなストリームに `start_event` を配置します。
3. フォワードパスを実行します。
4. ストリームに `end_event` を配置し、明示的に `torch.cuda.synchronize()` を呼び出してGPU側の処理完了を同期します。
5. `start_event.elapsed_time(end_event)` を通じて経過時間を取得します。

## Factor Decomposition Profile / 要因分解プロファイリング

**[English]**
The benchmark suite (`bench_profile.py`) divides inference execution into four isolated stages:
1. **Tokenization and Host-to-Device Copy**: Encoding prompt text on CPU and moving input tensors to device memory.
2. **Backbone Forward Pass**: Transformer block prefill computation generating hidden state representations.
3. **Sliced LM-Head Projection**: Linear projection applied exclusively to candidate token indices.
4. **Post-Processing & Metrics**: Softmax computation, Shannon entropy calculation, and Python dictionary assembly.

*Measured Baseline (NVIDIA RTX 3060, Qwen2.5-0.5B-Instruct, Sequence Length ~135):*
- Backbone Forward Pass: $27.97\,\text{ms}$ ($97.56\%$)
- Sliced LM-Head: $0.05\,\text{ms}$ ($0.16\%$)
- Tokenization & Transfer: $0.46\,\text{ms}$ ($1.62\%$)
- Post-processing: $0.19\,\text{ms}$ ($0.66\%$)

The data confirms that the final classification layer represents a negligible fraction of overall execution time.

**[Japanese]**
プロファイル測定スクリプト（`bench_profile.py`）では、推論処理を以下の4段階に分離して計測します：
1. **トークナイズおよびホスト-デバイス間転送**: CPUでのテキストトークナイズと、入力テンソルのGPUメモリへの転送。
2. **バックボーン・フォワードパス**: TransformerブロックによるPrefill計算（隠れ状態の生成）。
3. **Sliced LM-Head 射影**: 選択肢インデックスに限定した線形層の射影計算。
4. **後処理および指標計算**: ソフトマックス計算、シャノンエントロピー算出、結果オブジェクトの生成。

*実機計測例 (NVIDIA RTX 3060, Qwen2.5-0.5B-Instruct, 系列長 約135トークン):*
- バックボーン・フォワードパス: $27.97\,\text{ms}$ ($97.56\%$)
- Sliced LM-Head: $0.05\,\text{ms}$ ($0.16\%$)
- トークナイズおよび転送: $0.46\,\text{ms}$ ($1.62\%$)
- 後処理: $0.19\,\text{ms}$ ($0.66\%$)

実測データより、最終分類層の演算オーバーヘッドは全体の極小部分にとどまることが確認されます。

## Position Bias Evaluation / 位置バイアス評価

**[English]**
Instruction-tuned language models can exhibit sensitivity to candidate order, potentially showing preferential selection toward earlier options (`A`) regardless of context.
To measure position bias, `test_robustness.py` generates all permutations of candidate lists for evaluation cases and calculates selection consistency:
$$\text{Consistency} = \frac{\text{Number of permutations where the semantic winner remains identical}}{\text{Total permutations}}$$
A consistency rate near $100\%$ indicates that candidate order does not shift the argmax decision.

**[Japanese]**
指示調整済み言語モデルは、選択肢の提示順序に対して偏り（先頭の選択肢 `A` が選択されやすい等）を示す場合があります。
位置バイアスを定量化するため、`test_robustness.py` では候補リストの順列（Permutation）を全パターン生成し、選択の一貫性（Consistency）を測定します：
$$\text{Consistency} = \frac{\text{意味的に同一の選択肢が最上位に選ばれた順列数}}{\text{総順列数}}$$
一貫性が $100\%$ に近い場合、提示順序の変化が判定結果に影響を与えていないことを示します。

## Out-of-Distribution (OOD) Rejection Evaluation / 分布外 (OOD) 入力検出評価

**[English]**
Queries that do not fit any of the provided categories should be flagged for rejection rather than matched to an incorrect choice.
The router uses Shannon entropy $H$ as an uncertainty signal. On in-domain test cases, the probability distribution is concentrated (low $H$, mean $\approx 0.014$). On deliberately mismatched out-of-distribution queries, probability mass spreads across candidates, resulting in elevated entropy (mean $\approx 0.834$). Applying an entropy cutoff threshold (typically $0.5 \le H \le 0.6$) separates clear classifications from ambiguous cases.

**[Japanese]**
提示されたどの候補カテゴリにも合致しない入力クエリは、誤った候補へ割り振られる前に検出・除外される必要があります。
本ルーターでは、不確実性の指標としてシャノンエントロピー $H$ を用います。ドメイン内の正当な入力では特定候補に確率が集中するため低エントロピー（平均値 $\approx 0.014$）となりますが、意図的に無関係なクエリを与えた場合は確率が分散し、エントロピーが上昇します（平均値 $\approx 0.834$）。適切なエントロピー閾値（通常 $0.5 \le H \le 0.6$）を設定することで、明確な分類と曖昧な入力を分離できます。
