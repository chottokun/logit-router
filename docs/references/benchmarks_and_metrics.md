---
type: concept
title: Benchmarks and Metrics / ベンチマークと評価指標
description: Details of latency measurement, decomposition profiles, position bias, and OOD rejection / レイテンシ測定、分解プロファイル、位置バイアス、およびOOD拒絶評価手順の詳細
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - references
  - benchmarks
  - metrics
sources:
  - benchmarks/bench_latency.py
  - benchmarks/bench_profile.py
---

# Benchmarks and Metrics / ベンチマークと評価指標

**[English]**
Validating an ultra-low latency system requires specialized profiling techniques. Standard Python timers are insufficient for GPU-bound operations. This document outlines our rigorous benchmarking methodologies and the robustness metrics used to evaluate the Logit Router.

**[Japanese]**
超低レイテンシシステムを検証するには、特殊なプロファイリング手法が必要です。標準の Python タイマーは、GPU バウンドな操作には不十分です。このドキュメントでは、Logit Router を評価するために使用される厳密なベンチマーク手法と堅牢性指標について概説します。

## CUDA Event Precision Latency Measurement / CUDAイベントによる精密なレイテンシ計測

**[English]**
When using PyTorch with CUDA, operations are asynchronous; the CPU dispatches kernels to the GPU and continues execution without waiting for them to finish. Using Python's `time.perf_counter()` around PyTorch code only measures the kernel launch overhead, not the actual execution time. 
To achieve accurate profiling, we utilize `torch.cuda.Event(enable_timing=True)`. We record a `start_event`, run the inference, record an `end_event`, and then explicitly call `torch.cuda.synchronize()` to wait for the GPU to finish before calculating the elapsed time via `start_event.elapsed_time(end_event)`.

**[Japanese]**
CUDA を搭載した PyTorch を使用する場合、操作は非同期です。CPU はカーネルを GPU にディスパッチし、それらの終了を待たずに実行を継続します。PyTorch コードの周囲で Python の `time.perf_counter()` を使用すると、実際の実行時間ではなく、カーネルの起動オーバーヘッドのみが測定されます。
正確なプロファイリングを実現するために、`torch.cuda.Event(enable_timing=True)` を利用します。`start_event` を記録し、推論を実行し、`end_event` を記録してから、明示的に `torch.cuda.synchronize()` を呼び出して GPU の終了を待機し、`start_event.elapsed_time(end_event)` を介して経過時間を計算します。

## Factor Decomposition Profile / 要因分解プロファイル

**[English]**
By instrumenting different phases of the routing process (`bench_profile.py`), we can decompose the latency into distinct bottlenecks:
1. **Tokenization & Tensor Transfer**: CPU-based text tokenization and transferring `input_ids` to the GPU.
2. **Backbone Forward Pass**: The core prefill computation through the Transformer blocks.
3. **Sliced LM-Head MatMul**: The optimized projection over the select tokens.
4. **Softmax, Entropy & Post-proc**: Final mathematical operations and dictionary construction.

*Typical Profile Result:*
The Backbone Forward Pass overwhelmingly dominates the execution time (often >98%), while the Sliced LM-Head optimization reduces the classification overhead to a mere ~0.16%.

**[Japanese]**
ルーティングプロセスのさまざまなフェーズ（`bench_profile.py`）を計測することで、レイテンシを明確なボトルネックに分解できます：
1. **トークナイズとテンソル転送**: CPU ベースのテキストトークナイズと、`input_ids` の GPU への転送。
2. **バックボーン・フォワードパス**: Transformer ブロックを通るコアな Prefill 計算。
3. **Sliced LM-Head MatMul**: 選択されたトークンに対する最適化された射影。
4. **Softmax、エントロピー、および後処理**: 最終的な数学的演算と辞書の構築。

*典型的なプロファイル結果:*
バックボーン・フォワードパスが実行時間を圧倒的に支配し（多くの場合 98% 以上）、Sliced LM-Head の最適化により分類のオーバーヘッドはわずか約 0.16% に削減されます。

## Position Bias Tolerance / 位置バイアス耐性

**[English]**
Large Language Models exhibit a known "position bias" (or order effect), where the model may unfairly favor choices placed at the very beginning (Option A) or the very end of a list, regardless of semantic correctness. 
The Logit Router mitigates this via strict instruction framing ("Select the single best choice based strictly on the context.") and relies on the inherent capability of modern instruction-tuned models like Qwen2.5. Further robustness can be evaluated by shuffling choices and ensuring the model consistently selects the semantically correct option rather than a fixed position.

**[Japanese]**
大規模言語モデルには、既知の「位置バイアス（順序効果）」が存在します。これは、意味の正しさに関係なく、リストの最初（オプションA）または最後に配置された選択肢をモデルが不当に好む可能性がある現象です。
Logit Router は、厳密な指示のフレーミング（「コンテキストに厳密に基づいて最適な選択肢を1つ選択してください」）によってこれを軽減し、Qwen2.5 のような最新の Instruction-Tuned モデルの固有の機能に依存しています。さらなる堅牢性は、選択肢をシャッフルし、モデルが固定された位置ではなく、意味的に正しいオプションを一貫して選択することを確認することで評価できます。

## OOD (Out-of-Distribution) Rejection Evaluation / OOD (分布外) 拒絶評価手順

**[English]**
Handling queries that do not match any available choice (OOD inputs) is critical for routing. Instead of hallucinating a forced choice, the Logit Router relies on Shannon Entropy and confidence margins. 
An OOD input causes the probability distribution across A, B, and C to flatten out (e.g., [0.33, 0.34, 0.33]), resulting in a high entropy score. By setting an `entropy_threshold` (e.g., 0.9), the system can cleanly reject the input and hand it off to a fallback mechanism or return an error to the user, ensuring deterministic and safe behavior.

**[Japanese]**
利用可能な選択肢のいずれにも一致しないクエリ（OOD入力）を処理することは、ルーティングにとって非常に重要です。Logit Router は、強制的な選択をハルシネーション（幻覚）させるのではなく、シャノンエントロピーと確信度のマージンに依存します。
OOD 入力は、A、B、C 全体の確率分布を平坦化させ（例：[0.33, 0.34, 0.33]）、高いエントロピースコアをもたらします。`entropy_threshold`（例：0.9）を設定することで、システムは入力をきれいに拒絶し、フォールバックメカニズムに引き継ぐか、ユーザーにエラーを返すことができ、決定論的で安全な動作を保証します。
