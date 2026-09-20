---
type: concept
title: Gemma Architecture & Quantization / Gemma アーキテクチャと量子化技術
description: Architectural specifications of Gemma 2 / Gemma 4 integration, vocabulary scaling, and INT4 quantization / Gemma 2 / Gemma 4 の統合仕様、語彙スケーリング、および INT4 量子化の解説
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T15:15:00Z"
tags:
  - architecture
  - gemma
  - quantization
  - awq
  - 4-bit
sources:
  - src/logit_router/router.py
---

# Gemma Architecture & Quantization / Gemma アーキテクチャと量子化技術

**[English]**
This document describes the architectural characteristics of integrating the Gemma series (Gemma 2 and Gemma 4) and 4-bit/8-bit quantization into Logit Router. It focuses on how Logit Router eliminates the latency bottleneck caused by large vocabulary projections and how model compression affects routing accuracy.

**[Japanese]**
本ドキュメントでは、Logit Router における Gemma 系列（Gemma 2 および Gemma 4）の統合アーキテクチャ、ならびに 4-bit / 8-bit 量子化技術の適用仕様について記述します。巨大な語彙サイズに伴う射影遅延の排除メカニズム、およびモデル圧縮がルーティング精度に与える影響について解説します。

## Vocabulary Scaling: 262K Vocab and Sliced LM-Head / 語彙スケーリング: 26.2万語と Sliced LM-Head

**[English]**
Gemma models feature an exceptionally large vocabulary of 256,128 tokens (Gemma 1/2/3) to 262,144 tokens (`vocab_size = 262,144` in Gemma 4). In standard text generation, projecting the hidden state $h \in \mathbb{R}^d$ across the entire vocabulary requires $O(d \cdot 262,144)$ FLOPs and heavy memory streaming.
Because routing only classifies among $K$ predefined categories (typically $K \le 5$), the router pre-slices the LM-head weights to retain only rows corresponding to target choice tokens ($W_{\text{sliced}} = W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$).
This cuts the linear projection workload by more than 99.998% (reducing it to approximately 0.0011% of the full projection cost), entirely removing the LM-head bottleneck.

**[Japanese]**
Gemma シリーズは、256,128 トークン（Gemma 1/2/3）〜 262,144 トークン（Gemma 4）という極めて大きな語彙サイズを持ちます。標準的なテキスト生成では、隠れ状態 $h \in \mathbb{R}^d$ を全語彙へ射影するために $O(d \cdot 262,144)$ の演算と大量の重みメモリストリーミングが発生します。
本ルーティングでは $K$ 個の事前定義カテゴリ（通常 $K \le 5$）のみを判別するため、ルーターは選択肢トークンに対応する行のみを事前にスライスした重み行列（$W_{\text{sliced}} = W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$）を保持します。
これにより、線形射影層の計算量は 99.998% 以上削減（全語彙射影の約 0.0011% に圧縮）され、LM-Head のボトルネックが完全に解消されます。
- モデルの語彙数が大きければ大きいほど、全射影に対する Sliced LM-Head の計算量削減比率は高まります。

## Japanese Tokenization Characteristics / 日本語トークナイズの特性

**[English]**
With a 262K vocabulary, Gemma 4 exhibits superior token compression on Japanese text. Common Japanese compound terms (such as 「クレジットカード」) are represented as single individual tokens rather than being split into multiple sub-words.
- Shorter token sequence lengths directly reduce Transformer prefill compute time.
- Single-letter candidate tokens (`'A'`, `'B'`) tokenize without requiring leading whitespace prefixes, as handled by the token extraction logic in `router.py`.

**[Japanese]**
26.2万語の語彙サイズを持つ Gemma 4 は、日本語テキストに対して高いトークン圧縮率を示します。頻出する複合語（例: 「クレジットカード」）がサブワードに細分化されず、単一トークンとしてトークナイズされます。
- 入力トークン系列長が短縮されることで、Transformer の Prefill 計算時間が直接的に低減されます。
- 選択肢文字（`'A'`, `'B'`）は先頭空白を必要とせず単独でエンコードされるため、`router.py` のトークン抽出ロジックで適切に分岐処理されます。

## Quantization (AWQ & 4-bit) / 量子化技術 (AWQ & 4-bit)

**[English]**
To allow deployment on hardware with limited VRAM, Logit Router supports 4-bit and 8-bit weight loading via `bitsandbytes` as well as dedicated AWQ (Activation-aware Weight Quantization) models using Marlin kernels.
In routing prefill, relative margins between choice logits determine classification outcomes. Minor numeric quantization noise typically does not alter the argmax ranking, maintaining high classification consistency while reducing accelerator memory consumption significantly.

*Empirical 100-Case Evaluation (NVIDIA RTX 3060 12GB):*
- **`google/gemma-4-E2B-it (4-bit bitsandbytes)`**:
  - **Accuracy**: Retains **91.0%** (91/100 correct), suffering only a minimal 4% drop compared to the 95.0% BF16 baseline.
  - **Peak VRAM**: Drops from 9.76 GB to **6.49 GB** (-3.12 GB, a **32.0% reduction**), easily fitting within commodity 8GB-12GB GPUs.
  - **Latency Trade-off**: p50 latency increases to **171.89 ms** due to dynamic on-the-fly dequantization in bitsandbytes.
- **`Qwen2.5-1.5B-Instruct-AWQ (Marlin 4-bit)`**:
  - **Accuracy**: Reaches **84.0%** (84/100 correct), matching or exceeding the 81.0% BF16 baseline.
  - **Latency**: Achieves a fast **31.41 ms** p50 latency thanks to optimized Marlin FP16 GEMM kernels (faster than 34.84 ms in BF16).
  - **Peak VRAM**: Remains low at **2.96 GB**.

**[Japanese]**
VRAM 制約のあるハードウェアでのデプロイを可能にするため、Logit Router は `bitsandbytes` による 4-bit / 8-bit ロード、および Marlin 最適化カーネルを用いた AWQ（Activation-aware Weight Quantization）専用モデルに対応しています。
ルーティング判定においては選択肢ロジット間の相対的な大小関係（マージン）が重要となるため、微小な量子化誤差が argmax の選択結果を反転させるリスクは限定的であり、高水準の判定一貫性を維持しながらメモリ使用量を大幅に削減できます。

*10大ドメイン100問実務データセットでの実機実測値（NVIDIA RTX 3060 12GB）:*
- **`google/gemma-4-E2B-it` (4-bit bitsandbytes)**:
  - **正解率**: BF16 ベースライン（95.0%）からわずか 4% 減の **91.0%**（91/100問）を維持。4-bit 化による文脈解釈能力の劣化は極めて軽微です。
  - **ピーク VRAM**: 9.76 GB から **6.49 GB**（約 3.12 GB 削減、**32.0% 削減**）へ大幅圧縮され、8GB〜12GB クラスのコンシューマ向け GPU で安全に常駐可能です。
  - **遅延のトレードオフ**: bitsandbytes の動的逆量子化オーバーヘッドにより、p50 遅延は **171.89 ms** となります（低 VRAM 環境と速度のトレードオフ）。
- **`Qwen2.5-1.5B-Instruct-AWQ` (AWQ Marlin 4-bit)**:
  - **正解率**: **84.0%**（84/100問）を達成し、BF16 ベースライン（81.0%）と同等以上の結果を記録。
  - **レイテンシ**: 最適化された Marlin FP16 GEMM カーネルの効果により、p50 遅延は **31.41 ms**（BF16 の 34.84 ms より高速）をマーク。
  - **ピーク VRAM**: **2.96 GB** に抑制され、低遅延エッジゲートとして理想的な挙動を示します。

## Attention Architecture and Soft-Capping (Gemma 2 vs Gemma 4) / アテンション機構とソフトキャッピングの差異

**[English]**
A crucial architectural distinction exists between Gemma 2 and Gemma 4 regarding attention computation:
1. **Gemma 2 Logit Soft-Capping**:
   Gemma 2 enforces logit soft-capping at both the attention query-key product level ($50.0$) and the final output logits ($30.0$):
   $$\text{Attention Logits} = \text{cap} \times \tanh\left(\frac{Q K^T}{\sqrt{d} \times \text{cap}}\right)$$
   Because standard vanilla FlashAttention-2 kernels do not natively implement this attention-level $\tanh$ capping, running Gemma 2 with incompatible FlashAttention-2 backends can lead to degraded scores or numerical instability. Hence, PyTorch SDPA or `eager` mode is strictly recommended.
2. **Gemma 4 Streamlined Architecture**:
   In Gemma 4, attention-level soft-capping is removed, retaining soft-capping only at the final LM-Head projection. Consequently, Gemma 4 natively supports FlashAttention-2 and SDPA without custom kernel modifications, enabling optimal throughput and latency during prompt prefill.

**[Japanese]**
Gemma 2 と Gemma 4 の間には、アテンション演算における重要なアーキテクチャ上の差異が存在します：
1. **Gemma 2 の二重ソフトキャッピング (Logit Soft-Capping)**:
   Gemma 2 はアテンション計算の $QK^T$ ロジット（上限 $50.0$）と最終出力ロジット（上限 $30.0$）の双方に $\tanh$ によるソフトキャッピングを強制します：
   $$\text{Attention Logits} = 50.0 \times \tanh\left(\frac{Q K^T}{\sqrt{d} \times 50.0}\right)$$
   一般的な標準 FlashAttention-2 カーネルはこのアテンション内ソフトキャッピングに対応していないため、非対応環境で無理に FlashAttention を適用すると出力の劣化や数値不安定（NaN）を招くリスクがあります。そのため、Gemma 2 では PyTorch SDPA または `eager` バックエンドの適用が必須となります。
2. **Gemma 4 のアーキテクチャ合理化**:
   Gemma 4 ではアテンション層でのソフトキャッピングが廃止され、最終の LM-Head 出力層のみに集約されました。これにより、Gemma 4 は標準の `flash_attention_2` および PyTorch `sdpa` の高速カーネルを一切の精度劣化なくネイティブに活用することが可能となり、Prefill 計算の高速化を最大限に享受できます。
