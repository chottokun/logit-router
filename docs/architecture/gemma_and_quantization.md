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
To allow deployment on hardware with limited VRAM, Logit Router supports 4-bit and 8-bit weight loading.
In routing prefill, relative margins between choice logits determine classification outcomes. Minor numeric quantization noise typically does not alter the argmax ranking, maintaining high classification consistency while reducing accelerator memory consumption by up to 75%.

**[Japanese]**
VRAM 制約のあるハードウェアでのデプロイを可能にするため、Logit Router は 4-bit および 8-bit の量子化ロードに対応しています。
ルーティング判定においては選択肢ロジット間の相対的な大小関係（マージン）が重要となるため、微小な量子化誤差が argmax の選択結果を反転させるリスクは限定的であり、メモリ使用量を最大約 75% 削減しつつ判定の一貫性を維持できます。

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
