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
Gemma models feature an exceptionally large vocabulary of 256,000 to 262,144 tokens (`vocab_size = 262,144` in Gemma 4). In standard text generation, projecting the hidden state $h \in \mathbb{R}^d$ across the entire vocabulary requires $O(d \cdot 262,144)$ FLOPs and heavy memory streaming.

Logit Router's **Sliced LM-Head** resolves this bottleneck. Because choices are mapped to deterministic token indices $\mathcal{C} = \{c_1, \dots, c_K\}$, only the corresponding rows $W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$ are evaluated.
- For $K=3$, the effective computation is compressed by a factor of $3 / 262,144 \approx 0.0011\%$.
- The larger the model's vocabulary, the greater the relative efficiency gains of the Sliced LM-Head compared to full projection.

**[Japanese]**
Gemma シリーズは、256,000〜262,144 トークンという極めて大きな語彙サイズを持ちます（Gemma 4 では `vocab_size = 262,144`）。標準的なテキスト生成では、隠れ状態 $h \in \mathbb{R}^d$ を全語彙へ射影するために $O(d \cdot 262,144)$ の演算と大量の重みメモリストリーミングが発生します。

Logit Router の **Sliced LM-Head** はこのボトルネックを解消します。選択肢が確定的なトークンインデックス $\mathcal{C} = \{c_1, \dots, c_K\}$ にマッピングされるため、該当する行 $W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$ のみを評価します。
- 選択肢数 $K=3$ の場合、計算量は全射影の約 $0.0011\%$ に圧縮されます。
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
