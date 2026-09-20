---
type: concept
title: Architecture Overview / アーキテクチャ概要
description: Overview of the Logit Router Architecture / Logit Routerアーキテクチャの概要
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - architecture
  - overview
sources:
  - src/logit_router/router.py
---

# Architecture Overview / アーキテクチャ概要

**[English]**
The Logit Router leverages a single-forward-pass prefill to extract probabilities for dynamic routing, avoiding autoregressive generation loops. This section outlines the architecture and optimizations designed to minimize inference latency while maintaining classification accuracy.

**[Japanese]**
Logit Router は、プロンプト末尾の単一フォワードパス（Prefillフェーズ）から選択肢確率を抽出し、自己回帰ループをバイパスしてルーティングを行います。本セクションでは、分類精度を維持しながら推論遅延を最小化するための設計および最適化手法について概説します。

## Core Components / 主要コンポーネント

**[English]**
1. **Single Forward Pass Extraction**: Avoids sequential token generation by extracting logits exclusively from the final prompt token.
2. **Sliced LM-Head**: Slices the language model projection layer to calculate logits only for active candidate tokens.
3. **Index Mapping**: Maps arbitrary candidate descriptions to single-character indices (`A`, `B`, `C`, ...) to ensure deterministic single-token classification.

For details, refer to:
- [Single Forward Pass Routing](single_forward_pass_routing.md)
- [Gemma Architecture & Quantization](gemma_and_quantization.md)
- [Cascade Routing & Critical Analysis](cascade_routing_and_critical_analysis.md)

**[Japanese]**
1. **単一フォワードパス抽出**: プロンプト末尾トークンのロジットのみを抽出することで、逐次トークン生成を省略します。
2. **Sliced LM-Head**: 言語モデルの最終射影層をスライスし、評価対象の選択肢トークンのみを行列積計算の対象とします。
3. **インデックスマッピング**: 任意の候補ラベルを単一文字（`A`, `B`, `C`, ...）へ対応付け、単一トークンによる確定的分類を実現します。

技術仕様の詳細は以下を参照してください：
- [Single Forward Pass Routing (単一フォワードパスルーティング)](single_forward_pass_routing.md)
- [Gemma Architecture & Quantization (Gemma アーキテクチャと量子化技術)](gemma_and_quantization.md)
- [Cascade Routing & Critical Analysis (カスケードルーティングと批判的分析)](cascade_routing_and_critical_analysis.md)
