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
The Logit Router leverages a highly optimized single-forward-pass prefill to extract probabilities for dynamic routing, deliberately bypassing the traditional autoregressive generation phase. This section outlines the structural foundations and primary optimizations that allow the system to achieve ultra-low latency while maintaining robust routing accuracy.

**[Japanese]**
Logit Router は、高度に最適化された単一フォワードパス（Prefill）を活用し、従来の自己回帰的な生成フェーズを意図的にバイパスすることで、動的ルーティングのための確率を抽出します。このセクションでは、システムが堅牢なルーティング精度を維持しながら超低レイテンシを実現するための構造的基盤と主要な最適化について概説します。

## Core Components / 主要コンポーネント

**[English]**
1. **Single Forward Pass Extraction**: Avoids sequential token generation by fetching the logits of the final prompt token.
2. **Sliced LM-Head**: Reduces matrix multiplication cost by slicing the LM-Head to compute only the necessary option tokens.
3. **A/B/C Index Mapping**: Simplifies semantic outputs into deterministic character classes for fast inference.

For an in-depth understanding, refer to [Single Forward Pass Routing](single_forward_pass_routing.md).

**[Japanese]**
1. **単一フォワードパス抽出**: プロンプトの最後のトークンのロジットを取得することで、シーケンシャルなトークン生成を回避します。
2. **Sliced LM-Head**: 必要な選択肢トークンのみを計算するために LM-Head をスライスし、行列積の計算コストを大幅に削減します。
3. **A/B/Cインデックスマッピング**: 複雑な意味的出力（カテゴリ名など）を決定論的な単一文字クラス（A, B, C）にマッピングすることで推論を高速化します。

詳細については、[Single Forward Pass Routing](single_forward_pass_routing.md) を参照してください。
