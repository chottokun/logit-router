---
type: concept
title: Routing Primitives / ルーティング・プリミティブ
description: Description of routing primitives like RouteRequest, RouteResult, and FallbackRouter / RouteRequest、RouteResult、FallbackRouterなどのルーティングプリミティブの説明
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - domain
  - primitives
  - fallback
sources:
  - src/logit_router/schema.py
  - src/logit_router/router.py
---

# Routing Primitives / ルーティング・プリミティブ

**[English]**
The routing primitives form the foundation of how the Logit Router understands inputs and articulates its decisions. By deeply integrating confidence and entropy into the result schema, the system provides a robust mechanism for handling uncertain or out-of-distribution inputs.

**[Japanese]**
ルーティング・プリミティブは、Logit Router が入力をどのように理解し、決定をどのように表現するかの基盤を形成します。確信度とエントロピーを結果スキーマに深く統合することで、システムは不確実な入力や分布外（OOD）の入力を処理するための堅牢なメカニズムを提供します。

## RouteRequest and RouteResult / RouteRequest と RouteResult

**[English]**
The `RouteRequest` acts as the primary input schema. It consists of:
- `context`: The foundational information or document the router must base its decision on.
- `instruction`: The specific task or query.
- `choices`: An arbitrary list of strings representing the possible categories or actions.
- `temperature`: Modifies the logits before the softmax operation (default is 1.0).

The `RouteResult` is the output schema. It returns:
- `best_choice`: The selected string from the choices.
- `best_letter`: The mapped token (e.g., A, B, C).
- `confidence`: The probability $P$ of the top choice.
- `entropy`: A calculated metric representing uncertainty.
- `distribution`: A dictionary mapping each letter to its probability.

**[Japanese]**
`RouteRequest` は主要な入力スキーマとして機能します。以下で構成されます：
- `context`: ルーターが判断の基準としなければならない基礎情報やドキュメント。
- `instruction`: 具体的なタスクやクエリ。
- `choices`: 可能なカテゴリやアクションを表す任意の文字列リスト。
- `temperature`: ソフトマックス操作の前にロジットを変更するパラメータ（デフォルトは 1.0）。

`RouteResult` は出力スキーマです。以下を返します：
- `best_choice`: 選択肢から選ばれた文字列。
- `best_letter`: マッピングされたトークン（例：A、B、C）。
- `confidence`: 最上位の選択肢の確率 $P$。
- `entropy`: 不確実性を表す計算された指標。
- `distribution`: 各文字をその確率にマッピングする辞書。

## Entropy and Confidence / エントロピーと確信度

**[English]**
- **Confidence**: This is simply the highest probability from the Softmax function over the subset of choice logits. A high confidence means the model strongly prefers one option over the others.
- **Entropy ($H$)**: We compute the Shannon Entropy over the selected probabilities:
  $H = -\sum_{i=1}^{K} P(i) \log_e P(i)$
  High entropy indicates the probability mass is distributed evenly across multiple choices. This usually means the model is unsure or the prompt is ambiguous.

**[Japanese]**
- **確信度 (Confidence)**: これは単に、選択肢のロジットのサブセットに対するソフトマックス関数からの最も高い確率です。確信度が高いということは、モデルが他の選択肢よりも一つの選択肢を強く好んでいることを意味します。
- **エントロピー ($H$)**: 選択された確率に対してシャノンエントロピーを計算します：
  $H = -\sum_{i=1}^{K} P(i) \log_e P(i)$
  エントロピーが高い場合、確率の質量が複数の選択肢に均等に分散していることを示します。これは通常、モデルが確信を持てないか、プロンプトが曖昧であることを意味します。

## Fallback Logic / フォールバックロジック

**[English]**
The `RouteResult` schema includes helper methods like `is_confident(threshold=0.7)` and `needs_fallback(entropy_threshold=0.9)`. When entropy is above a certain threshold, or the margin between the top two choices is small, a fallback system can be triggered. A Fallback Router might use full autoregressive generation (Chain-of-Thought) or route the request to a larger, more capable model (e.g., GPT-4 or Claude 3.5 Sonnet) at the cost of higher latency.

**[Japanese]**
`RouteResult` スキーマには、`is_confident(threshold=0.7)` や `needs_fallback(entropy_threshold=0.9)` などのヘルパーメソッドが含まれています。エントロピーが特定のしきい値を超えている場合、または上位2つの選択肢の差（マージン）が小さい場合、フォールバックシステムをトリガーできます。フォールバックルーターは、完全な自己回帰生成（Chain-of-Thought）を使用したり、よりレイテンシが高くなることを代償として、より大規模で高性能なモデル（GPT-4 や Claude 3.5 Sonnet など）にリクエストをルーティングしたりすることができます。
