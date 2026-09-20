---
type: concept
title: Routing Primitives / ルーティング・プリミティブ
description: Specification of routing data structures, uncertainty metrics, and fallback criteria / ルーティングのデータ構造、不確実性評価指標、およびフォールバック基準の技術仕様
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T15:00:00Z"
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
The routing subsystem relies on standardized data schemas to represent incoming requests and outgoing decisions. To support post-routing validation, outputs include both candidate probability distributions and calculated entropy values.

**[Japanese]**
ルーティングサブシステムは、入力リクエストおよび判定結果を標準化されたデータスキーマで扱います。ルーティング後の検証や分岐処理を可能にするため、出力には各候補の確率分布に加え、計算された情報エントロピーが含まれます。

## Data Schemas: RouteRequest and RouteResult / データスキーマ: RouteRequest と RouteResult

**[English]**
The `RouteRequest` object encapsulates input arguments:
- `context`: Background text or document snippet providing grounding for the routing decision.
- `instruction`: Task description or user query to be categorized.
- `choices`: List of candidate target categories (strings).
- `temperature`: Scaling factor applied to logits prior to softmax normalization (default: `1.0`).

The `RouteResult` object contains the classification outputs:
- `best_choice`: Candidate string associated with the highest logit.
- `best_letter`: Assigned letter index (`A`, `B`, `C`, etc.).
- `confidence`: Softmax probability $P(\text{best\_choice})$ among the evaluated choices.
- `entropy`: Shannon entropy computed over the candidate probability distribution.
- `distribution`: Mapping from candidate letter to normalized probability.

**[Japanese]**
`RouteRequest` オブジェクトは以下の入力引数を保持します：
- `context`: ルーティング判断の根拠となる背景情報または参照テキスト。
- `instruction`: 分類対象となるタスク指示またはユーザーの入力クエリ。
- `choices`: 分類先候補となる文字列のリスト。
- `temperature`: ソフトマックス正規化前にロジットをスケーリングする温度パラメータ（デフォルト値: `1.0`）。

`RouteResult` オブジェクトは分類結果を保持します：
- `best_choice`: 最も高いロジット値を得た候補文字列。
- `best_letter`: 該当するインデックス文字（`A`, `B`, `C` など）。
- `confidence`: 評価対象選択肢内における最上位候補のソフトマックス確率 $P(\text{best\_choice})$。
- `entropy`: 候補確率分布から算出されたシャノンエントロピー。
- `distribution`: 各候補文字と正規化された確率値の対応辞書。

## Uncertainty Metrics: Confidence and Entropy / 不確実性の評価: 確信度とエントロピー

**[English]**
- **Confidence ($P_{\max}$)**: Defined as $\max_i P(i)$ where $P(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$ over candidate logits $z$. High values indicate that the model exhibits a clear preference for a specific candidate.
- **Shannon Entropy ($H$)**: Computed over the $K$ normalized candidate probabilities:
  $$H = -\sum_{i=1}^{K} P(i) \ln P(i)$$
  When probabilities are uniformly distributed across candidates ($P(i) \approx 1/K$), $H$ approaches its maximum $\ln(K)$. Conversely, when one candidate dominates, $H \approx 0$.

**[Japanese]**
- **確信度 ($P_{\max}$)**: 候補ロジット $z$ に対するソフトマックス確率の最大値 $\max_i P(i)$（ここで $P(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$）として定義されます。値が高いほど、モデルの出力確率が特定の候補に集中していることを示します。
- **シャノンエントロピー ($H$)**: $K$ 個の正規化確率に基づいて算出されます：
  $$H = -\sum_{i=1}^{K} P(i) \ln P(i)$$
  各候補の確率が均等（$P(i) \approx 1/K$）に近い場合、$H$ は最大値 $\ln(K)$ に漸近します。単一の候補に確率が集中している場合は $H \approx 0$ となります。

## Fallback Routing / フォールバックルーティング

**[English]**
The schema provides utility methods `is_confident(threshold=0.7)` and `needs_fallback(entropy_threshold=0.9)` to support automated pipeline routing:
- When entropy exceeds a defined threshold or confidence falls below target, requests can be routed to an auxiliary handler.
- Auxiliary strategies include multi-step autoregressive generation (e.g., Chain-of-Thought) or dispatching to larger model endpoints when latency constraints permit.

**[Japanese]**
パイプライン処理との連携のため、スキーマには `is_confident(threshold=0.7)` や `needs_fallback(entropy_threshold=0.9)` などの判定メソッドが用意されています：
- エントロピーが閾値を超過した場合や確信度が基準値を下回った場合、後続の補助ハンドラへリクエストを分岐させることができます。
- 補助ハンドラとしては、自己回帰による多段生成（Chain-of-Thought等）の実行や、許容レイテンシに応じてより大規模なモデルへのフォールバックディスパッチが選択されます。
