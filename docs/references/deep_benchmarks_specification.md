---
type: concept
title: Deep Evaluation Suite Specification / 深層評価スイート仕様書
description: Specification of the Logit Router Deep Evaluation Suite including major domains and metrics / Logit Routerの深層評価スイートの仕様（ドメインとメトリクスを含む）
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - references
  - benchmarks
  - metrics
  - evaluation
sources:
  - benchmarks/eval_suite.py
  - benchmarks/data/eval_cases.json
---

# Deep Evaluation Suite Specification / 深層評価スイート仕様書

**[English]**
This document specifies the design, scope, and evaluation metrics used in the Logit Router Deep Evaluation Suite. The suite is designed to rigorously test the accuracy, latency, and robustness of the routing engine across diverse, real-world scenarios.

**[Japanese]**
このドキュメントでは、Logit Router の深層評価スイートの設計、範囲、および評価指標（メトリクス）の仕様について説明します。このスイートは、多様な現実世界のシナリオ全体で、ルーティングエンジンの精度、レイテンシ、および堅牢性を厳密にテストするように設計されています。

## 10 Major Domains (100 Cases) Criteria / 10大ドメイン（100ケース）の網羅基準

**[English]**
The evaluation dataset (`eval_cases.json`) is carefully curated to cover 10 major domains, comprising a total of 100 test cases (10 cases per domain). This ensures the routing model is well-calibrated and performs consistently across a variety of instruction types.

The 10 major domains include:
1. **Security & Moderation (セキュリティとモデレーション)**: Identifying prompt injection, malicious intent, or inappropriate content.
2. **Language Detection (言語判定)**: Accurately routing based on the primary language of the input (e.g., English vs. Japanese).
3. **Urgency & Triage (緊急度判定)**: Classifying requests by severity or required response time (e.g., critical bug vs. feature request).
4. **Multilingual Reasoning (多言語推論)**: Routing tasks that require understanding instructions mixed across multiple languages.
5. **Coding & Syntax (コーディングと構文)**: Differentiating between programming languages or identifying syntax errors.
6. **Factual Query vs. Creative Writing (事実検索 対 創作)**: Determining if the user wants hard facts or a creative story.
7. **Sentiment Analysis (感情分析)**: Routing based on the emotional tone of the prompt (positive, negative, neutral).
8. **Domain-Specific Expert Routing (特定ドメインエキスパート)**: Routing to specialized models (e.g., legal, medical, or financial experts).
9. **Task Complexity (タスクの複雑度)**: Routing simple queries to smaller, faster models and complex reasoning tasks to larger models.
10. **Out-of-Distribution / Unknown (分布外・未知)**: Testing the system's ability to reject or safely route ambiguous, nonsensical, or highly unusual prompts.

**[Japanese]**
評価データセット（`eval_cases.json`）は、10の主要なドメインを網羅するように慎重に厳選されており、合計100のテストケース（各ドメイン10ケース）で構成されています。これにより、ルーティングモデルが適切にキャリブレーションされ、様々な種類の指示に対して一貫して機能することが保証されます。

10大ドメインは以下の通りです：
1. **セキュリティとモデレーション (Security & Moderation)**: プロンプトインジェクション、悪意のある意図、または不適切なコンテンツの特定。
2. **言語判定 (Language Detection)**: 入力の主要言語（例：英語 対 日本語）に基づく正確なルーティング。
3. **緊急度判定 (Urgency & Triage)**: 重大度や必要な応答時間（例：致命的なバグ 対 機能リクエスト）によるリクエストの分類。
4. **多言語推論 (Multilingual Reasoning)**: 複数の言語が混在する指示を理解する必要があるタスクのルーティング。
5. **コーディングと構文 (Coding & Syntax)**: プログラミング言語の識別や構文エラーの特定。
6. **事実検索 対 創作 (Factual Query vs. Creative Writing)**: ユーザーが厳密な事実を求めているのか、創造的な物語を求めているのかの判断。
7. **感情分析 (Sentiment Analysis)**: プロンプトの感情的なトーン（ポジティブ、ネガティブ、ニュートラル）に基づくルーティング。
8. **特定ドメインエキスパート (Domain-Specific Expert Routing)**: 専門化されたモデル（例：法律、医療、金融エキスパート）へのルーティング。
9. **タスクの複雑度 (Task Complexity)**: 単純なクエリをより小さく高速なモデルへ、複雑な推論タスクを大規模モデルへルーティング。
10. **分布外・未知 (Out-of-Distribution / Unknown)**: 曖昧、無意味、または非常に異常なプロンプトを拒絶または安全にルーティングするシステムの能力のテスト。

## Evaluation Metrics Definition / 評価メトリクス定義

**[English]**
The evaluation suite measures performance beyond simple accuracy, focusing heavily on latency and statistical confidence, which are critical for a low-latency router.

- **Latency Percentiles (p50/p95/p99)**: We measure the 50th (median), 95th, and 99th percentiles of routing latency to evaluate overhead even in worst-case scenarios. This ensures tail latencies do not block the pipeline.
- **Accuracy per Millisecond**: A composite metric evaluating the trade-off between speed and correctness. It is defined as `Overall Accuracy / Average Latency (ms)`. A higher value indicates an efficient routing mechanism.
- **Entropy Calibration**: Measures the model's self-awareness of uncertainty. When entropy is high, the margin between the top choices is narrow, indicating the router is "unsure". The suite validates that instances of high entropy correlate properly with complex or Out-of-Distribution cases, triggering the `FallbackRouter` correctly.

**[Japanese]**
評価スイートは単純な正解率にとどまらず、低遅延ルーターにとって重要なレイテンシと統計的確信度に重点を置いてパフォーマンスを測定します。

- **レイテンシのパーセンタイル (p50/p95/p99)**: ルーティングレイテンシの50パーセンタイル（中央値）、95パーセンタイル、および99パーセンタイルを測定し、テールケースを含めたオーバーヘッドを評価します。これにより、テールレイテンシがパイプラインをブロックしないことを確認します。
- **Accuracy per Millisecond (ミリ秒あたりの正解率)**: 速度と正確性のトレードオフを評価する複合指標。`全体正解率 / 平均レイテンシ (ms)` として定義されます。値が高いほど、効率的なルーティングメカニズムであることを示します。
- **Entropy Calibration (エントロピーキャリブレーション)**: モデル自身の不確実性に対する認識を測定します。エントロピーが高い場合、上位の選択肢間のマージンが狭く、ルーターが「迷っている」ことを示します。スイートは、高エントロピーのインスタンスが複雑なケースや分布外（OOD）のケースと適切に相関し、`FallbackRouter` を正しくトリガーすることを検証します。
