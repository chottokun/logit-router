---
type: index
title: Logit Router Documentation
description: Overview of the Logit Router documentation built on OKF v0.2 / Logit Routerドキュメントの概要
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - index
  - overview
sources: []
---

# Logit Router Documentation / Logit Router ドキュメント

**[English]**
This documentation describes the Logit Router system conforming to the Open Knowledge Format (OKF) v0.2. It covers the system architecture, domain primitives, serving infrastructure, and empirical benchmark evaluations for logit-based routing.

**[Japanese]**
本ドキュメントは、Open Knowledge Format (OKF) v0.2 に準拠した Logit Router の仕様書です。ロジット抽出を用いたルーティングエンジンのアーキテクチャ、ドメインモデル、インフラ構成、および実測ベンチマーク評価手法を記述します。

## Directory Structure / ディレクトリ構成

**[English]**
- `architecture/` - Deep dive into architecture concepts such as Single Forward Pass Routing, Sliced LM-Head optimization, Gemma 4 token compression, and the **Two-Tier Cascade Routing Architecture**.
  - Highlight: [Cascade Routing & Critical Analysis](architecture/cascade_routing_and_critical_analysis.md)
- `domain/` - Explanations of domain primitives, request/response models, and the fallback strategies using entropy and confidence.
- `infrastructure/` - Details regarding the FastAPI server, lifecycle management, and inference endpoints.
- `references/` - Empirical on-device benchmarks, autoregressive comparison (up to 10x speedup), 100-case deep evaluation across 10 domains, position bias, and OOD metrics.
  - Highlight: [Deep 10-Domain Benchmark Report](references/deep_eval_report.md)
  - Highlight: [Comparison vs Generation Report](references/comparison_vs_generation_report.md)

**[Japanese]**
- `architecture/` - 単一フォワードパスルーティング、Sliced LM-Head 最適化、Gemma 4 の語彙圧縮効果、および **2段階カスケードルーティング設計** などのアーキテクチャ詳細。
  - 主要ドキュメント: [Cascade Routing & Critical Analysis (カスケードルーティングと批判的分析)](architecture/cascade_routing_and_critical_analysis.md)
- `domain/` - ドメインプリミティブ、リクエスト・レスポンスモデル、およびエントロピーと確信度を用いたフォールバック戦略の解説。
- `infrastructure/` - FastAPIサーバー、ライフサイクル管理、推論エンドポイントに関する実装詳細。
- `references/` - 実機ベンチマーク実測値、通常生成との直接対決（最大10倍・Gemma4で5.78倍高速化）、10大ドメイン100問深層評価、位置バイアス検証などのリファレンス。
  - 主要ドキュメント: [Deep 10-Domain Benchmark Report (10大ドメイン100問深層実機評価レポート)](references/deep_eval_report.md)
  - 主要ドキュメント: [Comparison vs Generation Report (通常生成との速度・レイテンシ比較レポート)](references/comparison_vs_generation_report.md)
