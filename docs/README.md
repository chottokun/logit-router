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
- `architecture/` - Deep dive into architecture concepts such as Single Forward Pass (Prefill) Routing and Sliced LM-Head optimization.
- `domain/` - Explanations of domain primitives, request/response models, and the fallback strategies using entropy and confidence.
- `infrastructure/` - Details regarding the FastAPI server, lifecycle management, and inference endpoints.
- `references/` - Benchmarks, latency decomposition via CUDA events, position bias, and OOD metrics.

**[Japanese]**
- `architecture/` - 単一フォワードパス（Prefill）ルーティングや Sliced LM-Head 最適化などのアーキテクチャ概念の詳細な解説。
- `domain/` - ドメインプリミティブ、リクエスト・レスポンスモデル、およびエントロピーと確信度を用いたフォールバック戦略の解説。
- `infrastructure/` - FastAPIサーバー、ライフサイクル管理、推論エンドポイントに関する実装詳細。
- `references/` - CUDAイベントによるレイテンシ分解、位置バイアス、OOD指標などのベンチマークとリファレンス。
