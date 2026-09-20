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
Welcome to the Logit Router documentation. This documentation is built according to the Open Knowledge Format (OKF) v0.2 structure. It provides comprehensive, highly-detailed explanations of the architecture, domain primitives, infrastructure, and benchmarks underlying the ultra-low latency LLM-based routing engine. 

**[Japanese]**
Logit Router のドキュメントへようこそ。このドキュメントは Open Knowledge Format (OKF) v0.2 の構造に従って構築されています。超低遅延のLLMベースのルーティングエンジンの基礎となるアーキテクチャ、ドメインプリミティブ、インフラストラクチャ、およびベンチマークに関する包括的かつ非常に詳細な説明を提供します。

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
