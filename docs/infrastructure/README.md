---
type: concept
title: Infrastructure Overview / インフラストラクチャ概要
description: Overview of the Infrastructure implementation / インフラストラクチャ実装の概要
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - infrastructure
  - overview
sources:
  - src/logit_router/server.py
---

# Infrastructure Overview / インフラストラクチャ概要

**[English]**
The Infrastructure layer provides an HTTP serving environment for the Logit Router using FastAPI and Uvicorn. By keeping model instances resident in memory, the server minimizes per-request dispatch latency.

**[Japanese]**
インフラストラクチャ層は、FastAPI および Uvicorn を用いて Logit Router を HTTP サービスとして提供します。モデルインスタンスをメモリ上に常駐させることで、リクエストごとのディスパッチ遅延を最小化します。

## Web Server and API / WebサーバーとAPI

**[English]**
The routing API is implemented with FastAPI. Model weights are managed statefully via lifespan context handlers to avoid redundant storage reads or GPU allocation overhead across requests.

For implementation details on memory residency and inference endpoints, see the [FastAPI Server](fastapi_server.md) documentation.

**[Japanese]**
ルーティングAPIは FastAPI により実装されています。リクエスト処理ごとのストレージ読み込みやGPUメモリ確保オーバーヘッドを回避するため、モデルの重みはライフスパンコンテキストを通じて常駐管理されます。

メモリ常駐化および推論エンドポイントの実装仕様については [FastAPI Server (FastAPI サーバー)](fastapi_server.md) を参照してください。
