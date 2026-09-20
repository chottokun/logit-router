---
type: concept
title: FastAPI Server / FastAPI サーバー
description: Details of the FastAPI server implementation, lifespan events, and inference endpoints / FastAPIサーバーの実装、ライフサイクルイベント、推論エンドポイントの詳細
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - infrastructure
  - fastapi
  - deployment
sources:
  - src/logit_router/server.py
---

# FastAPI Server / FastAPI サーバー

**[English]**
Deploying an LLM for real-time routing requires strict control over when and how the model is loaded into VRAM/RAM. The Logit Router uses FastAPI to provide a fast, asynchronous HTTP interface, optimized for minimal overhead.

**[Japanese]**
リアルタイムルーティングのためにLLMをデプロイするには、モデルをいつ、どのように VRAM / RAM にロードするかを厳密に制御する必要があります。Logit Router は FastAPI を使用して、最小限のオーバーヘッドに最適化された、高速で非同期の HTTP インターフェースを提供します。

## Model Residency via Lifespan Events / ライフスパンイベントによるモデルの常駐

**[English]**
Loading an LLM from disk and moving its weights to the GPU takes several seconds to minutes. It is impossible to do this on every request. 
Instead, we use FastAPI's `@asynccontextmanager` lifespan events. When the server starts, the `lifespan` function is triggered. It initializes the `LogitRouter` object (which loads the model weights and tokenizer) exactly once and attaches it to the global `app.state`. 
When the server shuts down, the lifespan context yields and cleans up resources.

Environment variables like `LOGIT_ROUTER_MODEL` and `LOGIT_ROUTER_DEVICE` are used within the lifespan context to configure the model dynamically at runtime.

**[Japanese]**
ディスクからLLMをロードし、その重みを GPU に移動するには数秒から数分かかります。これをリクエストごとに実行することは不可能です。
代わりに、FastAPI の `@asynccontextmanager` ライフスパンイベントを使用します。サーバーが起動すると、`lifespan` 関数がトリガーされます。この関数は、`LogitRouter` オブジェクト（モデルの重みとトークナイザーをロードする）を正確に一度だけ初期化し、それをグローバルな `app.state` にアタッチします。
サーバーがシャットダウンすると、ライフスパンコンテキストが `yield` され、リソースがクリーンアップされます。

`LOGIT_ROUTER_MODEL` や `LOGIT_ROUTER_DEVICE` などの環境変数は、ライフスパンコンテキスト内で使用され、実行時にモデルを動的に構成します。

## Inference Endpoint API Specification / 推論エンドポイントAPI仕様

**[English]**
The primary entry point is the `POST /route` endpoint.

**Request (`RouteRequestModel`)**:
Requires a JSON body containing `context` (string), `instruction` (string), `choices` (list of strings), and an optional `temperature` (float). Validation is automatically handled by Pydantic.

**Response (`RouteResponseModel`)**:
Returns the structured data from the router, including `best_choice`, `best_letter`, `confidence`, `entropy`, and the probabilistic `distribution` over the choices.

**Error Handling**:
If the number of choices exceeds the model's `max_choices`, a `400 Bad Request` is returned. Other internal failures (e.g., CUDA OOM) are caught and returned as a `500 Internal Server Error`.

**[Japanese]**
主要なエントリポイントは `POST /route` エンドポイントです。

**リクエスト (`RouteRequestModel`)**:
`context`（文字列）、`instruction`（文字列）、`choices`（文字列のリスト）、およびオプションの `temperature`（浮動小数点数）を含む JSON ボディが必要です。検証は Pydantic によって自動的に処理されます。

**レスポンス (`RouteResponseModel`)**:
`best_choice`、`best_letter`、`confidence`、`entropy`、および選択肢に対する確率的 `distribution`（分布）を含む、ルーターからの構造化データを返します。

**エラー処理**:
選択肢の数がモデルの `max_choices` を超えた場合、`400 Bad Request` が返されます。その他の内部障害（CUDA の OOM など）はキャッチされ、`500 Internal Server Error` として返されます。
