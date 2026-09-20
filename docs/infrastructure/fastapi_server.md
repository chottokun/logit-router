---
type: concept
title: FastAPI Server / FastAPI サーバー
description: Technical architecture of FastAPI inference service, lifespan management, and API specifications / FastAPI推論サービス、ライフスパン管理、およびAPI仕様の解説
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T15:00:00Z"
tags:
  - infrastructure
  - fastapi
  - deployment
sources:
  - src/logit_router/server.py
---

# FastAPI Server / FastAPI サーバー

**[English]**
Serving an LLM-based router with millisecond latency requirements necessitates keeping model weights resident in accelerator memory (VRAM). The service utilizes FastAPI with asynchronous handlers to expose HTTP endpoints without incurring per-request model initialization overhead.

**[Japanese]**
ミリ秒単位の応答が求められるLLMルーターの運用では、モデル重みをアクセラレータメモリ（VRAM）上に常駐させることが必須となります。本サービスでは、非同期ハンドラを備えたFastAPIを採用し、リクエストごとのモデル初期化オーバーヘッドを発生させずにHTTPエンドポイントを提供します。

## Model Lifecycle via Lifespan Handlers / ライフスパンハンドラによるモデル常駐

**[English]**
Model loading from secondary storage and tensor initialization on GPU devices require several seconds. Loading model weights during request processing would violate latency constraints.
To maintain constant residency, the service utilizes FastAPI's `@asynccontextmanager` lifespan interface:
1. During application startup, `lifespan` initializes the `LogitRouter` instance once, loading the tokenizer and model weights onto the target device (`torch.device("cuda")`).
2. The initialized instance is attached to `app.state.router` for direct reference by path operations.
3. During application shutdown, the context yields and handles GPU resource cleanup.

Configuration parameters (e.g., model path, precision, and device target) are read from environment variables such as `LOGIT_ROUTER_MODEL` and `LOGIT_ROUTER_DEVICE`.

**[Japanese]**
ストレージからのモデル読み込みおよびGPUデバイス上でのテンソル初期化には数秒を要するため、リクエスト処理中にモデルをロードする構成はレイテンシ要件を満たしません。
モデルをメモリ上に常時維持するため、FastAPI の `@asynccontextmanager` によるライフスパンインターフェースを採用しています：
1. アプリケーション起動時に、`lifespan` 関数が `LogitRouter` インスタンスを1度だけ初期化し、トークナイザおよびモデル重みを指定デバイス（`torch.device("cuda")`）へロードします。
2. 初期化済みインスタンスは `app.state.router` に格納され、各ルーティングハンドラから参照されます。
3. アプリケーション停止時にコンテキストが終了し、GPUリソースの解放処理を行います。

モデル識別子、演算精度、計算デバイスなどの設定は、`LOGIT_ROUTER_MODEL` や `LOGIT_ROUTER_DEVICE` などの環境変数から読み込まれます。

## Endpoint Specifications / エンドポイント仕様

**[English]**
The primary interface is the `POST /route` endpoint.

- **Request Body (`RouteRequestModel`)**:
  - `context` (str): Grounding text.
  - `instruction` (str): Categorization query.
  - `choices` (list[str]): Candidate choices (length bounded by `max_choices`).
  - `temperature` (float, optional): Logit temperature (default: `1.0`).
- **Response (`RouteResponseModel`)**:
  - `best_choice` (str), `best_letter` (str), `confidence` (float), `entropy` (float), and `distribution` (dict[str, float]).
- **Error Handling**:
  - Returns `400 Bad Request` if the number of provided choices exceeds `max_choices`.
  - Unhandled exceptions return `500 Internal Server Error` with relevant log records.

**[Japanese]**
主要インターフェースは `POST /route` エンドポイントです。

- **リクエストボディ (`RouteRequestModel`)**:
  - `context` (文字列): 判断基準となる背景情報。
  - `instruction` (文字列): 分類指示クエリ。
  - `choices` (文字列リスト): 選択肢リスト（要素数は `max_choices` 以下）。
  - `temperature` (浮動小数点数, 任意): ロジット温度係数（デフォルト: `1.0`）。
- **レスポンス (`RouteResponseModel`)**:
  - `best_choice` (文字列), `best_letter` (文字列), `confidence` (浮動小数点数), `entropy` (浮動小数点数), `distribution` (辞書型)。
- **エラーハンドリング**:
  - 提供された選択肢数が `max_choices` を超過した場合は `400 Bad Request` を返却します。
  - 予期しない例外が発生した場合は、ログを記録した上で `500 Internal Server Error` を返却します。
