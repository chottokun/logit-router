# Knowledge Base Change Log

## 2026-09-20
* **Creation**: LLM-Wiki (OKF v0.2) ナレッジベースの初期化（`docs/architecture/`, `docs/domain/`, `docs/infrastructure/`, `docs/references/`）。
* **Creation**: `docs/architecture/single_forward_pass_routing.md` - 単一フォワードパス（Prefill）、インデックス射影方式、Sliced LM-Head による極小レイテンシ計算量削減の設計原則を文書化。
* **Creation**: `docs/domain/routing_primitives.md` - `RouteRequest`, `RouteResult`, `FallbackRouter` のドメインモデルと確信度・エントロピー判定仕様。
* **Creation**: `docs/infrastructure/fastapi_server.md` - FastAPI / Uvicorn による Web API サーバーの構成と Lifespan モデルロード仕様。
* **Creation**: `docs/references/benchmarks_and_metrics.md` - CUDA Event 精密レイテンシ計測、要因分解プロファイリング、位置バイアス（順列テスト）、OOD 拒絶評価プロシージャ。
* **Update**: `benchmarks/bench_profile.py`, `benchmarks/bench_scaling.py`, `benchmarks/test_robustness.py` の追加と RTX 3060 での実機検証。
