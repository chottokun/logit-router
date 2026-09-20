# Knowledge Base Change Log

## 2026-09-20
* **Creation**: LLM-Wiki (OKF v0.2) ナレッジベースの初期化（`docs/architecture/`, `docs/domain/`, `docs/infrastructure/`, `docs/references/`）。
* **Creation**: `docs/architecture/single_forward_pass_routing.md` - 単一フォワードパス（Prefill）、インデックス射影方式、Sliced LM-Head による極小レイテンシ計算量削減の設計原則を文書化。
* **Creation**: `docs/domain/routing_primitives.md` - `RouteRequest`, `RouteResult`, `FallbackRouter` のドメインモデルと確信度・エントロピー判定仕様。
* **Creation**: `docs/infrastructure/fastapi_server.md` - FastAPI / Uvicorn による Web API サーバーの構成と Lifespan モデルロード仕様。
* **Creation**: `docs/references/benchmarks_and_metrics.md` - CUDA Event 精密レイテンシ計測、要因分解プロファイリング、位置バイアス（順列テスト）、OOD 拒絶評価プロシージャ。
* **Update**: `benchmarks/bench_profile.py`, `benchmarks/bench_scaling.py`, `benchmarks/test_robustness.py` の追加と RTX 3060 での実機検証。
* **Revision**: すべてのドキュメントを大幅に拡充し、英語 (English) と日本語 (Japanese) の完全なバイリンガル形式 (Bilingual JP/EN) に改訂。`docs/raw` の参照を削除し、実際のソースコードを根拠とするように修正。
* **Creation**: `benchmarks/eval_suite.py` および `benchmarks/data/eval_cases.json`（5大ドメイン50問）の構築と実機評価、`docs/references/eval_report.md` の発行。
* **Tone & Phrasing Revision**: 全ドキュメントの文体を技術者向けの実直・客観的な記述に推敲。過大な表現や誇張を排除し、数式、アルゴリズム計算量、および実機測定データ（RTX 3060）に基づく客観的な記述に統一。
* **Architecture Expansion**: `docs/architecture/gemma_and_quantization.md` - Gemma 2 / Gemma 4 の 262K 超巨大語彙に対する Sliced LM-Head のスケーリング優位性および 4-bit/8-bit 量子化仕様を文書化。
* **Deep Evaluation Suite**: `docs/references/deep_benchmarks_specification.md` - 10大ドメイン計100問の超網羅的データセット仕様（`deep_eval_cases.json`）を策定。
* **Empirical Speedup & Multi-Model Matrix**: `docs/references/comparison_vs_generation_report.md`（通常生成との直接対決実測: 2倍〜8.8倍高速化）および `docs/references/robustness_matrix_report.md`（順列一致率 100%）を発行。
* **Critical Failure Mode Analysis & Constructive Cascade Architecture**:
  - 10大ドメイン100問の実機実測（`google/gemma-4-E2B-it`: 95.0% 首位）から、1B未満モデルの「過信誤分類 (Overconfident Misclassification)」および「難易度判定の逆転現象 (Adverse Selection)」を批判的に解明。
  - 解決策として、`Qwen2.5-1.5B`（第1層: 約35ms）と `Gemma-4-E2B`（第2層: 95%精度）を統合した「2段階カスケードルーティング（Two-Tier Cascade Routing）」およびエントロピー＋ロジットマージンによる多層信頼度キャリブレーションを技術設計書（`docs/architecture/cascade_routing_and_critical_analysis.md`）として体系化。
  - `docs/references/benchmarks_and_metrics.md`, `docs/references/multi_model_benchmark_report.md`, `docs/README.md` を最新実測知見に基づき全面的に更新。

