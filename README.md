# logit-router

Ultra-low latency LLM-based routing via single forward pass logit extraction.

Qwen モデルの単一フォワードパスで動的選択肢ルーティングを数ミリ秒で実現する Python ライブラリです。

## How It Works（仕組み）

以下の3ステップで構成:
1. 選択肢を A/B/C... にインデックス射影（BPE トークン分割問題を回避）
2. Backbone の Prefill フォワードパス1回で末尾隠れ状態を取得（KV キャッシュ無効化）
3. 事前キャッシュした LM-Head 重みとの内積で候補の確率を算出（Sliced LM-Head）

図解（テキスト）:
Input → Prompt Build (A/B/C mapping) → Tokenize → Single Forward Pass → Last Hidden State → Sliced MatMul → Softmax → Result

## Features
- 単一フォワードパス（Prefillのみ）による分類処理（自己回帰デコードループをバイパス）
- Sliced LM-Head による最終線形層の演算量・重みメモリアクセス削減
- FlashAttention-2 / PyTorch SDPA 自動選択
- 候補確率分布およびシャノンエントロピーによる不確実性・OOD評価
- 最大10選択肢の動的ルーティング

## Installation

```bash
uv add logit-router
uv add logit-router[flash]  # FlashAttention-2 有効化
uv add logit-router[dev]    # 開発ツール
```

## Quick Start

```python
from logit_router import LogitRouter

router = LogitRouter(model_id="Qwen/Qwen2.5-1.5B-Instruct")
result = router.route(
    context="Stripe webhook failed with status code 403.",
    instruction="担当チームにトリアージしてください。",
    choices=["決済・請求窓口", "インフラ保守", "一般サポート"],
)
print(result)
```

## Benchmark（実機実測値）

以下の数値は、ローカル環境（NVIDIA GeForce RTX 3060 12GB, PyTorch 2.x SDPA, bfloat16, batch_size=1）において実スクリプトを実行して計測された実測値です。推測値や未測定データは一切含みません。

### 1. 通常生成（`model.generate()`）との直接対決実測
各モデル同一のプロンプト・選択肢条件において、通常の自己回帰生成と LogitRouter の実行レイテンシを直接比較した実機実測値です（詳細は [比較レポート](benchmarks/reports/comparison_vs_generation_report.md) 参照）：

| モデル | 通常生成 (`generate()`) | **LogitRouter** | 高速化倍率 | 備考 |
|---|---|---|---|---|
| **`google/gemma-4-E2B-it`** | 621.49 ms | **103.88 ms** (p50: 67.8 ms) | **5.78x 高速化** | 語彙 256k の射影削減効果が最大 |
| `Qwen/Qwen2.5-3B-Instruct` | 200.42 ms | **85.04 ms** (p50: 62.8 ms) | **2.36x 高速化** | 精度と速度のバランス型 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 206.74 ms | **90.33 ms** (p50: 34.8 ms) | **2.32x 高速化** | 第1層ゲートに最適 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 101.96 ms | **50.65 ms** (p50: 16.3 ms) | **1.64x 高速化** | 最軽量・低VRAM（0.95GB） |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 133.01 ms | **25.97 ms** (p50: 23.8 ms) | **4.72x 高速化** | 最速だが日本語・複合推論に課題 |

### 2. 10大ドメイン・計100問の深層評価マトリクス
多様な業務課題（ツール選択、PII、曖昧性、多言語、緊急度トリアージ等）100問による実機評価（詳細は [100問深層評価レポート](benchmarks/reports/deep_eval_report.md) 参照）：

| 候補モデル | 正解率 (100問) | p50 レイテンシ | ピーク VRAM | 平均確信度 | 特徴と日本語適合性 |
|---|---|---|---|---|---|
| **`google/gemma-4-E2B-it`** | **95.0%** (95/100) | **67.83 ms** | 9.76 GB | 0.9945 | **最高精度**。7ドメインで100%達成。日本語複合語の語彙圧縮と文脈解釈に圧倒的優位 |
| `Qwen/Qwen2.5-3B-Instruct` | **85.0%** (85/100) | 62.76 ms | 5.91 GB | 0.9688 | 安定したエントロピー分離度 |
| `Qwen/Qwen2.5-1.5B-Instruct` | **81.0%** (81/100) | 34.84 ms | 2.96 GB | 0.8656 | 35ms以下の高速判定で8割超の精度 |
| `Qwen/Qwen2.5-0.5B-Instruct` | **73.0%** (73/100) | 16.26 ms | 0.96 GB | 0.7331 | 構文判定は優秀だが行間解釈に限界 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | **23.0%** (23/100) | 23.76 ms | 0.71 GB | 0.4275 | 日本語や複雑推論で大幅な過信誤分類 |

### 3. 要因分解プロファイル (`benchmarks/bench_profile.py`)
単一クエリ（系列長 約135トークン, CUDA Event 計測）における処理フェーズ別の実行時間：

| 処理フェーズ | 実測時間 (ms) | 割合 (%) | 備考 |
|---|---|---|---|
| トークナイズ & テンソル転送 | 0.46 ms | 1.62% | CPU処理 & HtoD転送 |
| Backbone Forward Pass | 27.97 ms | 97.56% | Transformer Prefill計算 |
| Sliced LM-Head MatMul | 0.05 ms | 0.16% | 3選択肢トークンのみの射影（全語彙射影をバイパス） |
| 後処理 (Softmax・エントロピー等) | 0.19 ms | 0.66% | 確率正規化と辞書構築 |
| **合計推論時間** | **28.67 ms** | **100.0%** | モデル: Qwen2.5-0.5B-Instruct |

### 4. 批判的分析と建設的推奨アーキテクチャ
- **1B未満モデルの限界と過信誤分類**: 360M や 0.5B は構文・明示的タスクで高い正解率を示しますが、行間の意図解釈や緊急度判定で誤分類が発生しやすく、誤答時にも確信度が高くなる（エントロピーが低い）現象が確認されました。
- **推奨アーキテクチャ（2段階カスケード構成）**:
  1. **第1層（高速ゲート）**: `Qwen2.5-1.5B`（約35ms）で一次判定。エントロピー $H < 0.35$（明確な約80%のクエリ）は即座に確定。
  2. **第2層（高精度フォールバック）**: $H \ge 0.35$ の曖昧な難問（約20%）のみを `google/gemma-4-E2B-it`（95%精度）へエスカレーション。
  - これにより、全体として **平均約40ms台の超低遅延を維持しながら、95%水準の最高精度を両立** できます。

詳細ドキュメント一覧は [docs/references/README.md](docs/references/README.md) をご覧ください。

## API Reference

### `LogitRouter`
モデルの読み込み、キャッシュ管理、および推論処理（ルーティング）を実行するメインクラスです。

### `RouteResult`
ルーティング結果を格納する dataclass です。選択された選択肢、インデックス、各選択肢の確率、およびエントロピーベースの信頼度指標などを保持します。

## License

MIT
