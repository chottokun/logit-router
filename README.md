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

以下の数値は、ローカル環境（NVIDIA GeForce RTX 3060 12GB, PyTorch 2.x SDPA, batch_size=1）において実際にスクリプトを実行して計測された実測値です。未計測の環境や推測値は記載していません。

### 1. タスク総合評価 (`benchmarks/eval_suite.py`)
5大ドメイン（カスタマーサポート、ツール選択、モデルルーティング、セキュリティ、感情分析）計50問のテストデータセットに対する実測値：

| 評価モデル | 全体正解率 | 平均レイテンシ | 最小レイテンシ | 確信度 (平均) | エントロピー (平均) |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 80.00% (40/50) | 55.07 ms | 23.12 ms | 0.8820 | 0.2941 |

### 2. 要因分解プロファイル (`benchmarks/bench_profile.py`)
単一クエリ（系列長 約135トークン, CUDA Event 計測）における処理フェーズ別の実行時間：

| 処理フェーズ | 実測時間 (ms) | 割合 (%) | 備考 |
|---|---|---|---|
| トークナイズ & テンソル転送 | 0.46 ms | 1.62% | CPU処理 & HtoD転送 |
| Backbone Forward Pass | 27.97 ms | 97.56% | Transformer Prefill計算 |
| Sliced LM-Head MatMul | 0.05 ms | 0.16% | 3選択肢トークンのみの射影 |
| 後処理 (Softmax・エントロピー等) | 0.19 ms | 0.66% | 確率正規化と辞書構築 |
| **合計推論時間** | **28.67 ms** | **100.0%** | モデル: Qwen2.5-0.5B-Instruct |

※ 他のハードウェア環境（RTX 4090, A100 等）やモデルサイズ（7B等）の数値は、実機測定が完了次第順次追加します。

## API Reference

### `LogitRouter`
モデルの読み込み、キャッシュ管理、および推論処理（ルーティング）を実行するメインクラスです。

### `RouteResult`
ルーティング結果を格納する dataclass です。選択された選択肢、インデックス、各選択肢の確率、およびエントロピーベースの信頼度指標などを保持します。

## License

MIT
