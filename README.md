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
- 単一フォワードパスで極小レイテンシ（RTX 4090: ~5ms）
- FlashAttention-2 自動検出・SDPA フォールバック
- GPU/CPU 自動判定
- エントロピーベースの信頼度指標
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

router = LogitRouter(model_id='Qwen/Qwen2.5-1.5B-Instruct')
result = router.route(
    context='Stripe webhook failed with status code 403.',
    instruction='担当チームにトリアージしてください。',
    choices=['決済・請求窓口', 'インフラ保守', '一般サポート'],
)
print(result)
```

## Benchmark（ベンチマーク）

| Model | GPU | Latency (avg) |
|---|---|---|
| Qwen2.5-1.5B | RTX 4090 | ~5 ms |
| Qwen2.5-1.5B | RTX 3060 / T4 | ~15 ms |
| Qwen2.5-1.5B | CPU | ~50 ms |
| Qwen2.5-7B | RTX 4090 / A100 | ~15 ms |

## API Reference

### `LogitRouter`
モデルの読み込み、キャッシュ管理、および推論処理（ルーティング）を実行するメインクラスです。

### `RouteResult`
ルーティング結果を格納する dataclass です。選択された選択肢、インデックス、各選択肢の確率、およびエントロピーベースの信頼度指標などを保持します。

## License

MIT
