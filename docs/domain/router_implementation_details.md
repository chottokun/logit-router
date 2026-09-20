---
type: concept
title: LogitRouter Implementation Details / LogitRouter 実装詳細仕様
description: Concrete code-level implementation details of LogitRouter, tokenizer whitespace handling, attention fallback, AWQ quantized heads, and compiler optimizations / LogitRouterのコードレベル具象実装仕様、トークナイザー空白処理分岐、アテンション耐障害設計、AWQ量子化ヘッド、およびtorch.compile最適化
status: stable
generated:
  by: engineer/agent
  at: "2026-09-20T06:50:00Z"
tags:
  - domain
  - implementation
  - code-details
  - router
  - optimizations
sources:
  - src/logit_router/router.py
  - src/logit_router/optimizations.py
---

# LogitRouter Implementation Details / LogitRouter 実装詳細仕様

**[English]**
This document details the concrete code-level implementation of `LogitRouter` (`src/logit_router/router.py`) and runtime optimizations (`src/logit_router/optimizations.py`). It covers tokenizer whitespace bifurcations, attention backend fallbacks, dual-mode sliced LM-heads (standard vs. quantized AWQ), and `torch.compile` integrations.

**[Japanese]**
本ドキュメントは、`LogitRouter`（`src/logit_router/router.py`）および実行時最適化（`src/logit_router/optimizations.py`）の具象コードレベルの実装仕様書です。モデル系列ごとのトークナイズ空白処理分岐、アテンションバックエンドのフォールバック設計、通常ヘッドとAWQ量子化ヘッドのデュアルモード対応、および `torch.compile` 統合の詳細を記述します。

---

## 1. Class Structure: `LogitRouter` (`router.py`) / クラス構造

`LogitRouter` は、モデル重みのライフサイクル管理、選択肢インデックストークンの事前抽出、および Prefill フォワードパスからのロジット抽出を担うメインクラスです。

### 1.1 初期化シーケンス (`__init__`)

```python
class LogitRouter:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        device=None,
        max_choices=10,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str | dict | None = None,
        is_awq: bool = False,
    ):
```

#### A. アテンションバックエンドの耐障害設計 (Attention Fallback)
Gemma 2/4 と Qwen/Llama 系列で最適なアテンションカーネルを動的に切り替えます：
- **Gemma 系列 (`is_gemma2`)**: `sdpa` (Scaled Dot-Product Attention) を優先（Gemma のソフトキャッピングや RoPE 実装との互換性を確保）。
- **Qwen / Llama 系列**: まず `flash_attention_2` を試し、未インストール環境や非対応 GPU では自動的に `sdpa` にフォールバックする例外捕捉機構を実装。

#### B. モデル系列に応じたトークナイズ空白処理 (Tokenizer Whitespace Bifurcation)
トークナイザーの語彙エンコーディング方式の違いによるトークン不一致を防止するため、モデル種別に応じた分岐を実装しています：
- **Gemma 系列 (SentencePiece / 256k語彙)**:
  ```python
  self.choice_token_ids = [
      self.tokenizer.encode(letter, add_special_tokens=False)[-1]
      for letter in self.choice_letters
  ]
  ```
  ※ Gemma は語頭空白なしの単独アルファベット文字（`'A'` 等）をそのまま独立したトークンとして表現します。
- **Qwen / Llama 系列 (Byte-level BPE / 15万語彙)**:
  ```python
  self.choice_token_ids = [
      self.tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]
      for letter in self.choice_letters
  ]
  ```
  ※ BPE では語頭空白プレフィックス（`' A'`）としてエンコードすることで、プロンプトの `Answer: A` におけるコロン直後の空白付きトークン ID と完全一致させます。

#### C. Sliced LM-Head 重みの事前固定と AWQ 量子化ヘッド対応 (Dual-Mode LM-Head)
```python
choice_token_tensor = torch.tensor(self.choice_token_ids, device=self.device)
self.is_sliced_head = False
self.choice_head_weights = None
if hasattr(self.model.lm_head, "weight") and self.model.lm_head.weight is not None:
    self.choice_head_weights = (
        self.model.lm_head.weight[choice_token_tensor].detach().clone()
    )
    self.is_sliced_head = True
```
- **通常モデル (`is_sliced_head = True`)**:
  - `lm_head.weight` から選択肢トークン行（`max_choices` 行）のみをスライスし、`.detach().clone()` して `self.choice_head_weights` として GPU メモリに事前確保。
  - 推論時は、プロンプト末尾の隠れ状態（サイズ `1 x hidden_size`）と、この事前固定されたスライス重み（サイズ `K x hidden_size`）との転置行列積のみを計算。全語彙（15万〜26万語）への巨大 MatMul を完全にバイパスします。
- **量子化モデル (AWQ / GPTQ) 対応フォールバック (`is_sliced_head = False`)**:
  - 量子化された LM-Head は重みが直接のテンソル（`.weight`）として保持されず、カスタムカーネル（`qweight`, `scales`, `qzeros` 等）としてパックされています。
  - この場合、事前スライスを行わず、末尾隠れ状態を量子化 LM-Head に渡した直後に `full_logits[:, choice_token_tensor]` として必要なインデックスを抽出する安全なフォールバックパスを備えています。
  - **実機検証結果**: `gptqmodel` / `autoawq` バックエンドによる AWQ Marlin カーネルの適用時、このフォールバックパスを通じた実行においても、`Qwen2.5-1.5B-Instruct-AWQ` で **p50 レイテンシ 31.41 ms（正解率 84.0%）** という BF16 ベースライン（34.84 ms）を上回る高速処理が実証されています。また、`bitsandbytes` 4-bit (`load_in_4bit=True`) では `lm_head.weight` が保持されるため `is_sliced_head = True` が維持され、`google/gemma-4-E2B-it` で VRAM を 32% 削減しながら **91.0% の高精度** を維持することが確認されています。

---

## 2. Inference & Prompt Pipeline: `route()` / 推論パイプライン

`@torch.inference_mode()` デコレータにより、勾配計算および Autograd グラフ構築のオーバーヘッドを完全に排除して実行されます。

### 2.1 プロンプト生成と ChatML フォールバック
```python
formatted_choices = "\n".join(
    [f"{self.choice_letters[i]}. {choice}" for i, choice in enumerate(choices)]
)
messages = [
    {
        "role": "system",
        "content": "You are a fast routing engine. Select the single best choice based strictly on the context.",
    },
    {
        "role": "user",
        "content": f"Context: {context}\n\nTask: {instruction}\n\nChoices:\n{formatted_choices}\n\nSelect the single correct option letter.",
    },
]
```
- `tokenizer.apply_chat_template` を呼び出し、プロンプト末尾が `Answer: ` で終わるよう正規化。
- トークナイザーにチャットテンプレートが未定義の基本モデル（Base Model）に対しても、ChatML 形式（`<|im_start|>`）のテンプレートにフォールバックする例外処理を実装。

### 2.2 Backbone Prefill と末尾トークン抽出
```python
outputs = self.backbone(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    use_cache=False,  # KVキャッシュを確保しない
    return_dict=True,
)
last_hidden_state = outputs.last_hidden_state[:, -1, :]  # 末尾位置のみ抽出
```
- `use_cache=False` により、不要な Past Key-Value キャッシュの VRAM 割り当てを抑止。
- 出力テンソルから最後のトークン位置 `[:, -1, :]` の隠れ状態のみを取り出し、メモリ効率を最大化。

### 2.3 確率正規化とエントロピー計算
```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
entropy = -sum(p * math.log(p + 1e-9) for p in probs)
```
- 温度パラメータ `temperature`（デフォルト 1.0）によるロジットスケーリング。
- ソフトマックス正規化後、Shannon Entropy $H(P) = -\sum p_i \ln p_i$ を数値安定性（$10^{-9}$ オフセット）を考慮して算出。

---

## 3. Runtime Optimizations (`optimizations.py`) / 最適化機能

### 3.1 `apply_torch_compile`
```python
def apply_torch_compile(
    router: LogitRouter, mode: str = "reduce-overhead"
) -> LogitRouter:
    router.backbone = torch.compile(router.backbone, mode=mode, fullgraph=False)
    return router
```
- `router.backbone`（Transformer 本体系）に対して PyTorch 2.x の Inductor コンパイラを適用。
- `mode="reduce-overhead"`（CUDA Graphs の有効化）により、GPU カーネル投入の CPU オーバーヘッドを削減。

### 3.2 `FallbackRouter`
```python
class FallbackRouter:
    def __init__(
        self,
        router: LogitRouter,
        entropy_threshold: float = 0.9,
        margin_threshold: float = 0.1,
        fallback_fn: callable | None = None,
    ):
```
- 基本ルーターで推論を実行後、不確実性条件を評価：
  1. $H(P) > \text{entropy\_threshold}$（エントロピーが高い / 迷っている）
  2. $P_{\text{top1}} - P_{\text{top2}} < \text{margin\_threshold}$（上位2候補の確率差が僅差）
- 上記条件に該当した場合、指定された `fallback_fn`（例: 大規模モデル呼び出しやフォールバックカテゴリへのルーティング）を自動実行。
