ローカルの Python 環境（PyTorch / Hugging Face Transformers）で Qwen モデルを駆動し、Jev 型の「単一フォワードパス（Prefill）による Logits 直接抽出」で極小レイテンシ（数ミリ秒〜十数ミリ秒オーダ）を達成するための実装検討と最適化手法を解説します。

---

### 1. 標準的な推論が抱えるボトルネックと削減戦略

標準的な `AutoModelForCausalLM` のフォワードパスをそのまま実行すると、意思決定のみを目的とするタスクにおいては以下の無駄なオーバーヘッドが発生します。

1. **全語彙（Vocab）に対する線形射影の無駄**:
Qwen2.5 の語彙数は 152,064 あります。通常、最終隠れ状態 $h \in \mathbb{R}^{B \times L \times D}$ に対して全語彙分の線形層 $W_{lm\_head} \in \mathbb{R}^{V \times D}$ を計算しますが、判定に必要なのは「候補ラベル（数個〜十数個）の Logits」だけです。
* **対策**: `model.model`（Base Model）の最終隠れ状態を取り出し、候補ラベルに対応する重み行 $W_{lm\_head}[label\_ids, :]$ のみをスライシングして内積を取ることで、LM-Head の計算量を数万分の一に圧縮します。


2. **KV キャッシュの割り当て・管理オーバーヘッド**:
逐次生成（Decode）を行わないため、KV キャッシュの動的メモリ確保やトラッキングは不要です。
* **対策**: `use_cache=False` を明示してメモリ帯域消費を抑えます。


3. **全系列トークンの Logits 保持**:
プロンプトの途中トークンに対する Logits の計算やテンソル保持は不要です。
* **対策**: 最終トークンの隠れ状態のみを対象にします。


4. **PyTorch ランタイムおよび GPU カーネル起動オーバーヘッド**:
Python 側のディスパッチオーバーヘッドや GPU カーネル起動のレイテンシが存在します。
* **対策**: `torch.compile` によるカーネル融合、または静的入力に対する CUDA Graphs の活用。



---

### 2. 極小レイテンシ実装コード

以下のクラス `FastJevClassifier` は、Base Model から末尾隠れ状態を抽出し、事前抽出したターゲット重みとの最小限の行列積で確率を算出します。

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class FastJevClassifier:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
    ):
        self.device = device
        self.dtype = torch_dtype

        # 1. トークナイザとモデルの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FlashAttention-2 を利用可能な場合は指定
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="flash_attention_2",  # 未対応環境では "sdpa"
        )
        self.model.eval()

        # LM-Head の重み参照（射影短縮用）
        # Qwen は tie_word_embeddings=False / True のケースがあるため直接重みを取得
        self.lm_head_weight = self.model.lm_head.weight.detach()

        # Base Transformer バックボーンのみを直接参照
        self.backbone = self.model.model

    @torch.inference_mode()
    def classify(
        self,
        context: str,
        instruction: str,
        candidate_labels: list[str],
        pre_mapped_tokens: list[int] = None,
    ) -> dict[str, float]:
        """コンテキストと質問から各ラベルの確率を単一フォワードパスで計算"""
        # プロンプト末尾はスペースを空けずにトークン境界を固定
        prompt = (
            f"<|im_start|>system\nYou are a decision engine.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context}\nQuestion: {instruction}\n"
            f"Choices: {', '.join(candidate_labels)}<|im_end|>\n"
            f"<|im_start|>assistant\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        # 1. 候補トークン ID の解決（事前計算されていない場合）
        if pre_mapped_tokens is None:
            # プロンプト直後（"Answer:"の直後）に続く形式（先頭スペース有無）を考慮
            token_ids = []
            for label in candidate_labels:
                # " Answer: <label>" の文脈に合わせるため " " を付与したトークンを取得
                t_ids = self.tokenizer.encode(f" {label}", add_special_tokens=False)
                token_ids.append(t_ids[0])
        else:
            token_ids = pre_mapped_tokens

        target_token_tensor = torch.tensor(
            token_ids, device=self.device, dtype=torch.long
        )

        # 2. Base Model による 1 ステップのフォワードパス（KV キャッシュ無効化）
        outputs = self.backbone(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # 3. 最終トークン位置の隠れ状態 (1, Hidden_Dim) のみをスライス
        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        # 4. Sliced LM-Head: 候補トークンに対応する重みのみで内積を計算
        # candidate_weights: (Num_Candidates, Hidden_Dim)
        candidate_weights = self.lm_head_weight[target_token_tensor]

        # target_logits: (1, Num_Candidates)
        target_logits = torch.matmul(last_hidden_state, candidate_weights.t()).squeeze(
            0
        )

        # 5. Softmax による正規化確率
        probs = F.softmax(target_logits, dim=-1).cpu().tolist()

        return dict(zip(candidate_labels, probs))
```

---

### 3. トークン境界（BPE）とプロンプト設計の整合性

Qwen の BPE トークナイザにおいて、Logits を正確に抽出するためには**トークン境界の完全な一致**が必須となります。

* **末尾空白問題**:
プロンプト末尾が `Answer:`（末尾空白なし）の場合、続くトークンは空白付きの `" billing"`（例: ID 45120）として予測されます。末尾が `Answer: `（末尾空白あり）の場合、続くトークンは空白なしの `"billing"`（例: ID 23145）になります。
* **1 トークン境界の保証（A/B/C マッピング推奨）**:
単語ラベル（例: `"technical"`, `"unauthorized"`）は語彙によって 2 トークン以上に分割されるリスクがあります。最も安全で極小レイテンシを保てる設計は、プロンプト内で明示的にアルファベット（A, B, C）に射影することです。

```python
# A, B, C などの 1 文字トークン ID は事前に定数化して GPU メモリ上に保持可能
CHOICE_TOKENS = {
    "A": tokenizer.encode(" A", add_special_tokens=False)[0],
    "B": tokenizer.encode(" B", add_special_tokens=False)[0],
    "C": tokenizer.encode(" C", add_special_tokens=False)[0],
}
```

---

### 4. 極限までレイテンシを削る高度な最適化手法

実運用でさらなるレイテンシ短縮（ミリ秒単位の削減）を狙う場合、以下のチューニングを適用します。

#### (1) `torch.compile` の適用

PyTorch 2.0 以降の Dynamo/Inductor を使用し、Transformer のアテンション層や MLP 層のカーネルを融合します。

```python
# モデルのロード後
classifier.backbone = torch.compile(
    classifier.backbone,
    mode="reduce-overhead",  # カーネル融合と Python オーバーヘッド低減
    fullgraph=False,
)
```

#### (2) CUDA Graphs による CPU ディスパッチオーバーヘッドの排除

バッチサイズおよび入力系列長が固定（または最大長でパディング）できる環境であれば、CUDA Graphs をキャプチャすることで Python インタープリタおよび PyTorch の CPU ランタイムオーバーヘッドをほぼ 0 に抑えられます。

* 静的なテンソルバッファをあらかじめ確保。
* ウォームアップ実行後に `torch.cuda.capture_graph()` でグラフ化。
* 毎リクエスト時はバッファに入力トークンをコピーし、`graph.replay()` を呼ぶだけで GPU 上の計算が即時完了します。

#### (3) 複数質問（Multi-Head / System One 形式）の一括評価

Jev のように 1 つのコンテキストに対して「カテゴリ判定」「緊急度スコア」「真偽判定」など複数の質問を同時に行いたい場合、系列を複数回流すのではなく、**バッチ方向に展開**するか、**コンテキストを共有して並列評価**します。

OpenJev 方式のように、コンテキスト長 $L$ の Prefill を 1 回実行した上で、質問部分のトークンをバッチ化して並列評価することで、共通コンテキストの再計算コストを完全に排除できます。

---

### 5. 正確なレイテンシ計測スクリプト

GPU 上の非同期実行による測定誤差を排除し、正確な処理時間をミリ秒（ms）単位で計測するための手順です。

```python
import time

# ウォームアップ（初回実行時のカーネル初期化・コンパイル処理を完了させる）
for _ in range(5):
    _ = classifier.classify(
        context="Stripe 403 error encountered.",
        instruction="Route this issue.",
        candidate_labels=["billing", "technical", "sales"],
    )
torch.cuda.synchronize()

# CUDA Event を用いた精密測定
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

result = classifier.classify(
    context="Hi, I have been trying to connect Stripe but keep getting a 403 error on authentication.",
    instruction="Which team should handle this?",
    candidate_labels=["billing", "technical", "sales"],
)

end_event.record()
torch.cuda.synchronize()

latency_ms = start_event.elapsed_time(end_event)
print(f"Latency: {latency_ms:.2f} ms")
print(f"Result: {result}")
```

### アプローチのまとめ

| 最適化項目 | 通常の `model(**inputs)` | 極小レイテンシ最適化 |
| --- | --- | --- |
| **LM-Head 計算** | 全語彙（152,064 次元）を行列積 | ターゲット候補（数次元）のみスライス射影 |
| **KV キャッシュ** | キャッシュメモリ確保・管理を実行 | `use_cache=False` で完全停止 |
| **アテンション** | 標準 SDPA / Eager | FlashAttention-2 (`attn_implementation="flash_attention_2"`) |
| **実行グラフ** | 動的ディスパッチ（Python オーバーヘッド大） | `torch.compile(mode="reduce-overhead")` |
| **出力取得** | `outputs.logits` 全体展開後にスライス | 隠れ層末尾ベクトルのみ抽出して計算 |

この構成により、Qwen2.5-7B クラスのモデルであっても、数百トークン程度のコンテキストであれば単一 GPU（RTX 4090 / A100 等）上で 10〜30ms 前後のレイテンシで確率判定を完了させることが可能です。Qwen2.5-1.5B や 0.5B などの軽量モデルを採用した場合は、数ミリ秒台での応答が実現できます。