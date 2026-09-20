指示文（プロンプト）や判定カテゴリ、ルーティング先がリクエストごとに動的に変化するケースでは、**「自然言語の指示追従性（Zero-shot 性能）」** と **「極小レイテンシ（数ミリ秒〜十数ミリ秒）」** を両立させる必要があります。

この要件を満たす上で最も堅牢かつ高速なアプローチは、**「動的な選択肢をインデックス（A, B, C...）に射影し、末尾 1 トークンの Logits を競わせる設計」** です。

---

### 1. なぜ「インデックス射影（A, B, C）」が動的ルーティングで最強なのか

動的なルーティングにおいて、候補ラベル文字列（例: `"返金処理"`, `"技術的問い合わせ"`, `"SQL実行"`）の Logits を直接読もうとすると以下の重大な問題が生じます。

1. **BPE トークン分割の不一致**:
単語によって 1 トークンで完結するものと、2〜3 トークンに分解されるものが混在します。2 トークン以上の確率を正確に算出するには各トークンの結合確率（Chain Rule）を計算する必要があり、1 パスでの公平な比較が崩れます。
2. **語彙の事前出現頻度バイアス（Unconditional Prior）**:
LLM の語彙には「学習コーパス内で出現頻度が高いトークンは、文脈に関係なく素の Logit が高くなりやすい」という偏りがあります。
3. **レイテンシの増大**:
リクエストごとに単語をトークナイズし、動的に LM-Head の重みをスライスするオーバーヘッドが発生します。

**解決策**:
任意の選択肢を動的に `A, B, C...`（または `1, 2, 3...`）に割り振り、プロンプト内で定義します。あらかじめ GPU 上に固定キャッシュしておいた「` A`, ` B`, ` C`... に対応する LM-Head 重み」との内積を 1 回取るだけで、完全な 1 フォワードパス・公平な確率比較・オーバーヘッド最小化が実現します。

---

### 2. 動的ルーティング専用 `DynamicJevRouter` 実装

以下の実装は、指示文・コンテキスト・選択肢を動的に受け取り、Qwen のフォワードパス 1 回で各選択肢の確率分布と信頼度（エントロピー）を返します。

```python
import math
from typing import Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class DynamicJevRouter:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_choices: int = 10,
    ):
        self.device = device
        self.max_choices = max_choices

        # 1. モデルとトークナイザの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        self.backbone = self.model.model

        # 2. A〜Z のトークン ID と LM-Head 重みを事前キャッシュ（ゼロオーバーヘッド化）
        self.choice_letters = [chr(ord("A") + i) for i in range(self.max_choices)]
        # Qwen トークナイザで " A", " B" などの単一トークン ID を取得
        self.choice_token_ids = [
            self.tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
            for letter in self.choice_letters
        ]
        choice_token_tensor = torch.tensor(
            self.choice_token_ids, device=self.device, dtype=torch.long
        )

        # 選択肢用の重み行列のみを GPU 上に永続保持 (Max_Choices, Hidden_Dim)
        self.choice_head_weights = (
            self.model.lm_head.weight[choice_token_tensor].detach().clone()
        )

    @torch.inference_mode()
    def route(
        self,
        context: str,
        instruction: str,
        choices: list[str],
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        num_choices = len(choices)
        if num_choices > self.max_choices:
            raise ValueError(
                f"Choices count ({num_choices}) exceeds max_choices ({self.max_choices})"
            )

        # 1. プロンプトの動的構築
        # 選択肢を A, B, C... にフォーマット
        formatted_choices = "\n".join(
            [f"{self.choice_letters[i]}: {choices[i]}" for i in range(num_choices)]
        )

        prompt = (
            f"<|im_start|>system\nYou are an ultra-fast routing engine. "
            f"Select the single best choice based strictly on the context and task.<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context}\n\n"
            f"Task: {instruction}\n\n"
            f"Choices:\n{formatted_choices}\n\n"
            f"Which option is correct? Respond with the option letter only.<|im_end|>\n"
            f"<|im_start|>assistant\nAnswer:"
        )

        # 2. 入力のテンソル化
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        # 3. 単一フォワードパス（KV キャッシュ生成を抑制）
        outputs = self.backbone(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # 4. 末尾トークン（"Answer:" の直後位置）の隠れ状態 (1, Hidden_Dim)
        last_hidden = outputs.last_hidden_state[:, -1, :]

        # 5. 事前キャッシュした重みのうち、今回の選択肢数分だけスライスして内積
        # active_weights: (num_choices, Hidden_Dim)
        active_weights = self.choice_head_weights[:num_choices]
        # logits: (1, num_choices)
        logits = torch.matmul(last_hidden, active_weights.t()).squeeze(0)

        # 6. 温度付き Softmax
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1).cpu().tolist()

        # 7. 判定結果の整理と信頼度指標の算出
        results = {choices[i]: round(probs[i], 4) for i in range(num_choices)}
        best_index = int(torch.argmax(logits).item())

        # エントロピー計算（値が低いほど確信度が高い）
        entropy = -sum(p * math.log(p + 1e-9) for p in probs)

        return {
            "best_choice": choices[best_index],
            "best_letter": self.choice_letters[best_index],
            "confidence": round(probs[best_index], 4),
            "entropy": round(entropy, 4),
            "distribution": results,
        }
```

---

### 3. 動作検証と柔軟なユースケース

上記のコードは、初期化後はバックボーンの重みを再ロードすることなく、指示文やカテゴリ構成を柔軟に変えて呼び出せます。

#### ユースケース A: カスタマーサポートの動的トリアージ

```python
router = DynamicJevRouter(model_id="Qwen/Qwen2.5-7B-Instruct")

# 問い合わせ内容に応じた振り分け
res1 = router.route(
    context="クレジットカードの有効期限が切れたため更新したいのですが、管理画面で保存ボタンを押しても反映されません。",
    instruction="この問い合わせを最も適切な対応チームに振り分けてください。",
    choices=["決済・請求窓口", "システム障害対応", "一般的な操作案内"],
)
print(res1)
# 出力例:
# {
#   'best_choice': '決済・請求窓口',
#   'best_letter': 'A',
#   'confidence': 0.8842,
#   'entropy': 0.3812,
#   'distribution': {'決済・請求窓口': 0.8842, 'システム障害対応': 0.0821, '一般的な操作案内': 0.0337}
# }
```

#### ユースケース B: エージェントの動的ツールルーティング

```python
# 別のタスク（ツール選択）を同一モデルインスタンスで即座に実行
res2 = router.route(
    context="ユーザー発言: 『先週の売上合計を日別にグラフ化して社内Slackに投げて』",
    instruction="次に呼び出すべき最適なエージェントツールを1つ選択してください。",
    choices=[
        "Database_SQL_Runner",
        "Slack_Notification_Tool",
        "Chart_Generator",
        "Web_Search",
    ],
)
print(res2)
# 出力例:
# {
#   'best_choice': 'Database_SQL_Runner',
#   'best_letter': 'A',
#   'confidence': 0.7615,
#   'entropy': 0.6954,
#   'distribution': { ... }
# }
```

---

### 4. 動的運用におけるレイテンシと精度のチューニングポイント

1. **信頼度（Entropy）による安全なフォールバック**:
* 出力に含まれる `entropy` が閾値（例: 0.9 以上）を超えている場合、または 1 位と 2 位の確率差（Margin）が僅差（例: 0.1 未満）の場合は「モデルが判断に迷っている状態」です。
* このときのみ、フルスペックの自己回帰生成（CoT など）や上位の大型モデルにフォールバックさせるルーティング機構を組むことで、全体の 9 割以上のリクエストを十数ミリ秒で捌きつつ、エッジケースの精度を保護できます。


2. **位置バイアス（Position Bias）の抑制**:
* LLM は構造上、選択肢の最初（`A`）や最後をわずかに好む傾向があります。
* 精度を極限まで安定させたい場合、プロンプトの指示文に `Choices are listed in arbitrary order.` などのアンカー文を固定で入れるか、推論時に選択肢の順序をシャッフルする軽微なアンサンブル（2 パス実行）が有効です。


3. **Prefix Caching（共通プロンプトの再利用）**:
* 指示文（Instruction）のテンプレートやシステムプロンプトが共通で、ユーザーの `Context` のみが変わる場合、フレームワーク層で Prefix Caching を有効化（または手元で KV キャッシュの共通プレフィックス部分を静的保持）することで、アテンション計算量をさらに半減させることが可能です。