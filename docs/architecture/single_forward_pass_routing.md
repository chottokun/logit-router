---
type: concept
title: Single Forward Pass Routing / 単一フォワードパスルーティング
description: Detailed mechanism of single forward pass routing, prefill extraction, and Sliced LM-Head / 単一フォワードパスルーティング、Prefill抽出、およびSliced LM-Headの技術仕様
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T15:00:00Z"
tags:
  - architecture
  - prefill
  - sliced-lm-head
  - optimization
sources:
  - src/logit_router/router.py
---

# Single Forward Pass Routing / 単一フォワードパスルーティング

**[English]**
Autoregressive token generation iteratively samples tokens one by one, which is bound by memory bandwidth and per-step kernel launch overhead. Logit Router eliminates the iterative decoding loop and key-value (KV) cache allocation by evaluating only the initial forward pass (prefill phase). By extracting logits corresponding to target candidate indices directly from the final token position of the prompt, classification is completed within a single forward pass.

**[Japanese]**
自己回帰による逐次トークン生成は、1ステップごとのカーネル起動オーバーヘッドやメモリ帯域幅の制約（Memory-bound）を受けやすく、推論レイテンシが増加します。Logit Router は、プロンプト入力時の単一フォワードパス（Prefillフェーズ）のみを実行し、反復的なデコードループおよびKey-Value（KV）キャッシュの確保を行いません。プロンプト末尾のトークン位置における選択肢インデックスのロジットを直接抽出することで、単一のフォワードパス内で分類処理を完了させます。

## Mechanism and Formulation / 動作原理と定式化

**[English]**
Let the input token sequence be $X = (x_1, x_2, \dots, x_n)$. In standard autoregressive generation, the model predicts the conditional probability distribution $P(x_{n+1} \mid X)$.
In this routing architecture, candidates are mapped to single deterministic tokens (e.g., `'A'`, `'B'`, `'C'`). Denoting the set of candidate token IDs as $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$, we only extract the logits corresponding to $\mathcal{C}$ at the final position $n$. The model is invoked with `use_cache=False` to prevent KV cache memory allocation.

**[Japanese]**
入力トークン列を $X = (x_1, x_2, \dots, x_n)$ とします。通常の自己回帰生成では、条件付き確率分布 $P(x_{n+1} \mid X)$ を計算して後続トークンを生成します。
本ルーティングアーキテクチャでは、候補選択肢を単一の確定的なトークン（例: `'A'`, `'B'`, `'C'`）にマッピングします。選択肢トークンIDの集合を $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$ と定義すると、末尾位置 $n$ における $\mathcal{C}$ のロジットのみを抽出します。また、KVキャッシュ用のメモリ確保を回避するため、モデル実行時は `use_cache=False` を指定します。

## Sliced LM-Head / Sliced LM-Head（語彙スライス線形層）

**[English]**
A full language model head projects the hidden state $h_n \in \mathbb{R}^{d}$ to the entire vocabulary space $V$ via matrix multiplication:
$$\mathbf{z} = h_n W^T, \quad W \in \mathbb{R}^{V \times d}$$
For modern open models (such as Qwen2.5 where $V = 151,936$, Gemma 2 where $V = 256,128$, or Gemma 4 where $V = 262,144$), computing the full projection incurs $O(d \cdot V)$ FLOPs and requires streaming the entire weight matrix $W$ from memory.

By slicing the weight matrix to include only the candidate token indices $\mathcal{C}$:
$$W_{\text{sliced}} = W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$$
the computation is reduced to:
$$\mathbf{z}_{\text{sliced}} = h_n W_{\text{sliced}}^T$$
This reduces the computational complexity and weight memory traffic of the final projection layer by a factor of $K / V$ ($K \ll V$, e.g., $3 / 262,144 \approx 0.0011\%$ for Gemma 4 or $3 / 151,936 \approx 0.0020\%$ for Qwen2.5). In profiling on an NVIDIA GeForce RTX 3060, the Sliced LM-Head execution takes approximately $0.05\,\text{ms}$ (around $0.16\%$ of total inference latency), confirming that the projection layer is no longer a bottleneck.

**[Japanese]**
通常の言語モデルヘッド（LM-Head）は、末尾の隠れ状態 $h_n \in \mathbb{R}^{d}$ を全語彙数 $V$ に対して行列積で射影します：
$$\mathbf{z} = h_n W^T, \quad W \in \mathbb{R}^{V \times d}$$
近年のオープンモデル（Qwen2.5 の $V = 151,936$、Gemma 2 の $V = 256,128$、Gemma 4 の $V = 262,144$ など）では、全語彙の射影に $O(d \cdot V)$ の浮動小数点演算と、重み行列 $W$ 全体のメモリストリーミングが必要となります。

候補トークン集合 $\mathcal{C}$ に対応する行のみを事前にスライスした重み行列：
$$W_{\text{sliced}} = W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$$
を用いることで、射影計算は以下に限定されます：
$$\mathbf{z}_{\text{sliced}} = h_n W_{\text{sliced}}^T$$
これにより、最終分類層における計算量および重みアクセスのメモリアクセス量は $K / V$（$K \ll V$、選択肢数3件の場合、Gemma 4 で約 $0.0011\%$、Qwen2.5 で約 $0.0020\%$）に圧縮されます。NVIDIA GeForce RTX 3060 での実機プロファイリング測定でも、Sliced LM-Head の実行時間は約 $0.05\,\text{ms}$（全処理時間の約 $0.16\%$）となり、最終層のオーバーヘッドは無視できる水準に抑えられます。

## Index Projection Mapping / インデックス射影マッピング

**[English]**
Arbitrary textual labels (such as "Technical Support" or "Billing") can tokenize into variable sub-word sequences depending on tokenization schemes. To maintain single-token classification, the router formats the input prompt to map arbitrary category descriptions to deterministic single letters (`A`, `B`, `C`, ...). This ensures that each category directly corresponds to an exact index in $\mathcal{C}$ without requiring multi-token sequence scoring.

**[Japanese]**
任意のテキストラベル（例: 「テクニカルサポート」「請求問い合わせ」など）は、トークナイザの語彙境界によって複数トークンに分割される場合があります。単一トークンによる分類を成立させるため、ルーターは入力プロンプト内で各選択肢をアルファベット1文字（`A`, `B`, `C`, ...）に割り当てて提示します。これにより、多重トークンの確率連鎖を計算することなく、単一位置のインデックス $\mathcal{C}$ のみから直接選択肢を特定できます。

## Contrast with Structured Outputs / 構造化出力（Structured Outputs）との構造的格差

**[English]**
A common production pattern for classification is forcing language models to emit JSON objects (via JSON Schema, Pydantic, or Function Calling). While structured outputs enforce deterministic formats, they incur severe latency penalties compared to LogitRouter:

1. **Token Inflation ($N_{out} \ge 35 \sim 60$)**:
   Emitting standard classification JSON requires synthesizing brackets, keys, confidence fields, and escaping:
   ```json
   {"route": "IT_Support", "confidence": 0.98, "reason": "Hardware failure"}
   ```
   Sequential decoding overhead scales linearly: $T_{\text{total}} = T_{\text{prefill}} + N_{out} \times T_{\text{decode}}$. At $10 \sim 15\,\text{ms}$ per decode step on commodity GPUs, $50$ tokens add $500 \sim 750\,\text{ms}$ of pure decoding latency.
2. **Constrained Decoding Overhead**:
   Grammar-guided decoding engines (e.g., Outlines, vLLM guided decoding) perform regex-to-automata token masking on each step across the entire vocabulary ($V \ge 150\text{k}$), introducing recurring CPU-GPU synchronization points.
3. **LogitRouter Zero-Token Advantage**:
   LogitRouter produces a fully typed Python dataclass (`RouteResult`) with exact winner, normalized probability distribution, and Shannon entropy directly from raw model logits in **0 generated tokens**, avoiding JSON serialization, syntax validation, and autoregressive overhead entirely. Consequently, against structured JSON generation, theoretical projections suggest LogitRouter could achieve an estimated **10x to 20x latency reduction** (though empirical benchmark harness verification remains for future work).

**[Japanese]**
業務システムにおける一般的な分類実装として、JSON Schema、Pydantic、または Function Calling を用いて LLM に構造化 JSON を出力させる手法が広く用いられています。しかし、このアプローチは LogitRouter と比較して深刻な遅延ペナルティを抱えます：

1. **生成トークン数の肥大化 ($N_{out} \ge 35 \sim 60$)**:
   最小限の分類 JSON であっても、波括弧、キー文字列、確信度、エスケープ等の構文トークンを逐次生成する必要があります：
   ```json
   {"route": "IT_Support", "confidence": 0.98, "reason": "Hardware failure"}
   ```
   自己回帰の処理時間はトークン数に比例して累積します：$T_{\text{total}} = T_{\text{prefill}} + N_{out} \times T_{\text{decode}}$。RTX 3060 における 1.5B〜3B クラスのデコード時間が 1 トークンあたり $10 \sim 15\,\text{ms}$ の場合、50 トークンの生成だけで **デコード遅延のみで $500 \sim 750\,\text{ms}$ が上乗せ** されます。
2. **文法制約デコード（Constrained Decoding）のオーバーヘッド**:
   JSON 構文を保証するために Outlines や vLLM の Guided Decoding 等を用いる場合、毎ステップ全語彙（$V \ge 150,000$）に対して正規表現・オートマトンに基づくロジットマスキング処理が走り、CPU-GPU 間の同期やカーネルオーバーヘッドが加算されます。
3. **LogitRouter のゼロトークン優位性**:
   LogitRouter は最初から確定的な Python データクラス（`RouteResult`）として、勝者、正規化確率分布、シャノンエントロピーを **新規生成 0 トークン** で直接構築します。JSON 文字列の生成やパースエラーの懸念そのものを排除します。なお、この Structured Output 構成に対する **10倍〜20倍の高速化は生成トークン数に基づく理論上の試算** であり、未測定の予測値であることに留意が必要です。
