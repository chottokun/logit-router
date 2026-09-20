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
For modern open models (such as Qwen2.5 where $V = 151,936$ or Gemma 2 where $V = 256,000$), computing the full projection incurs $O(d \cdot V)$ FLOPs and requires streaming the entire weight matrix $W$ from memory.

By slicing the weight matrix to include only the candidate token indices $\mathcal{C}$:
$$W_{\text{sliced}} = W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$$
the computation is reduced to:
$$\mathbf{z}_{\text{sliced}} = h_n W_{\text{sliced}}^T$$
This reduces the computational complexity and weight memory traffic of the final projection layer by a factor of $K / V$ ($K \ll V$). In profiling on an NVIDIA GeForce RTX 3060, the Sliced LM-Head execution takes approximately $0.05\,\text{ms}$ (around $0.16\%$ of total inference latency), confirming that the projection layer is no longer a bottleneck.

**[Japanese]**
通常の言語モデルヘッド（LM-Head）は、末尾の隠れ状態 $h_n \in \mathbb{R}^{d}$ を全語彙数 $V$ に対して行列積で射影します：
$$\mathbf{z} = h_n W^T, \quad W \in \mathbb{R}^{V \times d}$$
近年のオープンモデル（Qwen2.5 の $V = 151,936$ や Gemma 2 の $V = 256,000$ など）では、全語彙の射影に $O(d \cdot V)$ の浮動小数点演算と、重み行列 $W$ 全体のメモリストリーミングが必要となります。

候補トークン集合 $\mathcal{C}$ に対応する行のみを事前にスライスした重み行列：
$$W_{\text{sliced}} = W[\mathcal{C}, :] \in \mathbb{R}^{K \times d}$$
を用いることで、射影計算は以下に限定されます：
$$\mathbf{z}_{\text{sliced}} = h_n W_{\text{sliced}}^T$$
これにより、最終分類層における計算量および重みアクセスのメモリアクセス量は $K / V$（$K \ll V$、選択肢数3件の場合は約 $0.002\%$）に圧縮されます。NVIDIA GeForce RTX 3060 での実機プロファイリング測定でも、Sliced LM-Head の実行時間は約 $0.05\,\text{ms}$（全処理時間の約 $0.16\%$）となり、最終層のオーバーヘッドは無視できる水準に抑えられます。

## Index Projection Mapping / インデックス射影マッピング

**[English]**
Arbitrary textual labels (such as "Technical Support" or "Billing") can tokenize into variable sub-word sequences depending on tokenization schemes. To maintain single-token classification, the router formats the input prompt to map arbitrary category descriptions to deterministic single letters (`A`, `B`, `C`, ...). This ensures that each category directly corresponds to an exact index in $\mathcal{C}$ without requiring multi-token sequence scoring.

**[Japanese]**
任意のテキストラベル（例: 「テクニカルサポート」「請求問い合わせ」など）は、トークナイザの語彙境界によって複数トークンに分割される場合があります。単一トークンによる分類を成立させるため、ルーターは入力プロンプト内で各選択肢をアルファベット1文字（`A`, `B`, `C`, ...）に割り当てて提示します。これにより、多重トークンの確率連鎖を計算することなく、単一位置のインデックス $\mathcal{C}$ のみから直接選択肢を特定できます。
