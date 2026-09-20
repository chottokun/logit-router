---
type: concept
title: Single Forward Pass Routing / 単一フォワードパスルーティング
description: Detailed mechanism of single forward pass routing, prefill extraction, and Sliced LM-Head / 単一フォワードパスルーティング、Prefill抽出、およびSliced LM-Headの詳細メカニズム
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
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
Standard autoregressive text generation is fundamentally bottlenecked by memory bandwidth and sequential token prediction, making it unsuitable for real-time routing engines. The Logit Router bypasses this by utilizing a "Single Forward Pass" (Prefill-only) approach. By evaluating the mathematical background and algorithmic complexity, we can dramatically reduce the latency footprint.

**[Japanese]**
標準的な自己回帰テキスト生成（Autoregressive Text Generation）は、メモリ帯域幅とシーケンシャルなトークン予測によって根本的にボトルネックとなっており、リアルタイムのルーティングエンジンには不向きです。Logit Router は、「単一フォワードパス（Prefillのみ）」のアプローチを利用することでこれを回避します。数学的背景とアルゴリズムの複雑さを評価することで、レイテンシのフットプリントを劇的に削減できます。

## Mechanism and Mathematical Background / メカニズムと数学的背景

**[English]**
The core idea is to extract the logits from the last token of the context prompt. Let $X = (x_1, x_2, ..., x_n)$ be the input sequence. In standard generation, the model computes $P(x_{n+1} | X)$. Because we map our choices to specific tokens (e.g., 'A', 'B', 'C'), we only need to extract the probability distribution over these specific tokens without entering the key-value (KV) cache autoregressive loop. We execute the model with `use_cache=False`.

**[Japanese]**
コアとなるアイデアは、コンテキストプロンプトの最後のトークンからロジットを抽出することです。入力シーケンスを $X = (x_1, x_2, ..., x_n)$ とします。標準的な生成では、モデルは $P(x_{n+1} | X)$ を計算します。私たちのシステムでは、選択肢を特定のトークン（例：'A'、'B'、'C'）にマッピングしているため、Key-Value (KV) キャッシュの自己回帰ループに入ることなく、これらの特定のトークンに対する確率分布を抽出するだけで済みます。モデルは `use_cache=False` で実行されます。

## Sliced LM-Head / Sliced LM-Head (スライスされた言語モデルヘッド)

**[English]**
A standard LM-Head performs a matrix multiplication across the entire vocabulary size $V$ (e.g., $V = 151,936$ for Qwen2.5). This operation requires computing $h \times W^T$, where $h \in \mathbb{R}^{d}$ is the hidden state and $W \in \mathbb{R}^{V \times d}$ is the vocabulary weight matrix. 

By identifying the tokens for choices 'A', 'B', 'C', ..., we create a Sliced LM-Head:
$W_{sliced} \in \mathbb{R}^{K \times d}$
where $K$ is the number of choices (e.g., $K=3$). This turns a $O(d \cdot V)$ operation into a $O(d \cdot K)$ operation, reducing the computational complexity and memory bandwidth required for the final classification layer by over 99%.

**[Japanese]**
標準の LM-Head は、語彙サイズ全体の $V$ （例：Qwen2.5 では $V = 151,936$）にわたって行列積を実行します。この操作は $h \times W^T$ を計算する必要があります（ここで $h \in \mathbb{R}^{d}$ は隠れ状態、$W \in \mathbb{R}^{V \times d}$ は語彙の重み行列です）。

選択肢 'A', 'B', 'C', ... のトークンを特定することで、Sliced LM-Head を作成します：
$W_{sliced} \in \mathbb{R}^{K \times d}$
ここで $K$ は選択肢の数です（例：$K=3$）。これにより、$O(d \cdot V)$ の操作が $O(d \cdot K)$ の操作に変換され、最終的な分類レイヤーに必要な計算量とメモリ帯域幅が99%以上削減されます。

## A/B/C Index Projection / A/B/C インデックス射影

**[English]**
Semantic labels (e.g., "Financial Support", "Technical Support") can often be tokenized into multiple sub-words unpredictably. The Logit Router projects these complex labels onto deterministic single-token indices (`A`, `B`, `C`). The prompt explicitly lists the mapping. This avoids the BPE (Byte-Pair Encoding) split problem and ensures we only need to inspect a single position in the logit distribution.

**[Japanese]**
意味的なラベル（例：「Financial Support」や「Technical Support」など）は、予測不可能な複数のサブワードにトークナイズされることがよくあります。Logit Router は、これらの複雑なラベルを決定論的な単一トークンのインデックス（`A`、`B`、`C`）に射影（投影）します。プロンプトにはこのマッピングが明示的にリストされます。これにより、BPE（Byte-Pair Encoding）の分割問題を回避し、ロジット分布の単一のインデックスのみを検査すれば済むようになります。
