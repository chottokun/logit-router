---
type: concept
title: Vision Routing
description: Architecture and implementation details of the VisionLogitRouter for multimodal LLM routing.
status: approved
generated: false
tags:
  - architecture
  - vision
  - multimodal
  - gemma
sources:
  - "src/logit_router/vision_router.py"
  - "src/logit_router/router.py"
---

# Vision Routing (ビジョンルーティング)

This document explains the architecture and implementation of the `VisionLogitRouter`, which enables routing decisions based on multimodal inputs (image and text).

このドキュメントでは、マルチモーダル入力（画像とテキスト）に基づくルーティング判定を可能にする `VisionLogitRouter` のアーキテクチャと実装について説明します。

## Overview (概要)

The `VisionLogitRouter` extends the single forward pass extraction mechanism to vision-language models (Vision2Seq), such as `google/gemma-4-E2B-it`. It allows the system to process an image along with a text context and instruction to directly output a routing decision by extracting the logits of specified choices.

`VisionLogitRouter` は、単一フォワードパスの抽出メカニズムを `google/gemma-4-E2B-it` などのビジョン言語モデル (Vision2Seq) に拡張します。画像とテキストのコンテキストおよび指示を処理し、指定された選択肢のロジットを抽出することで、直接ルーティング判定を出力できるようにします。

## Vision2Seq Pipeline (Vision2Seq パイプライン)

Vision language models utilize an `AutoProcessor` to handle both image preprocessing and text tokenization. The processed multimodal inputs are then passed to an `AutoModelForVision2Seq` model. 

ビジョン言語モデルは、画像の前処理とテキストのトークナイズの両方を処理するために `AutoProcessor` を使用します。処理されたマルチモーダル入力は、次に `AutoModelForVision2Seq` モデルに渡されます。

1. **Multimodal Prompting (マルチモーダルプロンプティング):** The `AutoProcessor.apply_chat_template` is used to format the instruction, context, and choices into a standard conversation prompt that includes an `<image>` token.
   
   **マルチモーダルプロンプティング:** `AutoProcessor.apply_chat_template` を使用して、指示、コンテキスト、および選択肢を `<image>` トークンを含む標準的な会話プロンプトにフォーマットします。

2. **Image Processing (画像処理):** The provided PIL Image is processed into `pixel_values` (or equivalent feature maps depending on the model's vision encoder).
   
   **画像処理:** 提供された PIL 画像は `pixel_values`（またはモデルのビジョンエンコーダに依存する同等の特徴マップ）に処理されます。

3. **Inference (推論):** Both textual and visual features are passed into the model. The forward pass is executed exactly once (`use_cache=False`), requesting hidden states to compute logits efficiently.

   **推論:** テキストと視覚の両方の特徴がモデルに渡されます。フォワードパスは1回だけ実行され（`use_cache=False`）、ロジットを効率的に計算するために隠れ状態を要求します。

## Sliced LM-Head Constraints (Sliced LM-Head の制約事項)

Just like the standard `LogitRouter`, the `VisionLogitRouter` attempts to apply the **Sliced LM-Head** optimization to avoid computing logits for the entire vocabulary.

標準の `LogitRouter` と同様に、`VisionLogitRouter` は語彙全体のロジット計算を避けるために **Sliced LM-Head** 最適化を適用しようとします。

1. **Dynamic Check (動的チェック):** At initialization, the router inspects the model to locate the language model head (`lm_head`). If it exists and contains a standard linear `weight` attribute, the weights for the required choice tokens are extracted and cloned.
   
   **動的チェック:** 初期化時に、ルーターはモデルを検査して言語モデルのヘッド (`lm_head`) を特定します。存在し、標準の線形 `weight` 属性が含まれている場合、必要な選択肢トークンの重みが抽出されて複製されます。

2. **Hidden States Extraction (隠れ状態の抽出):** During the forward pass, the model must output hidden states (`output_hidden_states=True`). The last hidden state of the final token is extracted.

   **隠れ状態の抽出:** フォワードパス中、モデルは隠れ状態を出力する必要があります (`output_hidden_states=True`)。最後のトークンの最後の隠れ状態が抽出されます。

3. **Fallback (フォールバック):** If the model architecture (e.g., heavily abstracted Vision2Seq wrappers) hides the hidden states or uses a quantized head (like AWQ) without a `weight` attribute, the router automatically falls back to full logit generation and subsequent slicing.

   **フォールバック:** モデルアーキテクチャ（高度に抽象化された Vision2Seq ラッパーなど）が隠れ状態を隠蔽している場合や、`weight` 属性のない量子化ヘッド（AWQ など）を使用している場合、ルーターは自動的にフルロジット生成とその後のスライスにフォールバックします。

## Multi-Modal Prompt Structure (マルチモーダルプロンプト構造)

The vision router injects images alongside formatted text choices. To ensure accuracy, the prompt is structured clearly:

ビジョンルーターは、フォーマットされたテキストの選択肢とともに画像を注入します。精度を確保するために、プロンプトは明確に構成されています。

```text
Context: {context}

Task: {instruction}

Choices:
A. {choice_1}
B. {choice_2}
C. {choice_3}

Select the single correct option letter.
Answer: 
```

The system ensures that the generation token (`Answer: `) is placed directly at the end of the prompt so that the very next predicted token corresponds to the logit probabilities of the choices `['A', 'B', 'C']`.

システムは、生成トークン（`Answer: `）がプロンプトの最後に直接配置されることを保証し、次に予測されるトークンが選択肢 `['A', 'B', 'C']` のロジット確率に対応するようにします。

