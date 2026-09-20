---
type: concept
title: Domain Overview / ドメイン概要
description: Overview of the Domain Models / ドメインモデルの概要
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - domain
  - overview
sources:
  - src/logit_router/schema.py
---

# Domain Overview / ドメイン概要

**[English]**
The Domain models define the core data structures and primitive operations of the Logit Router. These models map closely to the underlying system capabilities and define the contract for interactions between the inference engine and the API layer.

**[Japanese]**
ドメインモデルは、Logit Router の中核となるデータ構造と基本操作を定義します。これらのモデルは、基盤となるシステムの機能に密接にマッピングされ、推論エンジンと API レイヤー間の相互作用の契約（コントラクト）を定義します。

## Core Models / コアモデル

**[English]**
- **RouteRequest**: Encapsulates the context, instruction, and available choices for routing.
- **RouteResult**: Captures the best choice along with confidence, entropy, and the full probability distribution.
- **Fallback Evaluation**: Methods embedded within `RouteResult` that evaluate entropy and confidence to determine if a fallback strategy (like triggering a larger model or using full Chain-of-Thought) is required.

For details, refer to [Routing Primitives](routing_primitives.md).

**[Japanese]**
- **RouteRequest**: ルーティングのためのコンテキスト、指示、および利用可能な選択肢をカプセル化します。
- **RouteResult**: 最適な選択肢とともに、確信度、エントロピー、および完全な確率分布をキャプチャします。
- **フォールバック評価**: `RouteResult` に組み込まれたメソッドで、エントロピーと確信度を評価し、フォールバック戦略（より大きなモデルのトリガーや完全な Chain-of-Thought の使用など）が必要かどうかを判断します。

詳細については、[Routing Primitives](routing_primitives.md) を参照してください。
