---
type: concept
title: Routing Primitives
description: Description of routing primitives like RouteRequest, RouteResult, and FallbackRouter
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - domain
  - primitives
  - fallback
sources:
  - docs/raw/ref_1.md
  - docs/raw/ret_2.md
---
# Routing Primitives

## RouteRequest and RouteResult
The system takes a dynamic `RouteRequest` consisting of a context, an instruction, and arbitrary choices. It returns a `RouteResult` containing the selected choice, the confidence score, the entropy, and the full distribution.

## Confidence and Entropy
- **Confidence**: The probability of the top choice.
- **Entropy**: A metric of uncertainty. High entropy indicates the model is unsure.

## FallbackRouter
When entropy is above a certain threshold (e.g., >0.8), or the margin between top 2 choices is small, a `FallbackRouter` can be triggered to use full Chain-of-Thought (CoT) reasoning or a larger model.
