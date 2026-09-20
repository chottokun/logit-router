---
type: concept
title: Single Forward Pass Routing
description: Detailed mechanism of single forward pass routing, prefill extraction, and Sliced LM-Head
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - architecture
  - prefill
  - sliced-lm-head
sources:
  - docs/raw/ret_2.md
---
# Single Forward Pass Routing

Standard autoregressive text generation is too slow for real-time routing. We avoid this by utilizing a single forward pass (Prefill).

## Key Features

1. **Sliced LM-Head**: We avoid multiplying against the entire vocabulary. We only compute logits for the candidate labels (e.g. A, B, C) by slicing the LM-Head weight matrix.
2. **A/B/C Index Mapping**: We map complex intent labels to simple single tokens (A, B, C) to ensure token boundaries are stable and to reduce complexity.
3. **Qwen Prefill**: The Qwen model backbone is executed once with `use_cache=False` to save memory and avoid KV cache overhead.
