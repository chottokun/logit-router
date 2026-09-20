---
type: concept
title: Benchmarks and Metrics
description: Details of latency measurement, position bias, and Out-of-Distribution (OOD) detection
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - references
  - benchmarks
sources:
  - docs/raw/ret_2.md
---
# Benchmarks and Metrics

## CUDA Event Measurement
Standard python `time` module is inaccurate due to asynchronous GPU execution. We use `torch.cuda.Event(enable_timing=True)` for precise latency decomposition and measurement.

## Position Bias
LLMs have a tendency to prefer the first (`A`) or last choices. Techniques like prompt anchoring ("Choices are listed in arbitrary order.") or lightweight ensembling (shuffling choices) are used to mitigate this.

## OOD (Out-of-Distribution) Detection
Using entropy and confidence, the system can detect when a request does not fit any of the provided choices, allowing it to gracefully fail or fallback.
