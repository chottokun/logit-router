---
type: concept
title: FastAPI Server
description: Details of the FastAPI server implementation, lifespan events, and inference endpoints
status: stable
generated:
  by: jules/agent
  at: "2026-09-20T14:15:00Z"
tags:
  - infrastructure
  - fastapi
sources:
  - docs/raw/logit-router.md
---
# FastAPI Server

The routing system is deployed as a Web API using FastAPI and Uvicorn.

## Model Caching with Lifespan Events
To avoid loading the model into GPU memory on every request, we utilize FastAPI's `@asynccontextmanager` lifespan events. The model is loaded once at startup and kept in memory.

## Inference Endpoint
The primary endpoint (`POST /route`) accepts JSON requests, calls the underlying optimized Logit Router, and returns the classification results.
