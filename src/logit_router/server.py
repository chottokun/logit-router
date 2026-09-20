import argparse
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from logit_router.router import LogitRouter


class RouteRequestModel(BaseModel):
    """
    ルーティングリクエストのモデル。
    """

    context: str = Field(..., description="ルーティングの判断基準となるコンテキスト。")
    instruction: str = Field(..., description="実行すべき指示。")
    choices: list[str] = Field(..., description="選択肢のリスト。")
    temperature: float = Field(
        1.0, description="生成時の温度パラメータ（デフォルト値は1.0）。"
    )


class RouteResponseModel(BaseModel):
    """
    ルーティングレスポンスのモデル。
    """

    best_choice: str = Field(..., description="最適な選択肢テキスト。")
    best_letter: str = Field(
        ..., description="対応するインデックス文字（A, B, C...）。"
    )
    confidence: float = Field(..., description="最高確率（0〜1）。")
    entropy: float = Field(..., description="エントロピー（低いほど確信度が高い）。")
    distribution: dict[str, float] = Field(..., description="全選択肢の確率分布。")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPIアプリケーションのライフサイクルイベント。
    起動時にLogitRouterをロードし、終了時にクリーンアップする。
    """
    model_id = os.environ.get("LOGIT_ROUTER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    device = os.environ.get("LOGIT_ROUTER_DEVICE", None)

    try:
        router = LogitRouter(model_id=model_id, device=device)
        app.state.router = router
    except Exception as e:
        raise RuntimeError(f"Failed to load LogitRouter: {e}")

    yield

    # Clean up (if necessary)
    app.state.router = None


app = FastAPI(
    title="Logit Router API",
    description=(
        "Ultra-low latency LLM-based routing via "
        "single forward pass logit extraction"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/route", response_model=RouteResponseModel)
async def route(request: RouteRequestModel, req: Request):
    """
    リクエストを受け取り、LogitRouterを使って最適な選択肢を返すエンドポイント。
    """
    router: LogitRouter = req.app.state.router
    try:
        result = router.route(
            context=request.context,
            instruction=request.instruction,
            choices=request.choices,
            temperature=request.temperature,
        )
        return RouteResponseModel(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.get("/health")
async def health(req: Request):
    """
    サーバーのヘルスチェックエンドポイント。
    """
    router: LogitRouter = req.app.state.router
    model_name_or_path = getattr(router.model.config, "_name_or_path", "unknown")
    return {"status": "ok", "model": model_name_or_path}


def main():
    """
    コマンドラインから直接起動するためのエントリポイント。
    """
    parser = argparse.ArgumentParser(description="Run the Logit Router FastAPI server.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load the model on (e.g., 'cpu', 'cuda').",
    )

    args = parser.parse_args()

    # Set environment variables so the lifespan event picks them up
    os.environ["LOGIT_ROUTER_MODEL"] = args.model
    if args.device:
        os.environ["LOGIT_ROUTER_DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
