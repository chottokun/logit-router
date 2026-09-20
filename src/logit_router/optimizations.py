import torch

from logit_router.router import LogitRouter
from logit_router.schema import RouteResult


def apply_torch_compile(
    router: LogitRouter, mode: str = "reduce-overhead"
) -> LogitRouter:
    """
    ルーターのバックボーンモデルに torch.compile を適用します。

    Args:
        router (LogitRouter): コンパイル対象のルーターインスタンス。
        mode (str): torch.compile の最適化モード。デフォルトは 'reduce-overhead'。

    Returns:
        LogitRouter: コンパイル済みのバックボーンを持つルーターインスタンス自身。
    """
    router.backbone = torch.compile(router.backbone, mode=mode, fullgraph=False)
    return router


class FallbackRouter:
    """
    自信がない（エントロピーが高い、または上位の確率差が小さい）場合にフォールバック関数を実行するルーター。
    """

    def __init__(
        self,
        router: LogitRouter,
        entropy_threshold: float = 0.9,
        margin_threshold: float = 0.1,
        fallback_fn: callable | None = None,
    ):
        """
        FallbackRouter を初期化します。

        Args:
            router (LogitRouter): 基本となるルーターインスタンス。
            entropy_threshold (float): エントロピーの閾値。
            これより大きいと「迷っている」と判定されます。
            margin_threshold (float): 上位2つの確率の差の閾値。
            これより小さいと「迷っている」と判定されます。
            fallback_fn (callable | None): 迷っている場合に
            実行されるフォールバック関数。
        """
        self.router = router
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.fallback_fn = fallback_fn

    def route(
        self,
        context: str,
        instruction: str,
        choices: list[str],
        temperature: float = 1.0,
    ) -> dict | RouteResult:
        """
        ルーティングを実行し、必要に応じてフォールバックを呼び出します。

        Args:
            context (str): コンテキスト。
            instruction (str): 指示。
            choices (list[str]): 選択肢のリスト。
            temperature (float): 温度パラメータ。

        Returns:
            dict: ルーティング結果（またはフォールバック関数の結果）。
        """
        result = self.router.route(
            context=context,
            instruction=instruction,
            choices=choices,
            temperature=temperature,
        )

        if isinstance(result, dict):
            entropy = result.get("entropy", 0.0)
            distribution = result.get("distribution", {})
        else:
            entropy = getattr(result, "entropy", 0.0)
            distribution = getattr(result, "distribution", {})

        probs = sorted(distribution.values(), reverse=True)

        margin = 1.0
        if len(probs) >= 2:
            margin = probs[0] - probs[1]

        is_uncertain = (
            entropy > self.entropy_threshold or margin < self.margin_threshold
        )

        if is_uncertain and self.fallback_fn is not None:
            fallback_result = self.fallback_fn(context, instruction, choices)
            if isinstance(fallback_result, dict):
                fallback_result["fallback_executed"] = True
                return fallback_result
            return {"fallback_executed": True, "result": fallback_result}

        if isinstance(result, dict):
            result["fallback_executed"] = False
            return result

        # If it's a RouteResult or something else, return it as dict
        if hasattr(result, "__dict__"):
            out = dict(result.__dict__)
            out["fallback_executed"] = False
            return out

        return {"fallback_executed": False, "result": result}

    def classify(
        self,
        context: str,
        instruction: str,
        choices: list[str],
        temperature: float = 1.0,
    ) -> dict | RouteResult:
        """
        route メソッドのエイリアスです。
        """
        return self.route(
            context=context,
            instruction=instruction,
            choices=choices,
            temperature=temperature,
        )
