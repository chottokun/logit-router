from dataclasses import dataclass


@dataclass
class RouteRequest:
    """
    ルーターへのリクエストを表現するデータクラス。

    Attributes:
        context: ルーティングの判断基準となるコンテキスト。
        instruction: 実行すべき指示。
        choices: 選択肢のリスト。
        temperature: 生成時の温度パラメータ（デフォルト値は1.0）。
    """

    context: str
    instruction: str
    choices: list[str]
    temperature: float = 1.0


@dataclass
class RouteResult:
    """
    ルーティングの結果を表現するデータクラス。

    Attributes:
        best_choice: 最適な選択肢テキスト。
        best_letter: 対応するインデックス文字（A, B, C...）。
        confidence: 最高確率（0〜1）。
        entropy: エントロピー（低いほど確信度が高い）。
        distribution: 全選択肢の確率分布。
    """

    best_choice: str
    best_letter: str
    confidence: float
    entropy: float
    distribution: dict[str, float]

    def is_confident(self, threshold: float = 0.7) -> bool:
        """confidence が threshold 以上なら True を返す"""
        return self.confidence >= threshold

    def needs_fallback(self, entropy_threshold: float = 0.9) -> bool:
        """
        entropy が entropy_threshold を超えたら True を返す（判定に迷っている状態）
        """
        return self.entropy > entropy_threshold
