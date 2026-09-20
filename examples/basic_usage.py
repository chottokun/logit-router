import json

from logit_router import LogitRouter
from logit_router.schema import RouteResult


def main():
    print("Initializing LogitRouter...")
    router = LogitRouter()

    print("\n" + "=" * 50)
    print("ユースケース A: カスタマーサポートの動的トリアージ")
    print("=" * 50)

    context_a = (
        "クレジットカードの有効期限が切れたため更新したいのですが、"
        "管理画面で保存ボタンを押しても反映されません。"
    )
    instruction_a = "この問い合わせを最も適切な対応チームに振り分けてください。"
    choices_a = ["決済・請求窓口", "システム障害対応", "一般的な操作案内"]

    print(f"Context: {context_a}")
    print(f"Instruction: {instruction_a}")
    print(f"Choices: {choices_a}\n")

    # 実行
    raw_result_a = router.route(
        context=context_a, instruction=instruction_a, choices=choices_a
    )
    result_a = RouteResult(**raw_result_a)

    # JSON形式で出力
    print("結果:")
    print(json.dumps(raw_result_a, indent=2, ensure_ascii=False))

    # 確信度とフォールバックの判定
    print("\n確信度判定:")
    if result_a.is_confident(threshold=0.7):
        print(
            "✅ 高い確信度でルーティングされました "
            f"(confidence: {result_a.confidence:.4f})"
        )
    else:
        print(f"⚠️ 確信度が低いです (confidence: {result_a.confidence:.4f})")

    if result_a.needs_fallback(entropy_threshold=0.9):
        print(
            "⚠️ エントロピーが高いため、"
            "人間のオペレーターへのフォールバックを推奨します "
            f"(entropy: {result_a.entropy:.4f})"
        )
    else:
        print(f"✅ 自動ルーティングを継続します (entropy: {result_a.entropy:.4f})")

    print("\n" + "=" * 50)
    print("ユースケース B: エージェントの動的ツールルーティング")
    print("=" * 50)

    context_b = "ユーザー発言: 先週の売上合計を日別にグラフ化して社内Slackに投げて"
    instruction_b = "次に呼び出すべき最適なエージェントツールを1つ選択してください。"
    choices_b = [
        "Database_SQL_Runner",
        "Slack_Notification_Tool",
        "Chart_Generator",
        "Web_Search",
    ]

    print(f"Context: {context_b}")
    print(f"Instruction: {instruction_b}")
    print(f"Choices: {choices_b}\n")

    # 実行
    raw_result_b = router.route(
        context=context_b, instruction=instruction_b, choices=choices_b
    )
    result_b = RouteResult(**raw_result_b)

    # JSON形式で出力
    print("結果:")
    print(json.dumps(raw_result_b, indent=2, ensure_ascii=False))

    # 確信度とフォールバックの判定
    print("\n確信度判定:")
    if result_b.is_confident(threshold=0.7):
        print(
            "✅ 高い確信度でツールが選択されました "
            f"(confidence: {result_b.confidence:.4f})"
        )
    else:
        print(f"⚠️ 確信度が低いです (confidence: {result_b.confidence:.4f})")

    if result_b.needs_fallback(entropy_threshold=0.9):
        print(
            "⚠️ 複数ツールで迷っている可能性があるため、ユーザーに確認します "
            f"(entropy: {result_b.entropy:.4f})"
        )
    else:
        print(f"✅ 自動ツール実行を継続します (entropy: {result_b.entropy:.4f})")


if __name__ == "__main__":
    main()
