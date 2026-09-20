import argparse
import itertools
import random
import statistics

from logit_router.router import LogitRouter
from logit_router.schema import RouteResult


def main():
    parser = argparse.ArgumentParser(description="Test robustness of LogitRouter")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (e.g. cuda, cpu)"
    )
    args = parser.parse_args()

    print(f"Loading model {args.model} on {args.device or 'auto'}...")
    router = LogitRouter(model_id=args.model, device=args.device)

    print("\n" + "=" * 50)
    print("Test 1: Position Bias (Permutation Test)")
    print("=" * 50)

    # Case 1
    print("\n--- Case 1: Customer Support (3 choices, all permutations) ---")
    context1 = "I was charged twice for my subscription this month. Please help."
    instruction1 = "Classify the user's request."
    choices1 = ["Billing & Payments", "Technical Support", "Sales & Upgrades"]
    target_choice1 = "Billing & Payments"

    perms1 = list(itertools.permutations(choices1))

    correct_count1 = 0
    entropies1 = []

    for perm in perms1:
        res_dict = router.route(
            context=context1, instruction=instruction1, choices=list(perm)
        )
        res = RouteResult(**res_dict)
        if res.best_choice == target_choice1:
            correct_count1 += 1
        entropies1.append(res.entropy)

    consistency1 = (correct_count1 / len(perms1)) * 100
    entropy_range1 = max(entropies1) - min(entropies1)

    print(f"Consistency Rate: {consistency1:.2f}% ({correct_count1}/{len(perms1)})")
    print(
        f"Entropy Range: {min(entropies1):.4f} - {max(entropies1):.4f} (Delta: {entropy_range1:.4f})"
    )

    # Case 2
    print("\n--- Case 2: Tool Routing (4 choices, 12 random permutations) ---")
    context2 = "Calculate the square root of 144."
    instruction2 = "Select the best tool."
    choices2 = ["Calculator", "Web Search", "Database Query", "Weather API"]
    target_choice2 = "Calculator"

    perms2 = list(itertools.permutations(choices2))
    random.seed(42)
    selected_perms2 = random.sample(perms2, 12)

    correct_count2 = 0
    entropies2 = []

    for perm in selected_perms2:
        res_dict = router.route(
            context=context2, instruction=instruction2, choices=list(perm)
        )
        res = RouteResult(**res_dict)
        if res.best_choice == target_choice2:
            correct_count2 += 1
        entropies2.append(res.entropy)

    consistency2 = (correct_count2 / len(selected_perms2)) * 100
    entropy_range2 = max(entropies2) - min(entropies2)

    print(
        f"Consistency Rate: {consistency2:.2f}% ({correct_count2}/{len(selected_perms2)})"
    )
    print(
        f"Entropy Range: {min(entropies2):.4f} - {max(entropies2):.4f} (Delta: {entropy_range2:.4f})"
    )

    print("\n" + "=" * 50)
    print("Test 2: Out-of-Domain (OOD) Rejection Test")
    print("=" * 50)

    instruction3 = "Classify the input into one of the following domains."
    choices3 = ["Technology News", "Sports Updates", "Financial Markets"]

    normal_contexts = [
        "Apple just released the new iPhone with an upgraded M4 chip.",
        "The stock market saw a huge dip today after interest rate hikes.",
        "Manchester United won their match 3-1 last night.",
    ]

    ood_contexts = [
        "吾輩は猫である。名前はまだ無い。",
        "明日の東京の天気を教えて",
        "Pythonでクイックソートを実装して",
    ]

    normal_entropies = []
    for ctx in normal_contexts:
        res_dict = router.route(context=ctx, instruction=instruction3, choices=choices3)
        res = RouteResult(**res_dict)
        normal_entropies.append(res.entropy)

    ood_entropies = []
    ood_rejected = 0
    for ctx in ood_contexts:
        res_dict = router.route(context=ctx, instruction=instruction3, choices=choices3)
        res = RouteResult(**res_dict)
        ood_entropies.append(res.entropy)
        if res.needs_fallback(entropy_threshold=0.9):
            ood_rejected += 1

    avg_normal_entropy = statistics.mean(normal_entropies)
    avg_ood_entropy = statistics.mean(ood_entropies)
    ood_rejection_rate = (ood_rejected / len(ood_contexts)) * 100

    print(f"Normal Input Avg Entropy: {avg_normal_entropy:.4f}")
    print(f"OOD Input Avg Entropy:    {avg_ood_entropy:.4f}")
    print(
        f"OOD Rejection Rate:       {ood_rejection_rate:.2f}% ({ood_rejected}/{len(ood_contexts)})"
    )

    print("\n" + "=" * 50)
    print("Report Summary")
    print("=" * 50)

    print(f"{'Metric':<30} | {'Value':<20} | {'Status'}")
    print("-" * 65)

    def status(condition):
        return "PASS" if condition else "NEEDS IMPROVEMENT"

    print(
        f"{'Case 1 Consistency':<30} | {consistency1:>19.2f}% | {status(consistency1 == 100)}"
    )
    print(
        f"{'Case 2 Consistency':<30} | {consistency2:>19.2f}% | {status(consistency2 >= 90)}"
    )
    print(
        f"{'OOD Rejection Rate':<30} | {ood_rejection_rate:>19.2f}% | {status(ood_rejection_rate >= 66.6)}"
    )
    print(
        f"{'Entropy Margin (OOD - Normal)':<30} | {avg_ood_entropy - avg_normal_entropy:>19.4f} | {status((avg_ood_entropy - avg_normal_entropy) > 0.2)}"
    )
    print("-" * 65)

    all_passed = (
        (consistency1 == 100)
        and (consistency2 >= 90)
        and (ood_rejection_rate >= 66.6)
        and ((avg_ood_entropy - avg_normal_entropy) > 0.2)
    )
    print(f"Overall Result: {'PASSED' if all_passed else 'NEEDS IMPROVEMENT'}")


if __name__ == "__main__":
    main()
