import argparse
import json
import time

import torch

from logit_router import LogitRouter


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for LogitRouter")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of measurement iterations"
    )
    parser.add_argument("--warmup", type=int, default=3, help="Number of warm-up runs")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="'cuda' or 'cpu'",
    )

    args = parser.parse_args()

    print(f"Initializing LogitRouter with model={args.model}, device={args.device}")
    router = LogitRouter(model_id=args.model, device=args.device)
    device = router.device
    print(f"Using device: {device}")

    context = "Stripe webhook failed with status code 403. Invalid API secret key."
    instruction = "担当チームにトリアージしてください。"
    choices = ["決済・請求窓口", "インフラ保守", "一般サポート"]

    print(f"Running {args.warmup} warm-up iterations...")
    for _ in range(args.warmup):
        _ = router.route(context=context, instruction=instruction, choices=choices)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    print(f"Running {args.iterations} measurement iterations...")

    result = None
    for _ in range(args.iterations):
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = router.route(
                context=context, instruction=instruction, choices=choices
            )
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            start_time = time.perf_counter()
            result = router.route(
                context=context, instruction=instruction, choices=choices
            )
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0

        latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print("\n--- Benchmark Results ---")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Minimum Latency: {min_latency:.2f} ms")
    print(f"Maximum Latency: {max_latency:.2f} ms")

    print("\n--- Final Routing Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
