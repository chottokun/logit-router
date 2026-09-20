import argparse
import time

import torch

from logit_router.optimizations import apply_torch_compile
from logit_router.router import LogitRouter


def generate_context(tokenizer, target_length: int) -> str:
    """
    指定されたトークン数に一致するダミーコンテキストを生成する。
    """
    base_text = "This is a dummy context for benchmarking purpose. "
    tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if not tokens:
        tokens = tokenizer.encode("dummy ", add_special_tokens=False)

    repeats = (target_length // len(tokens)) + 1
    extended_tokens = (tokens * repeats)[:target_length]
    return tokenizer.decode(extended_tokens)


def measure_latency(router, context, instruction, choices, iterations, device):
    # Warmup
    for _ in range(3):
        _ = router.route(context=context, instruction=instruction, choices=choices)

    if "cuda" in str(device):
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iterations):
        if "cuda" in str(device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = router.route(context=context, instruction=instruction, choices=choices)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        else:
            start_time = time.perf_counter()
            _ = router.route(context=context, instruction=instruction, choices=choices)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0)

    return sum(latencies) / len(latencies)


def main():
    parser = argparse.ArgumentParser(
        description="Context length scaling benchmark for LogitRouter"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name"
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="64,256,512,1024",
        help="Comma separated context token lengths",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of measurement iterations"
    )
    parser.add_argument(
        "--with-compile",
        action="store_true",
        help="Include torch.compile mode in measurement",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="'cuda' or 'cpu'",
    )

    args = parser.parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",")]

    print(f"Initializing LogitRouter with model={args.model}, device={args.device}")
    router = LogitRouter(model_id=args.model, device=args.device)
    device = router.device
    print(f"Using device: {device}")

    instruction = (
        "以下のコンテキストに基づいて、最も適切なオプションを選択してください。"
    )
    choices = ["オプションA", "オプションB", "オプションC"]

    print("\nStarting Eager mode measurements...")
    eager_latencies = {}
    for length in lengths:
        print(f"Generating context for length {length}...")
        context = generate_context(router.tokenizer, length)
        print(f"Measuring Eager mode for length {length}...")
        avg_latency = measure_latency(
            router, context, instruction, choices, args.iterations, device
        )
        eager_latencies[length] = avg_latency

    compiled_latencies = {}
    if args.with_compile:
        print("\nApplying torch.compile...")
        router = apply_torch_compile(router)
        print("Starting Compiled mode measurements...")
        for length in lengths:
            context = generate_context(router.tokenizer, length)
            print(f"Measuring Compiled mode for length {length}...")
            avg_latency = measure_latency(
                router, context, instruction, choices, args.iterations, device
            )
            compiled_latencies[length] = avg_latency

    print("\n--- Benchmark Results ---")
    print("| Context Length (tokens) | Eager (ms) | Compiled (ms) | Speedup |")
    print("|---|---|---|---|")
    for length in lengths:
        e_lat = eager_latencies[length]
        if args.with_compile:
            c_lat = compiled_latencies[length]
            speedup = e_lat / c_lat if c_lat > 0 else 0
            print(f"| {length} | {e_lat:.2f} | {c_lat:.2f} | {speedup:.2f}x |")
        else:
            print(f"| {length} | {e_lat:.2f} | N/A | N/A |")

    print("\n--- Summary of Scaling Trend ---")
    print(
        "コンテキスト長が増加するにつれて、Prefill の計算量が増加し、"
        "レイテンシが増大します。"
    )
    print(
        "一般に、Transformer の自己注意機構 (Self-Attention) は系列長 N に対して "
        "O(N^2) の計算量を持ちますが、"
    )
    print("短〜中程度のコンテキストではメモリアクセス律速の影響も大きく見られます。")
    if args.with_compile:
        print(
            "torch.compile を適用することで、カーネル融合やオーバーヘッドの削減が働き、"
            "特に短い系列や"
        )
        print("特定のバッチサイズにおいてレイテンシ（Speedup）の改善が期待されます。")


if __name__ == "__main__":
    main()
