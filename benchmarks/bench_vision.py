import argparse
import json
import time

import torch
from PIL import Image

from logit_router import LogitRouter
from logit_router.vision_router import VisionLogitRouter


def create_synthetic_image(size=(224, 224), color="red") -> Image.Image:
    return Image.new("RGB", size, color=color)


def main():
    parser = argparse.ArgumentParser(
        description="Latency benchmark for VisionLogitRouter"
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-4-E2B-it", help="Model name"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with text LogitRouter"
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Text model name for comparison",
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

    print(
        f"Initializing VisionLogitRouter with model={args.model}, device={args.device}"
    )
    vision_router = VisionLogitRouter(model_id=args.model, device=args.device)
    device = vision_router.device
    print(f"Using device: {device}")

    text_router = None
    if args.compare:
        print(
            f"Initializing Text LogitRouter with model={args.text_model}, device={args.device}"
        )
        text_router = LogitRouter(model_id=args.text_model, device=args.device)

    context = "Web page shows an error overlay."
    instruction = "What is the primary color of the block in the image?"
    choices = ["Red", "Green", "Blue"]

    synthetic_image = create_synthetic_image(color="red")

    print(f"Running {args.warmup} warm-up iterations for VisionLogitRouter...")
    for _ in range(args.warmup):
        _ = vision_router.route(
            image=synthetic_image,
            context=context,
            instruction=instruction,
            choices=choices,
        )

    if args.compare:
        print(f"Running {args.warmup} warm-up iterations for Text LogitRouter...")
        for _ in range(args.warmup):
            _ = text_router.route(
                context=context, instruction=instruction, choices=choices
            )

    if device == "cuda":
        torch.cuda.synchronize()

    def run_benchmark(router, is_vision=True):
        latencies = []
        preprocess_times = []
        forward_times = []
        postprocess_times = []

        print(
            f"Running {args.iterations} measurement iterations for {'Vision' if is_vision else 'Text'} Router..."
        )

        result = None
        for _ in range(args.iterations):
            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                if is_vision:
                    result = router.route(
                        image=synthetic_image,
                        context=context,
                        instruction=instruction,
                        choices=choices,
                    )
                else:
                    result = router.route(
                        context=context, instruction=instruction, choices=choices
                    )

                end_event.record()
                torch.cuda.synchronize()
                latency_ms = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()

                if is_vision:
                    result = router.route(
                        image=synthetic_image,
                        context=context,
                        instruction=instruction,
                        choices=choices,
                    )
                else:
                    result = router.route(
                        context=context, instruction=instruction, choices=choices
                    )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000.0

            latencies.append(latency_ms)

            if is_vision:
                preprocess_times.append(router.preprocess_time_ms)
                forward_times.append(router.forward_time_ms)
                postprocess_times.append(router.postprocess_time_ms)

        avg_latency = sum(latencies) / len(latencies)

        print(f"\n--- {'Vision' if is_vision else 'Text'} Benchmark Results ---")
        print(f"Total Average Latency: {avg_latency:.2f} ms")
        if is_vision:
            avg_prep = sum(preprocess_times) / len(preprocess_times)
            avg_fwd = sum(forward_times) / len(forward_times)
            avg_post = sum(postprocess_times) / len(postprocess_times)
            print(f"  Preprocess Latency:  {avg_prep:.2f} ms")
            print(f"  Forward Latency:     {avg_fwd:.2f} ms")
            print(f"  Postprocess Latency: {avg_post:.2f} ms")

        # Handle dataclass safely (RouteResult vs dict)
        if hasattr(result, "__dict__"):
            result_dict = result.__dict__
        else:
            result_dict = result

        print("\n--- Final Routing Result ---")
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    run_benchmark(vision_router, is_vision=True)
    if args.compare:
        run_benchmark(text_router, is_vision=False)


if __name__ == "__main__":
    main()
