import argparse
import math
import time

import torch
import torch.nn.functional as F

from logit_router.router import LogitRouter


class Timer:
    def __init__(self, device):
        self.device = device
        self.is_cuda = (
            device != "cpu" and torch.cuda.is_available() and "cuda" in str(device)
        )
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        if self.is_cuda:
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def stop(self):
        if self.is_cuda:
            self.end_event.record()
        else:
            self.end_time = time.perf_counter()

    def elapsed_ms(self):
        if self.is_cuda:
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (self.end_time - self.start_time) * 1000.0


class ProfiledLogitRouter(LogitRouter):
    def __init__(
        self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device=None, max_choices=10
    ):
        super().__init__(model_id=model_id, device=device, max_choices=max_choices)
        self.timings = {}

    @torch.inference_mode()
    def route_with_profile(
        self,
        context: str,
        instruction: str,
        choices: list[str],
        temperature: float = 1.0,
    ) -> dict:
        num_choices = len(choices)
        if num_choices > self.max_choices:
            raise ValueError(
                f"Number of choices ({num_choices}) exceeds "
                f"max_choices ({self.max_choices})"
            )

        formatted_choices = "\n".join(
            [f"{self.choice_letters[i]}. {choice}" for i, choice in enumerate(choices)]
        )

        prompt = f"""<|im_start|>system
You are a fast routing engine. Select the single best choice based strictly on the \
context.<|im_end|>
<|im_start|>user
Context: {context}

Task: {instruction}

Choices:
{formatted_choices}

Select the single correct option letter.<|im_end|>
<|im_start|>assistant
Answer: """

        timer1 = Timer(self.device)
        timer2 = Timer(self.device)
        timer3 = Timer(self.device)
        timer4 = Timer(self.device)

        # 1. Tokenization & Tensor Transfer (CPU -> GPU)
        timer1.start()
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        timer1.stop()

        # 2. Backbone Forward Pass (Prefill, use_cache=False)
        timer2.start()
        outputs = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        timer2.stop()

        # 3. Sliced LM-Head MatMul
        timer3.start()
        active_head_weights = self.choice_head_weights[:num_choices]
        logits = torch.matmul(last_hidden_state, active_head_weights.t())
        timer3.stop()

        # 4. Softmax, Entropy, and Post-processing
        timer4.start()
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

        entropy = -sum(p * math.log(p + 1e-9) for p in probs)

        best_idx = probs.index(max(probs))
        best_letter = self.choice_letters[best_idx]
        best_choice = choices[best_idx]
        confidence = probs[best_idx]

        distribution = {self.choice_letters[i]: probs[i] for i in range(num_choices)}
        timer4.stop()

        self.timings = {
            "Tokenization & Tensor Transfer": timer1.elapsed_ms(),
            "Backbone Forward Pass": timer2.elapsed_ms(),
            "Sliced LM-Head MatMul": timer3.elapsed_ms(),
            "Softmax, Entropy & Post-proc": timer4.elapsed_ms(),
        }

        return {
            "best_choice": best_choice,
            "best_letter": best_letter,
            "confidence": confidence,
            "entropy": entropy,
            "distribution": distribution,
        }


def main():
    parser = argparse.ArgumentParser(description="Profile LogitRouter phases")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of profiling iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")

    args = parser.parse_args()

    print(f"Loading model {args.model} on {args.device or 'auto'}...")
    router = ProfiledLogitRouter(model_id=args.model, device=args.device)

    context = "Stripe webhook failed with status code 403. Invalid API secret key."
    instruction = "担当チームにトリアージしてください。"
    choices = ["決済・請求窓口", "インフラ保守", "一般サポート"]

    print(f"Starting warmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        router.route_with_profile(context, instruction, choices)
        if (
            router.device != "cpu"
            and torch.cuda.is_available()
            and "cuda" in str(router.device)
        ):
            torch.cuda.synchronize()

    print(f"Starting profiling ({args.iterations} iterations)...")
    phase_names = [
        "Tokenization & Tensor Transfer",
        "Backbone Forward Pass",
        "Sliced LM-Head MatMul",
        "Softmax, Entropy & Post-proc",
    ]

    results = {name: [] for name in phase_names}

    for _ in range(args.iterations):
        router.route_with_profile(context, instruction, choices)
        if (
            router.device != "cpu"
            and torch.cuda.is_available()
            and "cuda" in str(router.device)
        ):
            torch.cuda.synchronize()
        for name in phase_names:
            results[name].append(router.timings[name])

    avg_times = {name: sum(times) / len(times) for name, times in results.items()}
    min_times = {name: min(times) for name, times in results.items()}
    total_avg_time = sum(avg_times.values())

    print("\n" + "=" * 85)
    print(
        f"| {'Phase':<35} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'% of Total':<10} |"
    )
    print("|" + "-" * 37 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    for name in phase_names:
        avg = avg_times[name]
        min_val = min_times[name]
        pct = (avg / total_avg_time) * 100 if total_avg_time > 0 else 0
        print(f"| {name:<35} | {avg:>10.2f} | {min_val:>10.2f} | {pct:>9.2f}% |")
    print("=" * 85)
    print(f"Total Average Latency: {total_avg_time:.2f} ms\n")

    bottleneck_phase = max(avg_times, key=avg_times.get)
    bottleneck_pct = (
        (avg_times[bottleneck_phase] / total_avg_time) * 100
        if total_avg_time > 0
        else 0
    )

    print("【Analysis】")
    print(f"全体的な平均推論レイテンシは {total_avg_time:.2f} ms です。")
    print(
        f"最も時間がかかっているボトルネックは '{bottleneck_phase}' で、"
        f"全体の {bottleneck_pct:.2f}% を占めています。"
    )

    if bottleneck_phase == "Backbone Forward Pass":
        print(
            "これは LLM の深層ネットワーク（Transformer Block等）を通るための"
            "計算コストが支配的であることを示しています。"
        )
    elif bottleneck_phase == "Tokenization & Tensor Transfer":
        print("トークナイズおよびCPUからGPUへのテンソル転送に時間がかかっています。")
    elif bottleneck_phase == "Sliced LM-Head MatMul":
        print("LM-Headの行列積の計算に時間がかかっています。")
    else:
        print("ソフトマックスやエントロピー計算、後処理の計算に時間がかかっています。")


if __name__ == "__main__":
    main()
