"""
Comparative Benchmark: LogitRouter vs Standard Autoregressive Generation.
Directly measures the latency speedup of single forward pass logit extraction
against standard Hugging Face model.generate() on the same hardware and inputs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_router.router import LogitRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_autoregressive_vs_logit_router(
    model_id: str,
    test_cases: list[dict[str, Any]],
    warmup_runs: int = 3,
) -> dict[str, Any]:
    logger.info(f"Loading model {model_id} for comparative A/B test...")
    torch.cuda.empty_cache()

    # 1. Initialize Logit Router
    router = LogitRouter(model_id=model_id, device="cuda")

    # 2. Initialize Standard Model & Tokenizer for model.generate()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    raw_model.eval()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Prepare standard prompt template for generation
    def build_gen_prompt(case: dict[str, Any]) -> str:
        choices_text = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(case["choices"])])
        messages = [
            {"role": "system", "content": "You are a routing engine. Answer with only the option letter."},
            {"role": "user", "content": f"Context: {case['context']}\nTask: {case['instruction']}\nChoices:\n{choices_text}\nAnswer with only the single letter."},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Warmup
    logger.info("Warming up both engines...")
    dummy_case = test_cases[0]
    dummy_prompt = build_gen_prompt(dummy_case)
    dummy_inputs = tokenizer(dummy_prompt, return_tensors="pt").to("cuda")

    for _ in range(warmup_runs):
        router.route(dummy_case["context"], dummy_case["instruction"], dummy_case["choices"])
        with torch.inference_mode():
            raw_model.generate(**dummy_inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    logit_latencies: list[float] = []
    gen_latencies_short: list[float] = []  # max_new_tokens=5 (minimal answer like "A")
    gen_latencies_cot: list[float] = []    # max_new_tokens=30 (brief reasoning/word answer)
    generated_texts: list[str] = []

    logger.info(f"Measuring {len(test_cases)} cases...")

    for i, case in enumerate(test_cases):
        # --- Measure LogitRouter ---
        start_event.record()
        _res = router.route(case["context"], case["instruction"], case["choices"])
        end_event.record()
        torch.cuda.synchronize()
        logit_latencies.append(start_event.elapsed_time(end_event))

        # --- Measure Standard Generation (Minimal 5 tokens) ---
        prompt = build_gen_prompt(case)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        start_event.record()
        with torch.inference_mode():
            out = raw_model.generate(**inputs, max_new_tokens=5, do_sample=False)
        end_event.record()
        torch.cuda.synchronize()
        gen_latencies_short.append(start_event.elapsed_time(end_event))

        # Decode generated text
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated_texts.append(text)

        # --- Measure Standard Generation with brief reasoning (30 tokens) ---
        start_event.record()
        with torch.inference_mode():
            raw_model.generate(**inputs, max_new_tokens=30, do_sample=False)
        end_event.record()
        torch.cuda.synchronize()
        gen_latencies_cot.append(start_event.elapsed_time(end_event))

    del router
    del raw_model
    torch.cuda.empty_cache()

    l_arr = np.array(logit_latencies)
    g_arr = np.array(gen_latencies_short)
    cot_arr = np.array(gen_latencies_cot)

    mean_logit = float(np.mean(l_arr))
    mean_gen = float(np.mean(g_arr))
    mean_cot = float(np.mean(cot_arr))

    return {
        "model_id": model_id,
        "sample_count": len(test_cases),
        "logit_router_mean_ms": round(mean_logit, 2),
        "logit_router_p50_ms": round(float(np.percentile(l_arr, 50)), 2),
        "logit_router_p95_ms": round(float(np.percentile(l_arr, 95)), 2),
        "standard_gen_mean_ms": round(mean_gen, 2),
        "standard_gen_p50_ms": round(float(np.percentile(g_arr, 50)), 2),
        "standard_gen_p95_ms": round(float(np.percentile(g_arr, 95)), 2),
        "cot_gen_mean_ms": round(mean_cot, 2),
        "cot_gen_p50_ms": round(float(np.percentile(cot_arr, 50)), 2),
        "speedup_vs_min_gen": round(mean_gen / mean_logit, 2),
        "speedup_vs_cot_gen": round(mean_cot / mean_logit, 2),
    }


def main():
    data_path = Path("benchmarks/data/eval_cases.json")
    with open(data_path, encoding="utf-8") as f:
        all_cases = json.load(f)

    # Use first 20 cases for rigorous comparison
    cases = all_cases[:20]

    models = [
        "google/gemma-4-E2B-it",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
    ]

    results = []
    for m in models:
        r = benchmark_autoregressive_vs_logit_router(m, cases)
        results.append(r)

    # Output report
    report_path = Path("benchmarks/reports/comparison_vs_generation_report.md")
    content = [
        "---",
        "type: metrics",
        "title: LogitRouter vs Standard Generation Benchmark Report / LogitRouter vs 通常生成 実機比較レポート",
        "description: Rigorous measured speedup of single forward pass logit extraction vs autoregressive generate() on RTX 3060 / RTX 3060での通常生成との実機比較実測値",
        "status: completed",
        "generated:",
        "  by: benchmarks/bench_vs_autoregressive.py",
        f'  at: "{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}"',
        "tags: [speedup, autoregressive, generation, comparison, on-device]",
        "sources:",
        "  - benchmarks/bench_vs_autoregressive.py",
        "---",
        "",
        "# LogitRouter vs Standard Generation Benchmark Report / 通常自己回帰生成との実機比較レポート",
        "",
        "**[English]**",
        "This report provides strictly measured empirical comparison between LogitRouter (single forward pass prefill) and standard Hugging Face `model.generate()` (autoregressive generation) using identical model weights, prompts, and hardware (NVIDIA GeForce RTX 3060 12GB).",
        "",
        "**[Japanese]**",
        "本レポートは、同一のモデル重み、プロンプト、およびハードウェア環境（NVIDIA GeForce RTX 3060 12GB）において、LogitRouter（単一フォワードパスPrefill）と標準の Hugging Face `model.generate()`（自己回帰逐次生成）を実行し、レイテンシと高速化倍率を直接対決（A/Bテスト）で実機実測した結果です。",
        "",
        "## Comparative Latency & Speedup Summary / 比較サマリー（実測値）",
        "",
        "| Model / モデル | LogitRouter (p50 / Mean) | 通常生成: 最短1文字 (p50 / Mean) | 通常生成: 簡潔推論 (p50 / Mean) | 高速化倍率 (vs 最短生成) | 高速化倍率 (vs 簡潔推論) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        content.append(
            f"| `{r['model_id']}` | **{r['logit_router_p50_ms']} ms** / {r['logit_router_mean_ms']} ms "
            f"| {r['standard_gen_p50_ms']} ms / {r['standard_gen_mean_ms']} ms "
            f"| {r['cot_gen_p50_ms']} ms / {r['cot_gen_mean_ms']} ms "
            f"| **{r['speedup_vs_min_gen']}x 高速** "
            f"| **{r['speedup_vs_cot_gen']}x 高速** |"
        )

    content.extend([
        "",
        "## Technical Analysis / 技術的要因の分析",
        "",
        "**[English]**",
        "- **Autoregressive Overhead**: `model.generate()` incurs multiple sequential kernel launches, KV cache memory allocations, and memory bandwidth contention for every additional token generated.",
        "- **Single Forward Pass Advantage**: LogitRouter halts immediately after the prompt prefill phase, reducing the execution to exactly 1 Transformer forward pass plus a sliced linear projection of candidate tokens.",
        "",
        "**[Japanese]**",
        "- **自己回帰生成のオーバーヘッド**: 通常の `model.generate()` では、1トークン生成するごとに逐次カーネル起動、KVキャッシュ確保、およびメモリストリーミング（メモリバウンド制約）が発生します。",
        "- **単一フォワードパスの優位性**: LogitRouter はプロンプトのPrefill完了直後に処理を完了し、全語彙（15万語）への射影を行わず選択肢トークンのみにスライスしてロジットを抽出するため、オーバーヘッドが極小化されます。",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(content), encoding="utf-8")
    Path("docs/references/comparison_vs_generation_report.md").write_text("\n".join(content), encoding="utf-8")

    logger.info(f"Comparison report written to {report_path}")


if __name__ == "__main__":
    main()
