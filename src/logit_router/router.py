import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitRouter:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        device=None,
        max_choices=10,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str | dict | None = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_choices = max_choices
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        dtype = (
            "auto"
            if (load_in_4bit or load_in_8bit)
            else (torch.bfloat16 if self.device == "cuda" else torch.float32)
        )

        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )
            except ImportError:
                import warnings

                warnings.warn(
                    "bitsandbytes is not installed. Quantization will be disabled."
                )

        final_device_map = device_map if device_map is not None else self.device

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": final_device_map,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except (ImportError, Exception):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                **model_kwargs,
            )

        self.model.eval()
        self.backbone = self.model.model

        self.choice_letters = [chr(ord("A") + i) for i in range(max_choices)]
        self.choice_token_ids = [
            self.tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]
            for letter in self.choice_letters
        ]

        choice_token_tensor = torch.tensor(self.choice_token_ids, device=self.device)
        self.choice_head_weights = (
            self.model.lm_head.weight[choice_token_tensor].detach().clone()
        )

    @torch.inference_mode()
    def route(
        self,
        context: str,
        instruction: str,
        choices: list[str],
        temperature: float = 1.0,
    ) -> dict:
        num_choices = len(choices)
        if num_choices > self.max_choices:
            raise ValueError(
                f"Number of choices ({num_choices}) exceeds max_choices "
                f"({self.max_choices})"
            )

        formatted_choices = "\n".join(
            [f"{self.choice_letters[i]}. {choice}" for i, choice in enumerate(choices)]
        )

        messages = [
            {
                "role": "system",
                "content": "You are a fast routing engine. Select the single best choice based strictly on the context.",
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nTask: {instruction}\n\nChoices:\n{formatted_choices}\n\nSelect the single correct option letter.",
            },
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Ensure "Answer:" is at the end or proper generation space is left.
            if not prompt.endswith("Answer: "):
                prompt += "Answer: "
        except Exception:
            # Fallback if tokenizer lacks a chat template
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

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        outputs = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        # Calculate logits for the choices
        # head weights: (max_choices, hidden_size)
        # last_hidden_state: (batch_size, hidden_size)
        # logits: (batch_size, num_choices)
        active_head_weights = self.choice_head_weights[:num_choices]
        logits = torch.matmul(last_hidden_state, active_head_weights.t())

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

        entropy = -sum(p * math.log(p + 1e-9) for p in probs)

        best_idx = probs.index(max(probs))
        best_letter = self.choice_letters[best_idx]
        best_choice = choices[best_idx]
        confidence = probs[best_idx]

        distribution = {self.choice_letters[i]: probs[i] for i in range(num_choices)}

        return {
            "best_choice": best_choice,
            "best_letter": best_letter,
            "confidence": confidence,
            "entropy": entropy,
            "distribution": distribution,
        }

    def classify(
        self,
        context: str,
        instruction: str,
        choices: list[str],
        temperature: float = 1.0,
    ) -> dict:
        return self.route(
            context=context,
            instruction=instruction,
            choices=choices,
            temperature=temperature,
        )
