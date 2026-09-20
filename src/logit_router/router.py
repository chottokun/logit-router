import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitRouter:
    def __init__(
        self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device=None, max_choices=10
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_choices = max_choices
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=self.device,
                attn_implementation="flash_attention_2",
            )
        except (ImportError, Exception):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=self.device,
                attn_implementation="sdpa",
            )

        self.model.eval()
        self.backbone = self.model.model

        self.choice_letters = [chr(ord("A") + i) for i in range(max_choices)]
        self.choice_token_ids = [
            self.tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
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
