import math
import time
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq


from logit_router.schema import RouteResult


class VisionLogitRouter:
    def __init__(
        self,
        model_id="google/gemma-4-E2B-it",
        device=None,
        max_choices=10,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Union[str, dict, None] = None,
        is_awq: bool = False,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_choices = max_choices
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        # The processor should contain a tokenizer
        self.tokenizer = self.processor.tokenizer

        is_awq_model = is_awq or "awq" in model_id.lower()
        is_gemma = "gemma" in model_id.lower()

        dtype = (
            "auto"
            if (load_in_4bit or load_in_8bit or is_awq_model)
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
            attn_impl = "sdpa" if is_gemma else "flash_attention_2"
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                **model_kwargs,
            )
        except ImportError, Exception:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                **model_kwargs,
            )

        self.model.eval()

        # We need to get the language model part to extract the backbone.
        # Typically AutoModelForVision2Seq has `language_model`
        # But we can also just use the model directly for forward pass to avoid architecture-specific logic.
        # Wait, how to do sliced head?
        # Usually Vision2Seq models have `language_model.lm_head` or `lm_head` directly.
        # Let's try `lm_head` first, then `language_model.lm_head`.
        if hasattr(self.model, "lm_head"):
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "lm_head"
        ):
            self.lm_head = self.model.language_model.lm_head
        else:
            # Fallback
            self.lm_head = None

        self.choice_letters = [chr(ord("A") + i) for i in range(max_choices)]

        if is_gemma:
            self.choice_token_ids = [
                self.tokenizer.encode(letter, add_special_tokens=False)[-1]
                for letter in self.choice_letters
            ]
        else:
            self.choice_token_ids = [
                self.tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]
                for letter in self.choice_letters
            ]

        choice_token_tensor = torch.tensor(self.choice_token_ids, device=self.device)
        self.is_sliced_head = False
        self.choice_head_weights = None

        if (
            self.lm_head is not None
            and hasattr(self.lm_head, "weight")
            and self.lm_head.weight is not None
        ):
            self.choice_head_weights = (
                self.lm_head.weight[choice_token_tensor].detach().clone()
            )
            self.is_sliced_head = True

        # For performance metrics
        self.preprocess_time_ms = 0.0
        self.forward_time_ms = 0.0
        self.postprocess_time_ms = 0.0

    @torch.inference_mode()
    def route(
        self,
        image: Image.Image,
        context: str,
        instruction: str,
        choices: list[str],
        temperature: float = 1.0,
    ) -> RouteResult:
        num_choices = len(choices)
        if num_choices > self.max_choices:
            raise ValueError(
                f"Number of choices ({num_choices}) exceeds max_choices "
                f"({self.max_choices})"
            )

        t0 = time.perf_counter()

        formatted_choices = "\n".join(
            [f"{self.choice_letters[i]}. {choice}" for i, choice in enumerate(choices)]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"Context: {context}\n\nTask: {instruction}\n\nChoices:\n{formatted_choices}\n\nSelect the single correct option letter.",
                    },
                ],
            }
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if not prompt.endswith("Answer: "):
                prompt += "Answer: "
        except Exception:
            # Fallback if tokenizer lacks a chat template
            prompt = f"<image>\nContext: {context}\n\nTask: {instruction}\n\nChoices:\n{formatted_choices}\n\nSelect the single correct option letter.\nAnswer: "

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        t1 = time.perf_counter()
        self.preprocess_time_ms = (t1 - t0) * 1000.0

        # Forward pass
        # Since we use use_cache=False, we just get logits or hidden states
        # Vision2Seq returns logits directly if output_hidden_states is not True
        # But we want to slice the LM head for efficiency if possible.
        # Standard Vision2Seq model gives logits. To slice we'd need hidden states.
        # Let's request hidden states.
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        t2 = time.perf_counter()
        self.forward_time_ms = (t2 - t1) * 1000.0

        # We need the last hidden state of the language model part.
        # For Vision2Seq models, `hidden_states` might be from the language model, but some models return language_model.hidden_states.
        # outputs.hidden_states is typically available when output_hidden_states=True.
        # Let's handle generic case.
        # Find the last hidden state
        last_hidden_state = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        elif (
            hasattr(outputs, "language_model_outputs")
            and hasattr(outputs.language_model_outputs, "hidden_states")
            and outputs.language_model_outputs.hidden_states
        ):
            last_hidden_state = outputs.language_model_outputs.hidden_states[-1][
                :, -1, :
            ]
        elif (
            hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states
        ):
            last_hidden_state = outputs.decoder_hidden_states[-1][:, -1, :]

        logits = None
        if self.is_sliced_head and last_hidden_state is not None:
            active_head_weights = self.choice_head_weights[:num_choices]
            logits = torch.matmul(last_hidden_state, active_head_weights.t())
        else:
            # Fallback: use full logits
            full_logits = outputs.logits[:, -1, :]
            choice_token_tensor = torch.tensor(
                self.choice_token_ids[:num_choices], device=self.device
            )
            logits = full_logits[:, choice_token_tensor]

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

        entropy = -sum(p * math.log(p + 1e-9) for p in probs)

        best_idx = probs.index(max(probs))
        best_letter = self.choice_letters[best_idx]
        best_choice = choices[best_idx]
        confidence = probs[best_idx]

        distribution = {self.choice_letters[i]: probs[i] for i in range(num_choices)}

        t3 = time.perf_counter()
        self.postprocess_time_ms = (t3 - t2) * 1000.0

        return RouteResult(
            best_choice=best_choice,
            best_letter=best_letter,
            confidence=confidence,
            entropy=entropy,
            distribution=distribution,
        )
