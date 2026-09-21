from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from logit_router.schema import RouteResult
from logit_router.vision_router import VisionLogitRouter


@pytest.fixture
def mock_vision_router():
    with (
        patch("logit_router.vision_router.AutoProcessor") as mock_processor_cls,
        patch("logit_router.vision_router.AutoModelForVision2Seq") as mock_model_cls,
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_proc_inst = MagicMock()
        mock_proc_inst.tokenizer.encode.return_value = [42]

        # Ensure the fallback works when `apply_chat_template` is called.
        mock_proc_inst.apply_chat_template.return_value = "prompt Answer: "

        class FakeInputs(dict):
            def to(self, device):
                return self

        mock_proc_inst.return_value = FakeInputs(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.randn(1, 3, 224, 224),
            }
        )

        mock_processor_cls.from_pretrained.return_value = mock_proc_inst

        mock_model_inst = MagicMock()
        fake_weights = torch.randn(100, 32)

        # Setup lm_head mock
        mock_lm_head = MagicMock()
        mock_lm_head.weight = fake_weights
        mock_model_inst.lm_head = mock_lm_head

        mock_model_cls.from_pretrained.return_value = mock_model_inst

        router = VisionLogitRouter(
            model_id="fake/vision_model", device="cpu", max_choices=5
        )

        return router


def test_vision_route_result_structure(mock_vision_router):
    class FakeOutputs:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None

    # We need to simulate full logits fallback or hidden states.
    # Let's test the full logits fallback first.
    mock_vision_router.is_sliced_head = False
    logits = torch.zeros(1, 1, 100)
    logits[0, -1, 42] = 10.0  # Make one choice very likely
    mock_vision_router.model.return_value = FakeOutputs(logits)

    dummy_img = Image.new("RGB", (10, 10), color="red")

    result = mock_vision_router.route(
        image=dummy_img,
        context="ctx",
        instruction="inst",
        choices=["Choice A", "Choice B"],
    )

    assert isinstance(result, RouteResult)
    assert result.best_choice in ["Choice A", "Choice B"]
    assert "A" in result.distribution
    assert "B" in result.distribution
    # Assert time metrics were recorded
    assert mock_vision_router.preprocess_time_ms >= 0
    assert mock_vision_router.forward_time_ms >= 0
    assert mock_vision_router.postprocess_time_ms >= 0


def test_vision_max_choices_exceeded(mock_vision_router):
    dummy_img = Image.new("RGB", (10, 10), color="red")
    with pytest.raises(ValueError, match="exceeds max_choices"):
        mock_vision_router.route(
            image=dummy_img,
            context="ctx",
            instruction="inst",
            choices=["1", "2", "3", "4", "5", "6"],
        )


def test_vision_gemma_processor_fallback(mock_vision_router):
    # Test fallback if apply_chat_template raises an exception
    mock_vision_router.processor.apply_chat_template.side_effect = Exception(
        "No template"
    )

    class FakeOutputs:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None

    mock_vision_router.is_sliced_head = False
    mock_vision_router.model.return_value = FakeOutputs(torch.randn(1, 1, 100))

    dummy_img = Image.new("RGB", (10, 10), color="blue")
    result = mock_vision_router.route(
        image=dummy_img, context="ctx", instruction="inst", choices=["A", "B"]
    )
    assert isinstance(result, RouteResult)
    # the processor was still called
    assert mock_vision_router.processor.called


def test_vision_sliced_head_logic(mock_vision_router):
    class FakeOutputs:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states
            self.logits = None

    mock_vision_router.is_sliced_head = True
    hidden_states = [torch.randn(1, 10, 32)]  # (batch, seq, hidden)
    mock_vision_router.model.return_value = FakeOutputs(hidden_states)
    mock_vision_router.choice_head_weights = torch.randn(5, 32)

    dummy_img = Image.new("RGB", (10, 10), color="green")

    result = mock_vision_router.route(
        image=dummy_img, context="ctx", instruction="inst", choices=["Opt 1", "Opt 2"]
    )
    assert isinstance(result, RouteResult)
