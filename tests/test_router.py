import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from logit_router.router import LogitRouter
from logit_router.schema import RouteRequest, RouteResult


def test_route_result_structure():
    result = RouteResult(
        best_choice="Choice A",
        best_letter="A",
        confidence=0.8,
        entropy=0.5,
        distribution={"A": 0.8, "B": 0.2},
    )
    assert result.best_choice == "Choice A"
    assert result.best_letter == "A"
    assert result.confidence == 0.8
    assert result.entropy == 0.5
    assert result.distribution == {"A": 0.8, "B": 0.2}
    assert isinstance(result.best_choice, str)
    assert isinstance(result.best_letter, str)
    assert isinstance(result.confidence, float)
    assert isinstance(result.entropy, float)
    assert isinstance(result.distribution, dict)


def test_route_result_methods():
    result1 = RouteResult("Choice A", "A", 0.8, 1.0, {"A": 0.8, "B": 0.2})
    assert result1.is_confident(0.7) is True
    assert result1.needs_fallback(0.9) is True

    result2 = RouteResult("Choice B", "B", 0.6, 0.5, {"A": 0.4, "B": 0.6})
    assert result2.is_confident(0.7) is False
    assert result2.needs_fallback(0.9) is False


def test_route_request():
    req = RouteRequest(
        context="test context",
        instruction="test instruction",
        choices=["A", "B"],
        temperature=0.5,
    )
    assert req.context == "test context"
    assert req.instruction == "test instruction"
    assert req.choices == ["A", "B"]
    assert req.temperature == 0.5


@pytest.fixture
def mock_router():
    with (
        patch("logit_router.router.AutoTokenizer") as mock_tokenizer,
        patch("logit_router.router.AutoModelForCausalLM") as mock_model_cls,
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_tok_inst = MagicMock()
        mock_tok_inst.encode.return_value = [42]

        # Make the mock act as a callable for prompt tokenization
        class FakeInputs(dict):
            def to(self, device):
                return self

        mock_tok_inst.return_value = FakeInputs(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

        mock_tokenizer.from_pretrained.return_value = mock_tok_inst

        mock_model_inst = MagicMock()
        fake_weights = torch.randn(100, 32)
        mock_model_inst.lm_head.weight.__getitem__.return_value = (
            fake_weights.detach().clone()
        )
        mock_model_cls.from_pretrained.return_value = mock_model_inst

        router = LogitRouter(model_id="fake/model", device="cpu", max_choices=5)

        return router


def test_max_choices_exceeded(mock_router):
    with pytest.raises(ValueError, match="exceeds max_choices"):
        mock_router.route("ctx", "inst", ["1", "2", "3", "4", "5", "6"])


def test_distribution_probability_sum(mock_router):
    class FakeOutputs:
        def __init__(self, hidden_state):
            self.last_hidden_state = hidden_state

    hidden = torch.randn(1, 10, 32)
    mock_router.backbone.return_value = FakeOutputs(hidden)

    result = mock_router.route("ctx", "inst", ["Choice X", "Choice Y", "Choice Z"])
    probs = result["distribution"].values()
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)


def test_entropy_calculation(mock_router):
    class FakeOutputs:
        def __init__(self, hidden_state):
            self.last_hidden_state = hidden_state

    # Provide identity weights to cleanly map hidden_state to logits
    mock_router.choice_head_weights = torch.eye(32)

    # Case 1: Uniform distribution (all logits zero)
    hidden_uniform = torch.zeros(1, 10, 32)
    mock_router.backbone.return_value = FakeOutputs(hidden_uniform)

    res_uniform = mock_router.route("ctx", "inst", ["Choice A", "Choice B", "Choice C"])
    entropy_uniform = res_uniform["entropy"]

    # Case 2: Skewed distribution
    hidden_skewed = torch.zeros(1, 10, 32)
    hidden_skewed[0, -1, 0] = 100.0  # huge logit for the first choice
    mock_router.backbone.return_value = FakeOutputs(hidden_skewed)

    res_skewed = mock_router.route("ctx", "inst", ["Choice A", "Choice B", "Choice C"])
    entropy_skewed = res_skewed["entropy"]

    assert entropy_uniform > entropy_skewed
    assert math.isclose(entropy_uniform, math.log(3), rel_tol=1e-2)
    assert math.isclose(entropy_skewed, 0.0, abs_tol=1e-2)


def test_fallback_router():
    from logit_router.optimizations import FallbackRouter

    mock_router = MagicMock()
    # Case where router is confident
    mock_router.route.return_value = {
        "best_choice": "A",
        "best_letter": "A",
        "confidence": 0.95,
        "entropy": 0.1,
        "distribution": {"A": 0.95, "B": 0.05},
    }

    fallback_fn = MagicMock(return_value={"best_choice": "Fallback"})
    fb_router = FallbackRouter(
        router=mock_router,
        entropy_threshold=0.9,
        margin_threshold=0.1,
        fallback_fn=fallback_fn,
    )

    res = fb_router.route("ctx", "inst", ["A", "B"])
    assert res["fallback_executed"] is False
    assert res["best_choice"] == "A"
    assert not fallback_fn.called

    # Case where router is uncertain (high entropy)
    mock_router.route.return_value = {
        "best_choice": "A",
        "best_letter": "A",
        "confidence": 0.51,
        "entropy": 1.5,
        "distribution": {"A": 0.51, "B": 0.49},
    }

    res_uncertain = fb_router.route("ctx", "inst", ["A", "B"])
    assert res_uncertain["fallback_executed"] is True
    assert fallback_fn.called


def test_apply_torch_compile():
    from logit_router.optimizations import apply_torch_compile

    mock_router = MagicMock()
    orig_backbone = mock_router.backbone
    with patch("torch.compile", return_value="compiled_backbone") as mock_compile:
        result = apply_torch_compile(mock_router)
        assert result is mock_router
        assert mock_router.backbone == "compiled_backbone"
        mock_compile.assert_called_once_with(
            orig_backbone, mode="reduce-overhead", fullgraph=False
        )

def test_gemma_tokenizer(mock_router):
    with patch("logit_router.router.AutoTokenizer") as mock_tokenizer, \
         patch("logit_router.router.AutoModelForCausalLM") as mock_model_cls, \
         patch("torch.cuda.is_available", return_value=False):

        mock_tok_inst = MagicMock()
        mock_tok_inst.encode.side_effect = lambda x, **kwargs: [ord(x[-1])]
        mock_tokenizer.from_pretrained.return_value = mock_tok_inst

        mock_model_inst = MagicMock()
        fake_weights = torch.randn(100, 32)
        mock_model_inst.lm_head.weight = fake_weights
        mock_model_cls.from_pretrained.return_value = mock_model_inst

        LogitRouter(model_id="google/gemma-2b", device="cpu", max_choices=2)

        # In Gemma, it should encode exactly 'A' not ' A'
        mock_tok_inst.encode.assert_any_call("A", add_special_tokens=False)
        mock_tok_inst.encode.assert_any_call("B", add_special_tokens=False)

        LogitRouter(model_id="Qwen/Qwen2.5-1.5B", device="cpu", max_choices=2)
        mock_tok_inst.encode.assert_any_call(" A", add_special_tokens=False)
        mock_tok_inst.encode.assert_any_call(" B", add_special_tokens=False)

def test_awq_quantized_head(mock_router):
    with patch("logit_router.router.AutoTokenizer") as mock_tokenizer, \
         patch("logit_router.router.AutoModelForCausalLM") as mock_model_cls, \
         patch("torch.cuda.is_available", return_value=False):

        mock_tok_inst = MagicMock()
        mock_tok_inst.encode.return_value = [42]

        class FakeInputs(dict):
            def to(self, device): return self

        mock_tok_inst.return_value = FakeInputs({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })
        mock_tokenizer.from_pretrained.return_value = mock_tok_inst

        mock_model_inst = MagicMock()
        # Remove weight attribute to simulate AWQ quantized head
        del mock_model_inst.lm_head.weight

        # Create a mock for lm_head call
        def mock_lm_head_call(x):
            # return shape: (batch, seq, vocab)
            return torch.ones(x.shape[0], 100)

        mock_model_inst.lm_head.side_effect = mock_lm_head_call

        class FakeOutputs:
            def __init__(self, hidden_state):
                self.last_hidden_state = hidden_state
        mock_model_inst.model.return_value = FakeOutputs(torch.randn(1, 10, 32))

        mock_model_cls.from_pretrained.return_value = mock_model_inst

        router_awq = LogitRouter(model_id="TheBloke/Llama-2-7B-AWQ", device="cpu", max_choices=2)

        assert not router_awq.is_sliced_head

        result = router_awq.route("ctx", "inst", ["A", "B"])
        assert result["best_choice"] in ["A", "B"]
        assert len(result["distribution"]) == 2
