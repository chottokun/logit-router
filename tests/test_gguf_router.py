import unittest
from unittest.mock import patch, MagicMock
from logit_router.router import LogitRouter

class TestGGUFRouter(unittest.TestCase):
    @patch("logit_router.router.AutoTokenizer.from_pretrained")
    @patch("logit_router.router.AutoModelForCausalLM.from_pretrained")
    @patch("torch.cuda.is_available", return_value=False)
    def test_gguf_load(self, mock_cuda, mock_model, mock_tokenizer):
        mock_tokenizer_instance = MagicMock()
        # Return different token ids depending on the input to simulate is_qwen
        def mock_encode(text, **kwargs):
            if text == " A":
                return [362]
            if text == "A":
                return [32]
            if text == " B":
                return [363]
            if text == "B":
                return [33]
            return [1]
        mock_tokenizer_instance.encode.side_effect = mock_encode
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        # Test Qwen GGUF
        model_id = "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2-GGUF"
        gguf_file = "Qwen3.5-0.8B-Japanese-SFT-v2-Q4_K_M.gguf"
        
        router = LogitRouter(model_id=model_id, gguf_file=gguf_file, max_choices=2)

        mock_tokenizer.assert_called_with(model_id, gguf_file=gguf_file, use_fast=True)
        # Should use encode("A") not encode(" A") because of Qwen
        self.assertEqual(router.choice_token_ids, [32, 33])
        
if __name__ == '__main__':
    unittest.main()
