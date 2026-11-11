import unittest
from unittest.mock import Mock, patch

import requests

from llm_client import GeminiProvider, LLMClient, LLMProvider, OllamaProvider
from main import main


class FakeProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "AI stands for Artificial Intelligence."


class TestLLMClient(unittest.TestCase):
    def test_generate(self):
        client = LLMClient(FakeProvider())
        self.assertEqual(
            client.generate("Explain AI"),
            "AI stands for Artificial Intelligence.",
        )

    def test_forwards_prompt(self):
        provider = Mock(spec=LLMProvider)
        provider.generate.return_value = "answer"
        self.assertEqual(LLMClient(provider).generate("Explain AI"), "answer")
        provider.generate.assert_called_once_with("Explain AI")

    def test_rejects_empty_prompt(self):
        provider = Mock(spec=LLMProvider)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            LLMClient(provider).generate("   ")
        provider.generate.assert_not_called()


class TestOllamaProvider(unittest.TestCase):
    @patch("llm_client.requests.post")
    def test_generate(self, post):
        post.return_value.json.return_value = {"response": "Local answer"}
        self.assertEqual(OllamaProvider().generate("Explain AI"), "Local answer")
        post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": "Explain AI", "stream": False},
            timeout=120,
        )
        post.return_value.raise_for_status.assert_called_once()

    @patch("llm_client.requests.post")
    def test_http_failure(self, post):
        post.return_value.raise_for_status.side_effect = requests.HTTPError("404")
        with self.assertRaises(requests.HTTPError):
            OllamaProvider().generate("Explain AI")
        post.return_value.json.assert_not_called()

    @patch("llm_client.requests.post")
    def test_connection_failure(self, post):
        post.side_effect = requests.ConnectionError("Offline")
        with self.assertRaises(requests.ConnectionError):
            OllamaProvider().generate("Explain AI")

    @patch("llm_client.requests.post")
    def test_empty_response(self, post):
        post.return_value.json.return_value = {}
        with self.assertRaisesRegex(ValueError, "no text"):
            OllamaProvider().generate("Explain AI")


class TestGeminiProvider(unittest.TestCase):
    @patch("llm_client.genai.Client")
    def test_generate(self, client):
        client.return_value.models.generate_content.return_value.text = "Cloud answer"
        provider = GeminiProvider("test-key", model="test-model")
        self.assertEqual(provider.generate("Explain AI"), "Cloud answer")
        client.return_value.models.generate_content.assert_called_once_with(
            model="test-model", contents="Explain AI"
        )

    def test_missing_key(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            GeminiProvider(" ")

    @patch("llm_client.genai.Client")
    def test_empty_response(self, client):
        client.return_value.models.generate_content.return_value.text = None
        with self.assertRaisesRegex(ValueError, "no text"):
            GeminiProvider("test-key").generate("Explain AI")


class TestMain(unittest.TestCase):
    @patch("main.load_dotenv")
    @patch("builtins.input", return_value="unknown")
    def test_invalid_provider(self, user_input, load_env):
        with self.assertRaisesRegex(ValueError, "Invalid provider"):
            main()

    @patch("main.load_dotenv")
    @patch.dict("os.environ", {}, clear=True)
    @patch("builtins.input", return_value="gemini")
    def test_missing_gemini_key(self, user_input, load_env):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            main()

    @patch("main.load_dotenv")
    @patch("main.OllamaProvider")
    @patch("builtins.input", side_effect=["ollama", "Explain AI"])
    @patch("builtins.print")
    def test_ollama_selection(self, output, user_input, provider, load_env):
        provider.return_value.generate.return_value = "Local answer"
        main()
        provider.assert_called_once_with("llama3.2")
        provider.return_value.generate.assert_called_once_with("Explain AI")
        output.assert_any_call("Local answer")


if __name__ == "__main__":
    unittest.main()
