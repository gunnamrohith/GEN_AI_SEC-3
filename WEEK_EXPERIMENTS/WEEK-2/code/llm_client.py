from abc import ABC, abstractmethod

from google import genai
import requests


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        if not api_key.strip():
            raise ValueError("Gemini API key is required.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        if not response.text or not response.text.strip():
            raise ValueError("Gemini returned no text. Try another prompt.")
        return response.text


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Ollama returned no text. Check the model and try again.")
        return answer


class LLMClient:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Enter a non-empty question.")
        return self.provider.generate(prompt)
