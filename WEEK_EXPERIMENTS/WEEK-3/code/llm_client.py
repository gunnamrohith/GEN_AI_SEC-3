import sys
import time
from abc import ABC, abstractmethod
from typing import TypedDict

import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pymongo.errors import PyMongoError

from cost_calculator import calculate_cost
from mongo_logger import MongoLogger


class GenerationResult(TypedDict):
    response: str
    input_tokens: int | None
    output_tokens: int | None


class LLMProvider(ABC):
    model: str

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        pass

    def close(self) -> None:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash") -> None:
        if not api_key.strip() or api_key.startswith("YOUR_"):
            raise ValueError("Set GEMINI_API_KEY in the Week 3 .env file.")
        if not model.strip():
            raise ValueError("GEMINI_MODEL must not be empty.")
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000),
        )

    def generate(self, prompt: str) -> GenerationResult:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        if not response.text or not response.text.strip():
            raise ValueError("Gemini returned no text. Try another prompt.")
        usage = response.usage_metadata
        output_tokens = usage.candidates_token_count if usage else None
        if output_tokens is not None and usage is not None:
            output_tokens += usage.thoughts_token_count or 0
        return {
            "response": response.text,
            "input_tokens": usage.prompt_token_count if usage else None,
            "output_tokens": output_tokens,
        }

    def close(self) -> None:
        self.client.close()


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.2") -> None:
        if not model.strip():
            raise ValueError("OLLAMA_MODEL must not be empty.")
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> GenerationResult:
        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Ollama returned no text. Check the model.")
        return {
            "response": answer,
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        }


def is_retryable(error: Exception) -> bool:
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    status = None
    if isinstance(error, APIError):
        status = error.code
    elif isinstance(error, requests.HTTPError) and error.response is not None:
        status = error.response.status_code
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


class LLMClient:
    def __init__(self, provider: LLMProvider, logger: MongoLogger) -> None:
        self.provider = provider
        self.logger = logger

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Enter a non-empty question.")
        start = time.perf_counter()
        for attempt in range(1, 4):
            try:
                result = self.provider.generate(prompt)
            except (APIError, requests.RequestException, ValueError) as error:
                if is_retryable(error) and attempt < 3:
                    delay = 2 ** (attempt - 1)
                    print(
                        f"Attempt {attempt} failed; retrying in {delay} seconds.",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                try:
                    self.logger.log_call({
                        "record_type": "llm_call",
                        "provider": type(self.provider).__name__,
                        "model": self.provider.model,
                        "prompt": prompt,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                        "cost": None,
                        "cost_basis": "unknown",
                        "attempts": attempt,
                        "status": "failed",
                        "error": str(error),
                    })
                except PyMongoError:
                    print(
                        "MongoDB logging also failed; the failed call was not saved.",
                        file=sys.stderr,
                    )
                raise

            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]
            cost = calculate_cost(
                type(self.provider).__name__,
                self.provider.model,
                input_tokens,
                output_tokens,
            )
            # Keep database writes outside the provider retry block.
            try:
                inserted_id = self.logger.log_call({
                    "record_type": "llm_call",
                    "provider": type(self.provider).__name__,
                    "model": self.provider.model,
                    "prompt": prompt,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": (
                        input_tokens + output_tokens
                        if input_tokens is not None and output_tokens is not None
                        else None
                    ),
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    "cost": cost,
                    "cost_basis": (
                        "local_no_api_fee"
                        if isinstance(self.provider, OllamaProvider)
                        else "sample_rates" if cost is not None else "unknown"
                    ),
                    "attempts": attempt,
                    "status": "success",
                    "error": None,
                })
            except PyMongoError:
                print(
                    "\nAI response (NOT saved to MongoDB):\n" + result["response"],
                    file=sys.stderr,
                )
                raise
            print(f"Call details saved in MongoDB! Log ID: {inserted_id}")
            return result["response"]
        raise RuntimeError("No generation attempt completed.")
