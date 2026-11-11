import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from google.genai.errors import APIError
import requests

from llm_client import GeminiProvider, LLMClient, LLMProvider, OllamaProvider


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"), override=True)
    provider_name = input("Choose provider (gemini/ollama): ").strip().lower()
    provider: LLMProvider

    if provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY in the Week 2 .env file.")
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        if not model:
            raise ValueError("GEMINI_MODEL must not be empty.")
        provider = GeminiProvider(api_key, model=model)
    elif provider_name == "ollama":
        provider = OllamaProvider("llama3.2")
    else:
        raise ValueError("Invalid provider. Choose gemini or ollama.")

    client = LLMClient(provider)
    prompt = input("Enter your question: ")
    answer = client.generate(prompt)
    print("\nLLM Response:")
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except requests.ConnectionError:
        print("Error: Cannot connect to Ollama. Start it with 'ollama serve'.", file=sys.stderr)
        sys.exit(1)
    except requests.Timeout:
        print("Error: Ollama exceeded the 120-second timeout. Try again.", file=sys.stderr)
        sys.exit(1)
    except (requests.RequestException, APIError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
