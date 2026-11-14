import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from google.genai.errors import APIError
from pymongo.errors import ConfigurationError, PyMongoError

from llm_client import GeminiProvider, LLMClient, LLMProvider, OllamaProvider
from mongo_logger import MongoLogger


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    print("WEEK 3: LLM MONITORING WITH MONGODB (LOCAL OR ATLAS)")
    print("1. Gemini (cloud; no local server)")
    print("2. Ollama (optional; local service required)")
    choice = input("Select provider (1/2 or gemini/ollama): ").strip().lower()
    if choice not in {"1", "2", "gemini", "ollama"}:
        raise ValueError("Invalid provider. Choose 1 (Gemini) or 2 (Ollama).")
    prompt = input("Enter your question: ").strip()
    if not prompt:
        raise ValueError("Enter a non-empty question.")
    provider: LLMProvider
    if choice in {"1", "gemini"}:
        provider = GeminiProvider(
            os.getenv("GEMINI_API_KEY", ""),
            os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        )
    else:
        provider = OllamaProvider(os.getenv("OLLAMA_MODEL", "llama3.2"))

    logger = None
    try:
        logger = MongoLogger()
        print("MongoDB connection successful.")
        answer = LLMClient(provider, logger).generate(prompt)
        print("\n========== AI RESPONSE ==========")
        print(answer)
        print("\n========== COMPLETED ==========")
    finally:
        if logger is not None:
            logger.close()
        provider.close()


if __name__ == "__main__":
    try:
        main()
    except (ConfigurationError, PyMongoError):
        print(
            "MongoDB error: check your URI, database user and server/network. "
            "For Atlas, also check the IP access list. "
            "Run 'python check_connection.py --write'. "
            "The lab is not complete until the write succeeds.",
            file=sys.stderr,
        )
        sys.exit(1)
    except APIError as error:
        print(
            f"Gemini API error (HTTP {error.code}): check your key, model "
            "availability and quota in AI Studio.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.RequestException as error:
        print(f"Ollama request failed ({type(error).__name__}).", file=sys.stderr)
        sys.exit(1)
    except (ValueError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
