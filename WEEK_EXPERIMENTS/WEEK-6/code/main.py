import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import TypedDict

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
RELEASE = "v0.1"
INPUT_FILE = Path(__file__).with_name("golden_set.json")
OUTPUT_FILE = Path(__file__).with_name("baseline_results.csv")
FIELDNAMES = [
    "Question_ID", "Question", "Expected_Answer", "LLM_Answer",
    "Matched_Keywords", "Missing_Keywords", "Score", "Status",
    "Source", "Model", "Release",
]


class GoldenQuestion(TypedDict):
    id: str
    question: str
    expected_answer: str
    keywords: list[str]
    source: str


class BaselineResult(TypedDict):
    Question_ID: str
    Question: str
    Expected_Answer: str
    LLM_Answer: str
    Matched_Keywords: str
    Missing_Keywords: str
    Score: float
    Status: str
    Source: str
    Model: str
    Release: str


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    with path.open(encoding="utf-8") as file:
        items = json.load(file)
    if not isinstance(items, list) or not items:
        raise ValueError("The golden set must be a non-empty JSON list.")

    questions: list[GoldenQuestion] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        for field in ("id", "question", "expected_answer", "source"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Question {index}: {field} must be non-empty text.")
        keywords = item.get("keywords")
        if not isinstance(keywords, list) or any(
            not isinstance(word, str) or not word.strip() for word in keywords
        ):
            raise ValueError(f"Question {index}: keywords must be a list of non-empty strings.")
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate question ID: {item['id']}")
        seen_ids.add(item["id"])
        questions.append(GoldenQuestion(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            keywords=keywords,
            source=item["source"],
        ))
    return questions


def get_llm_answer(question: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": question,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Ollama returned a non-object JSON response.")
    answer = result.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned no answer text.")
    return answer.strip()


def calculate_score(
    answer: str, expected_keywords: list[str]
) -> tuple[float, list[str], list[str]]:
    lowered = answer.lower()
    matched = [word for word in expected_keywords if word.lower() in lowered]
    missing = [word for word in expected_keywords if word.lower() not in lowered]
    score = round(len(matched) / len(expected_keywords) * 100, 2) if expected_keywords else 0.0
    return score, matched, missing


def run_baseline(questions: list[GoldenQuestion]) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    for index, item in enumerate(questions, start=1):
        print(f"\nQuestion {index}: {item['id']}", flush=True)
        print("Question:", item["question"])
        answer = ""
        score = 0.0
        matched: list[str] = []
        missing = item["keywords"]
        status = "ERROR"
        try:
            answer = get_llm_answer(item["question"])
            score, matched, missing = calculate_score(answer, item["keywords"])
            status = "PASS" if score >= 60 else "REVIEW"
            print("\nLLM Answer:\n" + answer)
        except (requests.RequestException, ValueError) as error:
            print(f"Error [{item['id']}]: {error}", flush=True)

        print("\nExpected Answer:", item["expected_answer"])
        print("Matched Keywords:", matched)
        print("Missing Keywords:", missing)
        print("Score:", score, "%")
        print("Status:", status, flush=True)
        results.append(BaselineResult(
            Question_ID=item["id"],
            Question=item["question"],
            Expected_Answer=item["expected_answer"],
            LLM_Answer=answer,
            Matched_Keywords=", ".join(matched),
            Missing_Keywords=", ".join(missing),
            Score=score,
            Status=status,
            Source=item["source"],
            Model=MODEL_NAME,
            Release=RELEASE,
        ))
    return results


def save_results(results: list[BaselineResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    questions = load_golden_set(INPUT_FILE)
    print("GOLDEN QUESTION SET")
    print("Total Questions:", len(questions))
    print("Model:", MODEL_NAME)
    print("\nBASELINE RUN STARTED", flush=True)
    results = run_baseline(questions)
    save_results(results, OUTPUT_FILE)
    successful = [result for result in results if result["Status"] != "ERROR"]
    errors = len(results) - len(successful)
    average = sum(result["Score"] for result in successful) / len(successful) if successful else 0

    print("\nBASELINE EVALUATION SUMMARY")
    print("Release Version:", RELEASE)
    print("Model:", MODEL_NAME)
    print("Total Questions:", len(results))
    print("Successful Responses:", len(successful))
    print("Passed Questions:", sum(result["Status"] == "PASS" for result in results))
    print("Questions for Review:", sum(result["Status"] == "REVIEW" for result in results))
    print("Failed Requests:", errors)
    print("Average Keyword Coverage:", round(average, 2), "%")
    print("Evaluation Date:", datetime.now().astimezone().isoformat(timespec="seconds"))
    print("Results saved to:", OUTPUT_FILE)
    print("Keyword coverage is not factual accuracy; manually review answers and sources.")
    if errors:
        print("Baseline completed with errors. Fix failed requests before finalizing Release v0.1.")
        return 1
    print("BASELINE RUN COMPLETED! Release v0.1 created successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
