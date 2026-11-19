import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

import main


class TestScoring(unittest.TestCase):
    def test_case_insensitive_keywords_and_phrases(self):
        self.assertEqual(
            main.calculate_score("INPUT TOKENS form the prompt.", ["input tokens", "prompt", "output"]),
            (66.67, ["input tokens", "prompt"], ["output"]),
        )

    def test_empty_keywords(self):
        self.assertEqual(main.calculate_score("An answer", []), (0.0, [], []))

    def test_preserves_handout_substring_matching(self):
        self.assertEqual(main.calculate_score("subword", ["word"])[0], 100)


class TestOllama(unittest.TestCase):
    @patch("main.requests.post")
    def test_request_and_answer(self, post):
        post.return_value.json.return_value = {"response": " Answer \n"}
        self.assertEqual(main.get_llm_answer("Question?"), "Answer")
        post.assert_called_once_with(
            main.OLLAMA_URL,
            json={
                "model": "llama3.2", "prompt": "Question?", "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        post.return_value.raise_for_status.assert_called_once()

    @patch("main.requests.post")
    def test_http_error_is_not_scored(self, post):
        post.return_value.raise_for_status.side_effect = requests.HTTPError("404 model missing")
        with self.assertRaises(requests.HTTPError):
            main.get_llm_answer("Question?")
        post.return_value.json.assert_not_called()

    @patch("main.requests.post")
    def test_invalid_or_empty_response(self, post):
        for payload in ({}, {"response": " "}, {"response": 3}, []):
            with self.subTest(payload=payload):
                post.return_value.json.return_value = payload
                with self.assertRaises(ValueError):
                    main.get_llm_answer("Question?")


class TestGoldenSet(unittest.TestCase):
    def test_invalid_input_and_duplicate_ids(self):
        valid = main.load_golden_set(main.INPUT_FILE)[0]
        for data in ({}, [], [None], [{**valid, "source": ""}],
                     [{**valid, "keywords": [""]}], [valid, valid]):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "questions.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    main.load_golden_set(path)


class TestBaseline(unittest.TestCase):
    def test_threshold_errors_continuation_and_csv(self):
        questions = [
            main.GoldenQuestion(
                id=f"Q{index}", question="Explain?", expected_answer="Reference",
                keywords=["alpha", "beta", "gamma", "delta", "epsilon"],
                source="https://example.com/reference",
            )
            for index in range(1, 5)
        ]
        with patch("main.get_llm_answer", side_effect=[
            "alpha beta gamma", "alpha beta", requests.Timeout("Timed out"),
            "alpha beta gamma delta epsilon",
        ]), contextlib.redirect_stdout(io.StringIO()) as output:
            results = main.run_baseline(questions)
        self.assertEqual([row["Status"] for row in results], ["PASS", "REVIEW", "ERROR", "PASS"])
        self.assertEqual([row["Score"] for row in results], [60, 40, 0, 100])
        self.assertEqual(results[2]["LLM_Answer"], "")
        self.assertEqual(results[2]["Missing_Keywords"], "alpha, beta, gamma, delta, epsilon")
        self.assertIn("Timed out", output.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            main.save_results(results, path)
            with path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, main.FIELDNAMES)
                rows = list(reader)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["Source"], questions[0]["source"])
            self.assertEqual(rows[2]["Status"], "ERROR")
            self.assertTrue(all(row["Release"] == "v0.1" for row in rows))

    def test_failed_release_returns_nonzero(self):
        questions = main.load_golden_set(main.INPUT_FILE)[:1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with patch("main.load_golden_set", return_value=questions), \
                 patch("main.get_llm_answer", side_effect=requests.ConnectionError("Ollama offline")), \
                 patch("main.OUTPUT_FILE", path), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main.main(), 1)
            self.assertIn("Failed Requests: 1", output.getvalue())
            self.assertNotIn("created successfully", output.getvalue())
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
