import unittest
from datetime import timezone
from unittest.mock import Mock, patch

import requests
from bson import ObjectId
from google.genai.errors import APIError
from pymongo.errors import ConnectionFailure, OperationFailure

from check_connection import check_connection
from cost_calculator import calculate_cost
from llm_client import GeminiProvider, LLMClient, LLMProvider, OllamaProvider
from main import main
from mongo_logger import MongoLogger


class TestCost(unittest.TestCase):
    def test_sample_rates(self):
        self.assertEqual(
            calculate_cost("GeminiProvider", "gemini-3.8-flash", 100, 50),
            0.00015,
        )

    def test_unknown_and_missing_usage(self):
        self.assertIsNone(calculate_cost("GeminiProvider", "other", 100, 50))
        self.assertIsNone(
            calculate_cost("GeminiProvider", "gemini-3.8-flash", None, 50)
        )
        self.assertEqual(calculate_cost("OllamaProvider", "llama3.2", 100, 50), 0)


class TestMonitoring(unittest.TestCase):
    def setUp(self):
        self.provider = Mock(spec=LLMProvider)
        self.provider.model = "gemini-3.8-flash"
        self.provider.generate.return_value = {
            "response": "AI answer", "input_tokens": 10, "output_tokens": 5,
        }
        self.logger = Mock(spec=MongoLogger)
        self.client = LLMClient(self.provider, self.logger)

    def test_success_shape_and_latency(self):
        with patch("llm_client.time.perf_counter", side_effect=[1.0, 1.125]):
            self.assertEqual(self.client.generate("Explain AI"), "AI answer")
        document = self.logger.log_call.call_args.args[0]
        self.assertEqual(document["record_type"], "llm_call")
        self.assertEqual(document["total_tokens"], 15)
        self.assertEqual(document["latency_ms"], 125.0)
        self.assertEqual(document["attempts"], 1)
        self.assertEqual(document["status"], "success")
        self.assertIsNone(document["error"])
        self.logger.log_call.assert_called_once()

    @patch("llm_client.time.sleep")
    def test_transient_failure_retries_then_logs_once(self, sleep):
        result = self.provider.generate.return_value
        self.provider.generate.side_effect = [requests.Timeout(), result]
        self.client.generate("Hello")
        sleep.assert_called_once_with(1)
        self.assertEqual(self.provider.generate.call_count, 2)
        self.logger.log_call.assert_called_once()
        self.assertEqual(self.logger.log_call.call_args.args[0]["attempts"], 2)

    @patch("llm_client.time.sleep")
    def test_exhausted_retries(self, sleep):
        self.provider.generate.side_effect = requests.ConnectionError("offline")
        with self.assertRaises(requests.ConnectionError):
            self.client.generate("Hello")
        self.assertEqual(self.provider.generate.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.logger.log_call.assert_called_once()
        document = self.logger.log_call.call_args.args[0]
        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["attempts"], 3)
        self.assertIsNone(document["cost"])

    @patch("llm_client.time.sleep")
    def test_gemini_rate_limit_uses_sdk_code(self, sleep):
        result = self.provider.generate.return_value
        self.provider.generate.side_effect = [APIError(429, {}), result]
        self.client.generate("Hello")
        self.assertEqual(self.provider.generate.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_non_retryable_failure(self):
        self.provider.generate.side_effect = APIError(400, {})
        with self.assertRaises(APIError):
            self.client.generate("Hello")
        self.provider.generate.assert_called_once()
        self.assertEqual(
            self.logger.log_call.call_args.args[0]["status"], "failed"
        )

    def test_http_503_is_retryable(self):
        response = requests.Response()
        response.status_code = 503
        self.provider.generate.side_effect = [
            requests.HTTPError(response=response),
            self.provider.generate.return_value,
        ]
        with patch("llm_client.time.sleep"):
            self.client.generate("Hello")
        self.assertEqual(self.provider.generate.call_count, 2)

    def test_logging_failure_does_not_repeat_successful_ai_call(self):
        self.logger.log_call.side_effect = ConnectionFailure("offline")
        with self.assertRaises(ConnectionFailure):
            self.client.generate("Hello")
        self.provider.generate.assert_called_once()
        self.logger.log_call.assert_called_once()

    def test_logging_failure_preserves_original_provider_error(self):
        self.provider.generate.side_effect = ValueError("no text")
        self.logger.log_call.side_effect = ConnectionFailure("offline")
        with self.assertRaisesRegex(ValueError, "no text"):
            self.client.generate("Hello")
        self.logger.log_call.assert_called_once()

    def test_empty_prompt_does_not_call_services(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.client.generate(" ")
        self.provider.generate.assert_not_called()
        self.logger.log_call.assert_not_called()

    def test_missing_usage_is_not_reported_as_zero(self):
        self.provider.generate.return_value["input_tokens"] = None
        self.client.generate("Hello")
        document = self.logger.log_call.call_args.args[0]
        self.assertIsNone(document["total_tokens"])
        self.assertIsNone(document["cost"])


class TestProviders(unittest.TestCase):
    @patch("llm_client.genai.Client")
    def test_gemini_usage_includes_thinking(self, client):
        response = client.return_value.models.generate_content.return_value
        response.text = "Cloud answer"
        response.usage_metadata = Mock(
            prompt_token_count=10, candidates_token_count=5, thoughts_token_count=3
        )
        provider = GeminiProvider("test-key")
        self.assertEqual(provider.generate("Hello"), {
            "response": "Cloud answer", "input_tokens": 10, "output_tokens": 8,
        })
        provider.close()
        client.return_value.close.assert_called_once()

    @patch("llm_client.genai.Client")
    def test_gemini_missing_usage_and_empty_text(self, client):
        response = client.return_value.models.generate_content.return_value
        response.text = "Answer"
        response.usage_metadata = None
        provider = GeminiProvider("test-key")
        self.assertIsNone(provider.generate("Hello")["input_tokens"])
        response.text = None
        with self.assertRaisesRegex(ValueError, "no text"):
            provider.generate("Hello")

    def test_missing_key(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            GeminiProvider("")

    @patch("llm_client.requests.post")
    def test_ollama_usage(self, post):
        post.return_value.json.return_value = {
            "response": "Local answer", "prompt_eval_count": 4, "eval_count": 7,
        }
        self.assertEqual(OllamaProvider().generate("Hello")["output_tokens"], 7)
        post.return_value.raise_for_status.assert_called_once()
        self.assertEqual(post.call_args.kwargs["timeout"], 120)

    @patch("llm_client.requests.post")
    def test_ollama_http_error(self, post):
        post.return_value.raise_for_status.side_effect = requests.HTTPError()
        with self.assertRaises(requests.HTTPError):
            OllamaProvider().generate("Hello")
        post.return_value.json.assert_not_called()


@patch.dict("os.environ", {
    "MONGODB_URI": "mongodb+srv://test:fake@cluster.example.mongodb.net/",
}, clear=True)
class TestMongoLogger(unittest.TestCase):
    @patch.dict("os.environ", {"MONGODB_URI": "mongodb://localhost:27017/"})
    @patch("mongo_logger.MongoClient")
    def test_local_connection_does_not_force_tls(self, client):
        logger = MongoLogger()
        self.assertEqual(client.call_args.args[0], "mongodb://localhost:27017/")
        self.assertNotIn("tlsCAFile", client.call_args.kwargs)
        client.return_value.admin.command.assert_called_once_with("ping")
        logger.close()
        client.return_value.close.assert_called_once()

    @patch.dict("os.environ", {"MONGODB_URI": "http://localhost:27017/"})
    @patch("mongo_logger.MongoClient")
    def test_rejects_non_mongodb_scheme(self, client):
        with self.assertRaisesRegex(ValueError, "MONGODB_URI"):
            MongoLogger()
        client.assert_not_called()

    @patch("mongo_logger.MongoClient")
    def test_ping_insert_and_close(self, client):
        logger = MongoLogger()
        client.return_value.admin.command.assert_called_once_with("ping")
        self.assertIn("tlsCAFile", client.call_args.kwargs)
        document = {"status": "success"}
        logger.log_call(document)
        saved = logger.collection.insert_one.call_args.args[0]
        self.assertEqual(saved["timestamp"].tzinfo, timezone.utc)
        self.assertNotIn("timestamp", document)
        logger.close()
        client.return_value.close.assert_called_once()

    @patch("mongo_logger.MongoClient")
    def test_failed_ping_closes_connection(self, client):
        client.return_value.admin.command.side_effect = ConnectionFailure()
        with self.assertRaises(ConnectionFailure):
            MongoLogger()
        client.return_value.close.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_uri(self):
        with self.assertRaisesRegex(ValueError, "MONGODB_URI"):
            MongoLogger()

    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb+srv://YOUR_DB_USER:password@YOUR_CLUSTER/",
    })
    def test_rejects_placeholders(self):
        with self.assertRaisesRegex(ValueError, "placeholders"):
            MongoLogger()


class TestConnectionCheck(unittest.TestCase):
    @patch("check_connection.load_dotenv")
    @patch("check_connection.MongoLogger")
    def test_insert_and_readback(self, logger_class, load_env):
        logger = logger_class.return_value
        inserted_id = ObjectId()
        logger.log_call.return_value = inserted_id
        logger.collection.find_one.return_value = {"_id": inserted_id}
        check_connection(write=True)
        logger.collection.find_one.assert_called_once_with({"_id": inserted_id})
        self.assertEqual(
            logger.log_call.call_args.args[0]["record_type"], "connectivity_check"
        )
        logger.close.assert_called_once()

    @patch("check_connection.load_dotenv")
    @patch("check_connection.MongoLogger")
    def test_write_failure_is_not_success(self, logger_class, load_env):
        logger = logger_class.return_value
        logger.log_call.side_effect = OperationFailure("not authorized")
        with self.assertRaises(OperationFailure):
            check_connection(write=True)
        logger.close.assert_called_once()

    @patch("check_connection.load_dotenv")
    @patch("check_connection.MongoLogger")
    def test_readback_failure_is_not_success(self, logger_class, load_env):
        logger = logger_class.return_value
        logger.collection.find_one.return_value = None
        with self.assertRaisesRegex(RuntimeError, "could not be read"):
            check_connection(write=True)
        logger.close.assert_called_once()


class TestMain(unittest.TestCase):
    @patch("main.load_dotenv")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("main.GeminiProvider")
    @patch("main.MongoLogger")
    @patch("builtins.input", side_effect=["1", "Explain AI"])
    def test_gemini_flow(self, user_input, logger_class, provider_class, load_env):
        provider_class.return_value.model = "gemini-3.8-flash"
        provider_class.return_value.generate.return_value = {
            "response": "Answer", "input_tokens": 2, "output_tokens": 3,
        }
        main()
        provider_class.return_value.generate.assert_called_once_with("Explain AI")
        logger_class.return_value.log_call.assert_called_once()
        logger_class.return_value.close.assert_called_once()
        provider_class.return_value.close.assert_called_once()

    @patch("main.load_dotenv")
    @patch("main.MongoLogger")
    @patch("builtins.input", return_value="unknown")
    def test_invalid_provider_does_not_connect(self, user_input, logger, load_env):
        with self.assertRaisesRegex(ValueError, "Invalid provider"):
            main()
        logger.assert_not_called()


if __name__ == "__main__":
    unittest.main()
