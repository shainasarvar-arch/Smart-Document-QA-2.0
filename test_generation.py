import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai.errors import ClientError, ServerError

from app import (
    FALLBACK_GENERATION_MODEL,
    GENERATION_MODEL,
    DocumentChunk,
    answer_was_not_found,
    answer_question,
)


class _FakeModels:
    def __init__(self, primary_error: Exception) -> None:
        self.primary_error = primary_error
        self.models_called: list[str] = []

    def generate_content(self, *, model: str, contents: str) -> SimpleNamespace:
        self.models_called.append(model)
        if model == GENERATION_MODEL:
            raise self.primary_error
        return SimpleNamespace(text="fallback answer")


class GenerationFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = [(DocumentChunk("source text", "source.pdf", 1), 1.0)]

    def test_retries_transient_errors_then_uses_fallback(self) -> None:
        models = _FakeModels(ServerError(503, {"error": {"code": 503}}))
        client = SimpleNamespace(models=models)

        with patch("app.time.sleep") as sleep:
            result = answer_question("question", self.sources, client)

        self.assertEqual(result, ("fallback answer", FALLBACK_GENERATION_MODEL))
        self.assertEqual(models.models_called, [GENERATION_MODEL] * 3 + [FALLBACK_GENERATION_MODEL])
        self.assertEqual(sleep.call_count, 2)

    def test_does_not_retry_or_fallback_on_client_error(self) -> None:
        models = _FakeModels(ClientError(400, {"error": {"code": 400}}))
        client = SimpleNamespace(models=models)

        with patch("app.time.sleep") as sleep:
            with self.assertRaises(ClientError):
                answer_question("question", self.sources, client)

        self.assertEqual(models.models_called, [GENERATION_MODEL])
        sleep.assert_not_called()

    def test_not_found_answers_hide_sources_but_grounded_answers_do_not(self) -> None:
        self.assertTrue(answer_was_not_found("This was not found in the uploaded documents."))
        self.assertTrue(answer_was_not_found("I couldn't find this in the uploaded documents."))
        self.assertFalse(answer_was_not_found("The report states that sales increased by 12%."))


if __name__ == "__main__":
    unittest.main()
