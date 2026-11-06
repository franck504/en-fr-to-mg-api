import unittest
from unittest.mock import MagicMock, patch
from app.services.language_utils import detect_source_language
from app.services.providers.base import TranslationResult
from app.services.providers.hf_seq2seq import MODEL_FAMILY_SPECS, infer_model_family
from app.services.dispatcher import extract_text_from_messages, resolve_service_by_model

try:
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials
    from app.services.security import verify_api_key
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class TestLanguageDetection(unittest.TestCase):
    def test_detect_french(self):
        self.assertEqual(detect_source_language("Bonjour comment allez-vous ?"), "fr")
        self.assertEqual(detect_source_language("Appliquez la crème deux fois par jour."), "fr")
        self.assertEqual(detect_source_language("Le médecin est arrivé avec nous."), "fr")

    def test_detect_english(self):
        self.assertEqual(detect_source_language("Hello how are you today?"), "en")
        self.assertEqual(detect_source_language("Please consult a doctor promptly."), "en")
        self.assertEqual(detect_source_language("Good morning with all the team."), "en")


class TestMessageExtraction(unittest.TestCase):
    def test_extract_from_dict_messages(self):
        messages = [
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        self.assertEqual(extract_text_from_messages(messages), "Hello, how are you?")

    def test_extract_last_user_message(self):
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Malgache response"},
            {"role": "user", "content": "Second message"},
        ]
        self.assertEqual(extract_text_from_messages(messages), "Second message")

    def test_extract_from_multipart_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please translate"},
                    {"type": "text", "text": "this text."},
                ],
            }
        ]
        self.assertEqual(extract_text_from_messages(messages), "Please translate this text.")

    def test_extract_empty_messages(self):
        self.assertEqual(extract_text_from_messages([]), "")


class TestModelResolution(unittest.TestCase):
    def test_resolve_gemma(self):
        service, model_id = resolve_service_by_model("gemma4")
        self.assertIn("gemma", model_id.lower())

        service, model_id = resolve_service_by_model("google/gemma-4-E2B-it")
        self.assertIn("gemma", model_id.lower())

    def test_resolve_nllb_or_default(self):
        service, model_id = resolve_service_by_model("nllb")
        self.assertIsNotNone(service)

        service, model_id = resolve_service_by_model("default")
        self.assertIsNotNone(service)

        service, model_id = resolve_service_by_model("gpt-4o")
        self.assertIsNotNone(service)


class TestSeq2SeqSpecs(unittest.TestCase):
    def test_nllb_spec(self):
        spec = MODEL_FAMILY_SPECS["nllb"]
        self.assertEqual(spec.language_codes["en"], "eng_Latn")
        self.assertEqual(spec.language_codes["fr"], "fra_Latn")
        self.assertEqual(spec.language_codes["mg"], "plt_Latn")

    def test_m2m100_spec(self):
        spec = MODEL_FAMILY_SPECS["m2m100"]
        self.assertEqual(spec.language_codes["en"], "en")
        self.assertEqual(spec.language_codes["fr"], "fr")
        self.assertEqual(spec.language_codes["mg"], "mg")

    def test_infer_family(self):
        self.assertEqual(infer_model_family("facebook/nllb-200-distilled-600M"), "nllb")
        self.assertEqual(infer_model_family("facebook/m2m100_1.2B"), "m2m100")

    def test_translation_result(self):
        res = TranslationResult(
            text="Hello",
            translated_text="Salama",
            source_lang="en",
            target_lang="mg",
            provider="local_llm",
            model_name="facebook/nllb-200-distilled-600M",
        )
        self.assertEqual(res.translated_text, "Salama")
        self.assertEqual(res.target_lang, "mg")


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed in the current environment")
class TestSecurity(unittest.TestCase):
    @patch("app.services.security.get_settings")
    def test_verify_api_key_valid_bearer(self, mock_settings):
        mock_settings.return_value = MagicMock(api_key="secret123")
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret123")
        token = verify_api_key(bearer=credentials, header_key=None)
        self.assertEqual(token, "secret123")

    @patch("app.services.security.get_settings")
    def test_verify_api_key_valid_header(self, mock_settings):
        mock_settings.return_value = MagicMock(api_key="secret123")
        token = verify_api_key(bearer=None, header_key="secret123")
        self.assertEqual(token, "secret123")

    @patch("app.services.security.get_settings")
    def test_verify_api_key_invalid(self, mock_settings):
        mock_settings.return_value = MagicMock(api_key="secret123")
        with self.assertRaises(HTTPException) as ctx:
            verify_api_key(bearer=None, header_key="wrong_key")
        self.assertEqual(ctx.exception.status_code, 401)

    @patch("app.services.security.get_settings")
    def test_verify_api_key_unconfigured_server(self, mock_settings):
        mock_settings.return_value = MagicMock(api_key=None)
        with self.assertRaises(HTTPException) as ctx:
            verify_api_key(bearer=None, header_key="some_key")
        self.assertEqual(ctx.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
