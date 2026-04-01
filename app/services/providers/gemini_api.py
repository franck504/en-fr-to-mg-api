from __future__ import annotations

import os

from app.services.language_utils import detect_source_language
from app.services.providers.base import TranslationProvider, TranslationResult


LANGUAGE_LABELS = {
    "en": "English",
    "fr": "French",
    "mg": "Malagasy",
}


class GeminiApiProvider(TranslationProvider):
    provider_name = "gemini_api"

    def __init__(
        self,
        model_name: str,
        api_key: str | None,
        temperature: float,
        thinking_budget: int,
        timeout_seconds: float,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.timeout_seconds = timeout_seconds
        self.model_family = "gemini"
        self.runtime_device = "api"
        self.runtime_dtype = "n/a"
        self._client = None

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def warmup(self) -> None:
        self._get_client()

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to the environment before starting the Gemini provider."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Install dependencies before starting the Gemini provider."
            ) from exc

        http_options = types.HttpOptions(timeout=self.timeout_seconds)
        self._client = genai.Client(api_key=self.api_key, http_options=http_options)
        return self._client

    def _build_contents(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        source_label = LANGUAGE_LABELS[source_lang]
        target_label = LANGUAGE_LABELS[target_lang]
        return (
            f"Translate the following text from {source_label} to {target_label}.\n"
            "Return only the translated text.\n"
            "Do not explain, do not summarize, do not add markdown, and do not omit medical terms.\n"
            "Preserve all sentences and preserve the meaning as faithfully as possible.\n\n"
            f"Text:\n{text}"
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        if target_lang != "mg":
            raise ValueError(f"Unsupported target language: {target_lang}")

        resolved_source_lang = detect_source_language(text) if source_lang == "auto" else source_lang
        if resolved_source_lang not in {"en", "fr"}:
            raise ValueError(f"Unsupported source language: {resolved_source_lang}")

        client = self._get_client()

        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Install dependencies before starting the Gemini provider."
            ) from exc

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=self._build_contents(text, resolved_source_lang, target_lang),
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a professional translation engine specialized in faithful English/French "
                        "to Malagasy translation. Return only the final translation."
                    ),
                    temperature=self.temperature,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        translated_text = (response.text or "").strip()
        if not translated_text:
            raise RuntimeError("Gemini returned an empty translation.")

        return TranslationResult(
            text=text,
            translated_text=translated_text,
            source_lang=resolved_source_lang,
            target_lang=target_lang,
            provider=self.provider_name,
            model_name=self.model_name,
        )
