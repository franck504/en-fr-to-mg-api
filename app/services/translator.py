from app.config import Settings
from app.services.providers.base import TranslationProvider, TranslationResult
from app.services.providers.gemini_api import GeminiApiProvider
from app.services.providers.hf_seq2seq import (
    HuggingFaceSeq2SeqProvider,
    infer_model_family,
)


def build_provider(settings: Settings) -> TranslationProvider:
    if settings.provider == "gemini_api":
        return GeminiApiProvider(
            model_name=settings.gemini_model_name,
            api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature,
            thinking_budget=settings.gemini_thinking_budget,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_retries=settings.gemini_max_retries,
            retry_default_delay_seconds=settings.gemini_retry_default_delay_seconds,
        )

    if settings.provider in {"hf_seq2seq", "local_nllb", "local_m2m100"}:
        if settings.provider == "local_nllb":
            model_family = "nllb"
            provider_name = "local_nllb"
        elif settings.provider == "local_m2m100":
            model_family = "m2m100"
            provider_name = "local_m2m100"
        else:
            model_family = (
                infer_model_family(settings.hf_model_name)
                if settings.hf_model_family == "auto"
                else settings.hf_model_family
            )
            provider_name = settings.provider

        return HuggingFaceSeq2SeqProvider(
            provider_name=provider_name,
            model_name=settings.hf_model_name,
            model_family=model_family,
            cache_dir=settings.model_cache_dir,
            max_length=settings.translation_max_length,
            device=settings.hf_device,
        )

    raise ValueError(f"Unsupported provider: {settings.provider}")


class TranslationService:
    def __init__(self, provider: TranslationProvider) -> None:
        self.provider = provider

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        return self.provider.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
