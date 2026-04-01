from app.config import Settings
from app.services.providers.base import TranslationProvider, TranslationResult
from app.services.providers.local_nllb import LocalNllbProvider


def build_provider(settings: Settings) -> TranslationProvider:
    if settings.provider == "local_nllb":
        return LocalNllbProvider(
            model_name=settings.hf_model_name,
            cache_dir=settings.model_cache_dir,
            max_length=settings.translation_max_length,
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
