from __future__ import annotations
from collections.abc import Iterable
from app.services.providers.base import TranslationProvider, TranslationResult

# Language codes specific to NLLB models
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "mg": "plt_Latn",
}

# Frequent keywords for language scoring
FRENCH_HINTS = {
    "bonjour", "avec", "pour", "être", "dans", "nous", 
    "vous", "une", "des", "pas", "est", "merci",
}

ENGLISH_HINTS = {
    "hello", "with", "for", "the", "and", "you", 
    "we", "are", "this", "that", "please", "thank",
}

def _score_language(text: str, hints: Iterable[str]) -> int:
    """Computes a score for a language based on keyword frequency."""
    lowered = f" {text.lower()} "
    return sum(1 for hint in hints if f" {hint} " in lowered)

def detect_source_language(text: str) -> str:
    """Attempts to detect the source language (French or English)."""
    french_score = _score_language(text, FRENCH_HINTS)
    english_score = _score_language(text, ENGLISH_HINTS)

    if any(character in text.lower() for character in "àâæçéèêëîïôœùûüÿ"):
        french_score += 2

    return "fr" if french_score > english_score else "en"

class LocalNllbProvider(TranslationProvider):
    """
    Deprecated or legacy provider for NLLB specifically. 
    Note: HuggingFaceSeq2SeqProvider is preferred for general use.
    """
    provider_name = "local_nllb"

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_length: int,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Checks if tokenizer and model are loaded in memory."""
        return self._tokenizer is not None and self._model is not None

    def warmup(self) -> None:
        """Pre-loads the model for faster initial translation."""
        self._load_model()

    def _load_model(self) -> None:
        """Loads the NLLB model and tokenizer."""
        if self.is_loaded:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers library is not installed.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Executes a translation using the local NLLB model."""
        if target_lang not in NLLB_LANGUAGE_CODES:
            raise ValueError(f"Unsupported target language: {target_lang}")

        resolved_source_lang = (
            detect_source_language(text) if source_lang == "auto" else source_lang
        )
        if resolved_source_lang not in NLLB_LANGUAGE_CODES:
            raise ValueError(f"Unsupported source language: {resolved_source_lang}")

        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None

        self._tokenizer.src_lang = NLLB_LANGUAGE_CODES[resolved_source_lang]
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True)
        
        generated_tokens = self._model.generate(
            **encoded,
            forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(
                NLLB_LANGUAGE_CODES[target_lang]
            ),
            max_length=self.max_length,
        )
        translated_text = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        return TranslationResult(
            text=text,
            translated_text=translated_text,
            source_lang=resolved_source_lang,
            target_lang=target_lang,
            provider=self.provider_name,
            model_name=self.model_name,
        )
