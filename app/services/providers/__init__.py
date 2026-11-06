"""Translation providers."""

from app.services.providers.base import TranslationProvider, TranslationResult
from app.services.providers.hf_seq2seq import HuggingFaceSeq2SeqProvider
from app.services.providers.gemma4_local import Gemma4LocalProvider

__all__ = [
    "TranslationProvider",
    "TranslationResult",
    "HuggingFaceSeq2SeqProvider",
    "Gemma4LocalProvider",
]
