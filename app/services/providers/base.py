from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationResult:
    text: str
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str
    model_name: str


class TranslationProvider(ABC):
    provider_name: str
    model_name: str

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        return None

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        raise NotImplementedError
