from __future__ import annotations
import gc
from dataclasses import dataclass
from typing import Any
from app.services.language_utils import detect_source_language
from app.services.providers.base import TranslationProvider, TranslationResult

@dataclass(frozen=True)
class ModelFamilySpec:
    """Defines the specific requirements and language codes for a model family."""
    family_name: str
    language_codes: dict[str, str]
    target_token_method: str

# Specifications for supported model families (NLLB and M2M100)
MODEL_FAMILY_SPECS: dict[str, ModelFamilySpec] = {
    "nllb": ModelFamilySpec(
        family_name="nllb",
        language_codes={
            "en": "eng_Latn",
            "fr": "fra_Latn",
            "mg": "plt_Latn",
        },
        target_token_method="nllb",
    ),
    "m2m100": ModelFamilySpec(
        family_name="m2m100",
        language_codes={
            "en": "en",
            "fr": "fr",
            "mg": "mg",
        },
        target_token_method="m2m100",
    ),
}

def infer_model_family(model_name: str) -> str:
    """Attempts to guess the model family from the model string name."""
    lowered = model_name.lower()
    if "nllb" in lowered:
        return "nllb"
    if "m2m100" in lowered:
        return "m2m100"
    raise ValueError(
        "Unable to infer the Hugging Face model family from HF_MODEL_NAME. "
        "Please set HF_MODEL_FAMILY explicitly."
    )

class HuggingFaceSeq2SeqProvider(TranslationProvider):
    """
    Provider for sequence-to-sequence translation models hosted on Hugging Face.
    Supports NLLB and M2M100 architectures.
    """
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        model_family: str,
        cache_dir: str,
        max_length: int,
        device: str,
    ) -> None:
        if model_family not in MODEL_FAMILY_SPECS:
            supported = ", ".join(sorted(MODEL_FAMILY_SPECS))
            raise ValueError(f"Unsupported HF model family: {model_family}. Use one of: {supported}")

        self.provider_name = provider_name
        self.model_name = model_name
        self.model_family = model_family
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.device = device
        self.runtime_device = "uninitialized"
        self.runtime_dtype = "uninitialized"
        self._spec = MODEL_FAMILY_SPECS[model_family]
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Checks if the tokenizer and model are loaded."""
        return self._tokenizer is not None and self._model is not None

    def warmup(self) -> None:
        """Loads the model to ensure it is ready for immediate inference."""
        self._load_model()

    def unload(self) -> None:
        """Clears the model from memory and releases GPU cache."""
        if not self.is_loaded:
            return

        try:
            import torch
        except ImportError:
            torch = None

        if self._model is not None and self.runtime_device == "cuda":
            try:
                self._model.cpu()
            except Exception:
                pass

        self._tokenizer = None
        self._model = None
        self.runtime_device = "uninitialized"
        self.runtime_dtype = "uninitialized"

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def _resolve_device(self) -> str:
        """Determines whether to use CPU or CUDA based on availability and settings."""
        if self.device not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("HF_DEVICE must be one of: auto, cpu, cuda")

        if self.device != "auto":
            if self.device == "cuda":
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError("torch is not installed.") from exc

                if not torch.cuda.is_available():
                    raise RuntimeError("HF_DEVICE set to cuda but no GPU found.")
            return self.device

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is not installed.") from exc

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_torch_dtype(self, runtime_device: str):
        """Resolves the torch data type (FP16 for CUDA, FP32 for CPU)."""
        import torch
        return torch.float16 if runtime_device == "cuda" else torch.float32

    def _load_model(self) -> None:
        """Loads the model and tokenizer from the local cache or Hugging Face Hub."""
        if self.is_loaded:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch must be installed.") from exc

        runtime_device = self._resolve_device()
        torch_dtype = self._resolve_torch_dtype(runtime_device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        if runtime_device != "cpu":
            self._model.to(runtime_device)

        self.runtime_device = runtime_device
        self.runtime_dtype = str(torch_dtype).replace("torch.", "")

    def _resolve_source_lang(self, text: str, source_lang: str) -> str:
        """Identifies the source language, using detection if auto-mode is active."""
        resolved_source_lang = detect_source_language(text) if source_lang == "auto" else source_lang
        if resolved_source_lang not in self._spec.language_codes:
            raise ValueError(f"Unsupported source language: {resolved_source_lang}")
        return resolved_source_lang

    def _resolve_forced_bos_token_id(self, tokenizer: Any, target_lang: str) -> int:
        """Retrieves the specific token ID required to force the target language output."""
        target_code = self._spec.language_codes[target_lang]
        if self._spec.target_token_method == "nllb":
            if hasattr(tokenizer, "lang_code_to_id"):
                return tokenizer.lang_code_to_id[target_code]
            return tokenizer.convert_tokens_to_ids(target_code)
        if self._spec.target_token_method == "m2m100":
            return tokenizer.get_lang_id(target_code)
        raise ValueError(f"Unsupported target token method: {self._spec.target_token_method}")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Executes the translation pipeline."""
        if target_lang not in self._spec.language_codes:
            raise ValueError(f"Unsupported target language: {target_lang}")

        resolved_source_lang = self._resolve_source_lang(text, source_lang)
        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None

        source_code = self._spec.language_codes[resolved_source_lang]
        self._tokenizer.src_lang = source_code
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True)

        if self.runtime_device != "cpu":
            encoded = {key: value.to(self.runtime_device) for key, value in encoded.items()}

        generated_tokens = self._model.generate(
            **encoded,
            forced_bos_token_id=self._resolve_forced_bos_token_id(self._tokenizer, target_lang),
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
