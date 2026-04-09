from __future__ import annotations

import gc

from app.services.language_utils import detect_source_language
from app.services.providers.base import TranslationProvider, TranslationResult


SUPPORTED_SOURCE_LANGUAGES = {"en", "fr"}


class Gemma4LocalProvider(TranslationProvider):
    provider_name = "gemma4"
    model_family = "gemma4"

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        device: str,
        max_new_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.runtime_device = "uninitialized"
        self.runtime_dtype = "uninitialized"
        self._processor = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def warmup(self) -> None:
        self._load_model()

    def unload(self) -> None:
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

        self._processor = None
        self._model = None
        self.runtime_device = "uninitialized"
        self.runtime_dtype = "uninitialized"

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def _resolve_device(self) -> str:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("GEMMA4_DEVICE must be one of: auto, cpu, cuda")

        if self.device != "auto":
            if self.device == "cuda":
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError(
                        "torch is not installed. Install dependencies before starting the service."
                    ) from exc

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "GEMMA4_DEVICE is set to cuda but no CUDA device is available. "
                        "Use GEMMA4_DEVICE=auto or GEMMA4_DEVICE=cpu."
                    )
            return self.device

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is not installed. Install dependencies before starting the service."
            ) from exc

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_torch_dtype(self, runtime_device: str):
        import torch

        if runtime_device == "cuda":
            return torch.float16
        return torch.float32

    def _load_model(self) -> None:
        if self.is_loaded:
            return

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 4 requires a recent transformers install. Upgrade requirements before use."
            ) from exc

        runtime_device = self._resolve_device()
        torch_dtype = self._resolve_torch_dtype(runtime_device)
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }

        if runtime_device == "cuda":
            model_kwargs["device_map"] = "auto"

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            padding_side="left",
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        if runtime_device == "cpu":
            self._model.to("cpu")

        self.runtime_device = runtime_device
        self.runtime_dtype = str(torch_dtype).replace("torch.", "")

    def _resolve_source_lang(self, text: str, source_lang: str) -> str:
        resolved_source_lang = detect_source_language(text) if source_lang == "auto" else source_lang
        if resolved_source_lang not in SUPPORTED_SOURCE_LANGUAGES:
            raise ValueError(f"Unsupported source language: {resolved_source_lang}")
        return resolved_source_lang

    def _build_messages(self, text: str, source_lang: str) -> list[dict[str, object]]:
        source_label = "English" if source_lang == "en" else "French"
        instruction = (
            f"Translate the following {source_label} text into Malagasy. "
            "Return only the translated Malagasy text. "
            "Do not explain your answer. Preserve important medical meaning and terminology."
        )

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a precise translation engine.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{instruction}\n\n{source_label}: {text}\nMalagasy:",
                    }
                ],
            },
        ]

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        if target_lang != "mg":
            raise ValueError(f"Unsupported target language: {target_lang}")

        resolved_source_lang = self._resolve_source_lang(text, source_lang)
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        messages = self._build_messages(text, resolved_source_lang)
        encoded = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        encoded = encoded.to(self._model.device)

        generated_tokens = self._model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        input_length = encoded["input_ids"].shape[-1]
        translated_text = self._processor.batch_decode(
            generated_tokens[:, input_length:],
            skip_special_tokens=True,
        )[0].strip()

        return TranslationResult(
            text=text,
            translated_text=translated_text,
            source_lang=resolved_source_lang,
            target_lang=target_lang,
            provider=self.provider_name,
            model_name=self.model_name,
        )
