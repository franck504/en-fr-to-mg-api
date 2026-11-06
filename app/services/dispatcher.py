from __future__ import annotations

import re
from typing import Any
from app.config import get_settings
from app.services.providers.base import TranslationProvider, TranslationResult
from app.services.translator import (
    TranslationService,
    build_gemma4_provider,
    build_local_llm_provider,
)

settings = get_settings()

# Initialize local providers
local_llm_provider = build_local_llm_provider(settings)
local_llm_translation_service = TranslationService(local_llm_provider)

gemma4_provider = build_gemma4_provider(settings)
gemma4_translation_service = TranslationService(gemma4_provider)

def resolve_default_provider() -> TranslationProvider:
    """Determines which provider is configured as default."""
    if settings.provider == "gemma4":
        return gemma4_provider
    return local_llm_provider

default_provider = resolve_default_provider()
default_translation_service = TranslationService(default_provider)

def unload_provider(provider_instance: TranslationProvider) -> None:
    """Unloads a model from memory to free up GPU/RAM resources."""
    provider_instance.unload()

def prepare_provider_for_request(provider_instance: TranslationProvider) -> None:
    """
    Manages model coexistence in memory for Colab/Kaggle environments.
    Prevents conflicting large models from occupying GPU VRAM simultaneously.
    """
    model_family = getattr(provider_instance, "model_family", "unknown")
    if model_family in {"nllb", "m2m100"}:
        unload_provider(gemma4_provider)
    elif model_family == "gemma4":
        unload_provider(local_llm_provider)

def resolve_service_by_model(model_name: str | None) -> tuple[TranslationService, str]:
    """
    Resolves client model strings (OpenAI, Anthropic, Ollama, etc.) to the appropriate local engine.
    """
    if not model_name or model_name.strip().lower() in {"default", "auto", "latest"}:
        return default_translation_service, default_provider.model_name

    lowered = model_name.strip().lower()
    if "gemma" in lowered:
        return gemma4_translation_service, gemma4_provider.model_name

    # Default to local seq2seq (NLLB / M2M100)
    return local_llm_translation_service, local_llm_provider.model_name

def extract_text_from_messages(messages: list[Any]) -> str:
    """
    Extracts the user text to translate from a list of chat messages.
    Finds the last message from the 'user' role.
    """
    if not messages:
        return ""

    user_text = ""
    # Find the last user message
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "user":
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif hasattr(part, "text"):
                        parts.append(getattr(part, "text", ""))
                user_text = " ".join(parts)
            break

    # If no user message was found, take the content of the very last message
    if not user_text and messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "content", None) or (last_msg.get("content") if isinstance(last_msg, dict) else None)
        if isinstance(content, str):
            user_text = content

    return user_text.strip()

def execute_translation(
    text: str,
    service: TranslationService,
    source_lang: str = "auto",
    target_lang: str = "mg",
) -> TranslationResult:
    """
    Safely executes translation on the resolved service with resource preparation.
    """
    prepare_provider_for_request(service.provider)
    return service.translate(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
    )
