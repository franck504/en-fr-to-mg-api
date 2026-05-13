from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import get_settings
from app.services.translator import (
    TranslationService,
    build_gemma4_provider,
    build_gemini_provider,
    build_local_llm_provider,
)
from app.routes import router

# Load application settings
settings = get_settings()

# Initialize providers and translation services
# These instances are shared across the application.
gemini_provider = build_gemini_provider(settings)
gemini_translation_service = TranslationService(gemini_provider)

local_llm_provider = build_local_llm_provider(settings)
local_llm_translation_service = TranslationService(local_llm_provider)

gemma4_provider = build_gemma4_provider(settings)
gemma4_translation_service = TranslationService(gemma4_provider)

def _resolve_default_provider():
    """
    Determines which provider to use as the default based on settings.
    """
    if settings.provider == "gemini_api":
        return gemini_provider
    if settings.provider == "gemma4":
        return gemma4_provider
    if settings.provider in {"hf_seq2seq", "local_llm", "local_nllb", "local_m2m100"}:
        return local_llm_provider
    raise ValueError(f"Provider '{settings.provider}' is not supported.")

# Setup default provider and service
provider = _resolve_default_provider()
translation_service = TranslationService(provider)

def _unload_provider(provider_instance) -> None:
    """Unloads a model from memory to free up resources."""
    provider_instance.unload()

def _prepare_provider_for_request(provider_instance) -> None:
    """
    Manages model coexistence in memory. 
    Prevents conflicting large models from occupying the same GPU resources simultaneously.
    """
    model_family = getattr(provider_instance, "model_family", "unknown")
    if model_family in {"nllb", "m2m100"}:
        _unload_provider(gemma4_provider)
    elif model_family == "gemma4":
        _unload_provider(local_llm_provider)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Handles application lifecycle events.
    Optionally warms up the default provider if configured to load on startup.
    """
    if settings.load_model_on_startup:
        provider.warmup()
    yield

# Create the FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Translation service for EN/FR to Malagasy using various AI models.",
    lifespan=lifespan,
)

# Attach routes to the application
app.include_router(router)
