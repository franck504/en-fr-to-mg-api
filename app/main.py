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

# Chargement de la configuration
settings = get_settings()

# Initialisation des providers et des services de traduction
# Ces instances sont partagées à travers l'application.
gemini_provider = build_gemini_provider(settings)
gemini_translation_service = TranslationService(gemini_provider)

local_llm_provider = build_local_llm_provider(settings)
local_llm_translation_service = TranslationService(local_llm_provider)

gemma4_provider = build_gemma4_provider(settings)
gemma4_translation_service = TranslationService(gemma4_provider)

def _resolve_default_provider():
    """
    Détermine le provider à utiliser par défaut en fonction de la configuration.
    """
    if settings.provider == "gemini_api":
        return gemini_provider
    if settings.provider == "gemma4":
        return gemma4_provider
    if settings.provider in {"hf_seq2seq", "local_llm", "local_nllb", "local_m2m100"}:
        return local_llm_provider
    raise ValueError(f"Le provider '{settings.provider}' n'est pas pris en charge.")

# Provider et service par défaut
provider = _resolve_default_provider()
translation_service = TranslationService(provider)

def _unload_provider(provider_instance) -> None:
    """Décharge un modèle de la mémoire."""
    provider_instance.unload()

def _prepare_provider_for_request(provider_instance) -> None:
    """
    Gère la coexistence des modèles en mémoire. 
    Certains modèles lourds ne peuvent pas résider simultanément sur le même GPU.
    """
    model_family = getattr(provider_instance, "model_family", "unknown")
    if model_family in {"nllb", "m2m100"}:
        _unload_provider(gemma4_provider)
    elif model_family == "gemma4":
        _unload_provider(local_llm_provider)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Gestion du cycle de vie de l'application.
    Permet de pré-charger les modèles au démarrage si configuré.
    """
    if settings.load_model_on_startup:
        provider.warmup()
    yield

# Création de l'application FastAPI
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Service de traduction EN/FR vers Malgache utilisant des modèles IA.",
    lifespan=lifespan,
)

# Inclusion des routes
app.include_router(router)
