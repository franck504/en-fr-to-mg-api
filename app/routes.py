from fastapi import APIRouter, Depends, HTTPException
from app.config import get_settings
from app.schemas import HealthResponse, TranslateRequest, TranslateResponse
from app.services.dispatcher import (
    default_provider,
    default_translation_service,
    execute_translation,
    gemma4_provider,
    gemma4_translation_service,
    local_llm_provider,
    local_llm_translation_service,
)
from app.services.security import verify_api_key
from app.services.translator import TranslationService

router = APIRouter()
settings = get_settings()

def _health_from_provider(provider_instance) -> HealthResponse:
    """Helper to build a HealthResponse from a provider instance."""
    return HealthResponse(
        status="ok",
        provider=provider_instance.provider_name,
        model_name=provider_instance.model_name,
        model_family=getattr(provider_instance, "model_family", "unknown"),
        model_loaded=provider_instance.is_loaded,
        runtime_device=getattr(provider_instance, "runtime_device", "unknown"),
        runtime_dtype=getattr(provider_instance, "runtime_dtype", "unknown"),
    )

def _translate_with_service(
    service: TranslationService,
    payload: TranslateRequest,
) -> TranslateResponse:
    """Handles the translation process and manages runtime errors/retries."""
    try:
        result = execute_translation(
            text=payload.text,
            service=service,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return TranslateResponse(**result.__dict__)

@router.get("/", tags=["meta"])
def read_root():
    """Returns basic service metadata and current default provider info."""
    return {
        "name": settings.app_name,
        "environment": settings.app_env,
        "default_provider": default_provider.provider_name,
        "model_family": getattr(default_provider, "model_family", "unknown"),
        "target_language": settings.target_language,
        "authentication": "API Key required via 'Authorization: Bearer <API_KEY>' or 'X-API-Key: <API_KEY>'",
        "compatible_apis": [
            "OpenAI (/v1/chat/completions, /v1/models)",
            "Anthropic (/v1/messages)",
            "Ollama (/api/chat, /api/generate, /api/tags)",
            "Native REST (/translate, /translate/local_llm, /translate/gemma4)",
        ],
    }

@router.get("/health", tags=["meta"])
def healthcheck():
    """Provides a global health status for the local providers."""
    return {
        "status": "ok",
        "default_provider": default_provider.provider_name,
        "providers": {
            "local_llm": _health_from_provider(local_llm_provider).model_dump(),
            "gemma4": _health_from_provider(gemma4_provider).model_dump(),
        },
    }

@router.get("/health/local_llm", response_model=HealthResponse, tags=["meta"])
def healthcheck_local_llm():
    """Check status of the Local LLM (HF Seq2Seq) provider."""
    return _health_from_provider(local_llm_provider)

@router.get("/health/gemma4", response_model=HealthResponse, tags=["meta"])
def healthcheck_gemma4():
    """Check status of the Gemma 4 provider."""
    return _health_from_provider(gemma4_provider)

@router.post(
    "/translate",
    response_model=TranslateResponse,
    tags=["translation"],
    dependencies=[Depends(verify_api_key)],
)
def translate(payload: TranslateRequest):
    """Translate using the default local provider defined in settings (requires API key)."""
    return _translate_with_service(default_translation_service, payload)

@router.post(
    "/translate/local_llm",
    response_model=TranslateResponse,
    tags=["translation"],
    dependencies=[Depends(verify_api_key)],
)
def translate_local_llm(payload: TranslateRequest):
    """Force translation via Local Hugging Face Seq2Seq model (requires API key)."""
    return _translate_with_service(local_llm_translation_service, payload)

@router.post(
    "/translate/gemma4",
    response_model=TranslateResponse,
    tags=["translation"],
    dependencies=[Depends(verify_api_key)],
)
def translate_gemma4(payload: TranslateRequest):
    """Force translation via local Gemma 4 model (requires API key)."""
    return _translate_with_service(gemma4_translation_service, payload)
