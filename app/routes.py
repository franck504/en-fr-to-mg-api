from fastapi import APIRouter, HTTPException
from app.schemas import HealthResponse, TranslateRequest, TranslateResponse
from app.services.translator import TranslationService

# Define a router to group all translation and meta endpoints
router = APIRouter()

# Service instances will be assigned or injected from main.py
# Using local imports within functions to avoid circular dependency issues
gemini_service = None
local_llm_service = None
gemma4_service = None
default_service = None

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
    from app.main import _prepare_provider_for_request
    
    # Ensure the model is ready and conflicting models are unloaded
    _prepare_provider_for_request(service.provider)
    try:
        result = service.translate(
            text=payload.text,
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
    from app.main import provider, settings
    return {
        "name": settings.app_name,
        "provider": provider.provider_name,
        "model_family": getattr(provider, "model_family", "unknown"),
        "target_language": settings.target_language,
    }

@router.get("/health", tags=["meta"])
def healthcheck():
    """Provides a global health status for the default and secondary providers."""
    from app.main import provider, gemini_provider, local_llm_provider, gemma4_provider
    return {
        "status": "ok",
        "default_provider": provider.provider_name,
        "providers": {
            "gemini": _health_from_provider(gemini_provider).model_dump(),
            "local_llm": _health_from_provider(local_llm_provider).model_dump(),
            "gemma4": _health_from_provider(gemma4_provider).model_dump(),
        },
    }

@router.get("/health/gemini", response_model=HealthResponse, tags=["meta"])
def healthcheck_gemini():
    """Check status of the Gemini API provider."""
    from app.main import gemini_provider
    return _health_from_provider(gemini_provider)

@router.get("/health/local_llm", response_model=HealthResponse, tags=["meta"])
def healthcheck_local_llm():
    """Check status of the Local LLM (HF) provider."""
    from app.main import local_llm_provider
    return _health_from_provider(local_llm_provider)

@router.get("/health/gemma4", response_model=HealthResponse, tags=["meta"])
def healthcheck_gemma4():
    """Check status of the Gemma 4 provider."""
    from app.main import gemma4_provider
    return _health_from_provider(gemma4_provider)

@router.post("/translate", response_model=TranslateResponse, tags=["translation"])
def translate(payload: TranslateRequest):
    """Translate using the default provider defined in settings."""
    from app.main import translation_service
    return _translate_with_service(translation_service, payload)

@router.post("/translate/gemini", response_model=TranslateResponse, tags=["translation"])
def translate_gemini(payload: TranslateRequest):
    """Force translation via Google Gemini API."""
    from app.main import gemini_translation_service
    return _translate_with_service(gemini_translation_service, payload)

@router.post("/translate/local_llm", response_model=TranslateResponse, tags=["translation"])
def translate_local_llm(payload: TranslateRequest):
    """Force translation via Local Hugging Face models."""
    from app.main import local_llm_translation_service
    return _translate_with_service(local_llm_translation_service, payload)

@router.post("/translate/gemma4", response_model=TranslateResponse, tags=["translation"])
def translate_gemma4(payload: TranslateRequest):
    """Force translation via local Gemma 4 model."""
    from app.main import gemma4_translation_service
    return _translate_with_service(gemma4_translation_service, payload)
