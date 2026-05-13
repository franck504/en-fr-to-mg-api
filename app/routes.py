from fastapi import APIRouter, HTTPException
from app.schemas import HealthResponse, TranslateRequest, TranslateResponse
from app.services.translator import TranslationService

# On crée un router pour regrouper les endpoints
router = APIRouter()

# Ces services seront injectés ou importés depuis main.py
# Pour simplifier le refactoring immédiat, on définit des variables globales 
# qui seront assignées lors de l'importation.
gemini_service = None
local_llm_service = None
gemma4_service = None
default_service = None

def _health_from_provider(provider_instance) -> HealthResponse:
    """Génère un modèle de santé à partir d'une instance de provider."""
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
    """Exécute la traduction en gérant les déchargements de modèles si nécessaire."""
    from app.main import _prepare_provider_for_request
    
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
    from app.main import provider, settings
    return {
        "name": settings.app_name,
        "provider": provider.provider_name,
        "model_family": getattr(provider, "model_family", "unknown"),
        "target_language": settings.target_language,
    }

@router.get("/health", tags=["meta"])
def healthcheck():
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
    from app.main import gemini_provider
    return _health_from_provider(gemini_provider)

@router.get("/health/local_llm", response_model=HealthResponse, tags=["meta"])
def healthcheck_local_llm():
    from app.main import local_llm_provider
    return _health_from_provider(local_llm_provider)

@router.get("/health/gemma4", response_model=HealthResponse, tags=["meta"])
def healthcheck_gemma4():
    from app.main import gemma4_provider
    return _health_from_provider(gemma4_provider)

@router.post("/translate", response_model=TranslateResponse, tags=["translation"])
def translate(payload: TranslateRequest):
    from app.main import translation_service
    return _translate_with_service(translation_service, payload)

@router.post("/translate/gemini", response_model=TranslateResponse, tags=["translation"])
def translate_gemini(payload: TranslateRequest):
    from app.main import gemini_translation_service
    return _translate_with_service(gemini_translation_service, payload)

@router.post("/translate/local_llm", response_model=TranslateResponse, tags=["translation"])
def translate_local_llm(payload: TranslateRequest):
    from app.main import local_llm_translation_service
    return _translate_with_service(local_llm_translation_service, payload)

@router.post("/translate/gemma4", response_model=TranslateResponse, tags=["translation"])
def translate_gemma4(payload: TranslateRequest):
    from app.main import gemma4_translation_service
    return _translate_with_service(gemma4_translation_service, payload)
