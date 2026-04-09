from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.schemas import HealthResponse, TranslateRequest, TranslateResponse
from app.services.translator import (
    TranslationService,
    build_gemma4_provider,
    build_gemini_provider,
    build_local_llm_provider,
)

settings = get_settings()
gemini_provider = build_gemini_provider(settings)
gemini_translation_service = TranslationService(gemini_provider)
local_llm_provider = build_local_llm_provider(settings)
local_llm_translation_service = TranslationService(local_llm_provider)
gemma4_provider = build_gemma4_provider(settings)
gemma4_translation_service = TranslationService(gemma4_provider)


def _resolve_default_provider():
    if settings.provider == "gemini_api":
        return gemini_provider

    if settings.provider == "gemma4":
        return gemma4_provider

    if settings.provider in {"hf_seq2seq", "local_llm", "local_nllb", "local_m2m100"}:
        return local_llm_provider

    raise ValueError(f"Unsupported provider: {settings.provider}")


provider = _resolve_default_provider()
translation_service = TranslationService(provider)


def _health_from_provider(provider_instance) -> HealthResponse:
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
    _prepare_provider_for_request(service.provider)
    try:
        result = service.translate(
            text=payload.text,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TranslateResponse(**result.__dict__)


def _unload_provider(provider_instance) -> None:
    provider_instance.unload()


def _prepare_provider_for_request(provider_instance) -> None:
    model_family = getattr(provider_instance, "model_family", "unknown")

    if model_family in {"nllb", "m2m100"}:
        _unload_provider(gemma4_provider)
    elif model_family == "gemma4":
        _unload_provider(local_llm_provider)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.load_model_on_startup:
        provider.warmup()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend de traduction EN/FR vers MG basé sur un modèle open source.",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def read_root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "provider": provider.provider_name,
        "model_family": getattr(provider, "model_family", "unknown"),
        "target_language": settings.target_language,
    }


@app.get("/health", tags=["meta"])
def healthcheck() -> dict[str, object]:
    return {
        "status": "ok",
        "default_provider": provider.provider_name,
        "providers": {
            "gemini": _health_from_provider(gemini_provider).model_dump(),
            "local_llm": _health_from_provider(local_llm_provider).model_dump(),
            "gemma4": _health_from_provider(gemma4_provider).model_dump(),
        },
    }


@app.get("/health/gemini", response_model=HealthResponse, tags=["meta"])
def healthcheck_gemini() -> HealthResponse:
    return _health_from_provider(gemini_provider)


@app.get("/health/local_llm", response_model=HealthResponse, tags=["meta"])
def healthcheck_local_llm() -> HealthResponse:
    return _health_from_provider(local_llm_provider)


@app.get("/health/gemma4", response_model=HealthResponse, tags=["meta"])
def healthcheck_gemma4() -> HealthResponse:
    return _health_from_provider(gemma4_provider)


@app.post("/translate", response_model=TranslateResponse, tags=["translation"])
def translate(payload: TranslateRequest) -> TranslateResponse:
    return _translate_with_service(translation_service, payload)


@app.post("/translate/gemini", response_model=TranslateResponse, tags=["translation"])
def translate_gemini(payload: TranslateRequest) -> TranslateResponse:
    return _translate_with_service(gemini_translation_service, payload)


@app.post("/translate/local_llm", response_model=TranslateResponse, tags=["translation"])
def translate_local_llm(payload: TranslateRequest) -> TranslateResponse:
    return _translate_with_service(local_llm_translation_service, payload)


@app.post("/translate/gemma4", response_model=TranslateResponse, tags=["translation"])
def translate_gemma4(payload: TranslateRequest) -> TranslateResponse:
    return _translate_with_service(gemma4_translation_service, payload)
