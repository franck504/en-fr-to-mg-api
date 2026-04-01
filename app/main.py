from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.schemas import HealthResponse, TranslateRequest, TranslateResponse
from app.services.translator import TranslationService, build_provider

settings = get_settings()
provider = build_provider(settings)
translation_service = TranslationService(provider)


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
        "target_language": settings.target_language,
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=provider.provider_name,
        model_name=provider.model_name,
        model_loaded=provider.is_loaded,
    )


@app.post("/translate", response_model=TranslateResponse, tags=["translation"])
def translate(payload: TranslateRequest) -> TranslateResponse:
    try:
        result = translation_service.translate(
            text=payload.text,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TranslateResponse(**result.__dict__)
