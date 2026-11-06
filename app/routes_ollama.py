from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from app.config import get_settings
from app.schemas_compat import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaModelDetails,
    OllamaModelItem,
    OllamaTagsResponse,
)
from app.services.dispatcher import (
    execute_translation,
    extract_text_from_messages,
    resolve_service_by_model,
)
from app.services.security import verify_api_key

router = APIRouter(
    prefix="/api",
    tags=["Ollama Compatibility"],
    dependencies=[Depends(verify_api_key)],
)
settings = get_settings()

def _current_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

@router.get("/version")
def get_version():
    """Returns Ollama API version string."""
    return {"version": "0.1.48"}

@router.get("/tags", response_model=OllamaTagsResponse)
def list_tags():
    """Returns models in Ollama /api/tags format."""
    now_iso = _current_iso_timestamp()
    models = [
        OllamaModelItem(
            name=settings.hf_model_name,
            model=settings.hf_model_name,
            modified_at=now_iso,
            size=600_000_000,
            digest="sha256:nllb200distilled600m",
            details=OllamaModelDetails(
                family="seq2seq",
                parameter_size="600M",
            ),
        ),
        OllamaModelItem(
            name="nllb",
            model="nllb",
            modified_at=now_iso,
            size=600_000_000,
            digest="sha256:nllbdefault",
            details=OllamaModelDetails(
                family="seq2seq",
                parameter_size="600M",
            ),
        ),
        OllamaModelItem(
            name="gemma4",
            model="gemma4",
            modified_at=now_iso,
            size=2_000_000_000,
            digest="sha256:gemma4e2bit",
            details=OllamaModelDetails(
                family="gemma",
                parameter_size="2B",
            ),
        ),
    ]
    return OllamaTagsResponse(models=models)

@router.post("/chat", response_model=OllamaChatResponse)
def ollama_chat(request: OllamaChatRequest):
    """
    Ollama /api/chat compatibility endpoint.
    Translates chat message text to Malagasy (requires API key).
    """
    text_to_translate = extract_text_from_messages(request.messages)
    if not text_to_translate:
        raise HTTPException(
            status_code=400,
            detail="No prompt text found in user messages.",
        )

    service, model_id = resolve_service_by_model(request.model)

    try:
        result = execute_translation(
            text=text_to_translate,
            service=service,
            source_lang="auto",
            target_lang="mg",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return OllamaChatResponse(
        model=request.model or model_id,
        created_at=_current_iso_timestamp(),
        message=OllamaMessage(
            role="assistant",
            content=result.translated_text,
        ),
        done=True,
    )

@router.post("/generate", response_model=OllamaGenerateResponse)
def ollama_generate(request: OllamaGenerateRequest):
    """
    Ollama /api/generate compatibility endpoint.
    Translates raw prompt text to Malagasy (requires API key).
    """
    text_to_translate = request.prompt.strip()
    if not text_to_translate:
        raise HTTPException(
            status_code=400,
            detail="Prompt text cannot be empty.",
        )

    service, model_id = resolve_service_by_model(request.model)

    try:
        result = execute_translation(
            text=text_to_translate,
            service=service,
            source_lang="auto",
            target_lang="mg",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return OllamaGenerateResponse(
        model=request.model or model_id,
        created_at=_current_iso_timestamp(),
        response=result.translated_text,
        done=True,
    )
